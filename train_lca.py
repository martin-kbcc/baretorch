import argparse
import glob
import logging
import os
import random
import subprocess
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.serialization
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from baretorch import (
    BareTorchConfig,
    BareTorchForCausalLM,
    CSLRADCSTTTTransformerConfig,
    CSLRADCSTTTTransformerForCausalLM,
    CSLRADConfig,
    CSLRADForCausalLM,
    CSLRADTransformerConfig,
    CSLRADTransformerForCausalLM,
    CSTTTConfig,
    CSTTTForCausalLM,
    CSTTTTransformerConfig,
    CSTTTTransformerForCausalLM,
    TransformerConfig,
    TransformerForCausalLM,
)
from baretorch.modeling.transformer import RotaryEmbedding

# Shared memory & Tensor Precision Setup
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

# NumPy Deserialization Allowlist (PyTorch 2.6+)
safe_numpy_types = [np.dtype, np.ndarray]
for mod_path in [
    "numpy._core.multiarray",
    "numpy.core.multiarray",
    "numpy._core.numerictypes",
]:
    try:
        mod = __import__(mod_path, fromlist=["scalar", "_reconstruct"])
        if hasattr(mod, "scalar"):
            safe_numpy_types.append(mod.scalar)
        if hasattr(mod, "_reconstruct"):
            safe_numpy_types.append(mod._reconstruct)
    except (ImportError, AttributeError):
        pass
try:
    torch.serialization.add_safe_globals(safe_numpy_types)
except Exception:
    pass

# Checkpoint Loader Patch
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("train_lca")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

MODEL_MAP = {
    "cs_lrad": (CSLRADConfig, CSLRADForCausalLM),
    "transformer": (TransformerConfig, TransformerForCausalLM),
    "cs_ttt": (CSTTTConfig, CSTTTForCausalLM),
    "cs_lrad_transformer": (
        CSLRADTransformerConfig,
        CSLRADTransformerForCausalLM,
    ),
    "cs_ttt_transformer": (CSTTTTransformerConfig, CSTTTTransformerForCausalLM),
    "cs_lrad_cs_ttt_transformer": (
        CSLRADCSTTTTransformerConfig,
        CSLRADCSTTTTransformerForCausalLM,
    ),
    "baretorch": (BareTorchConfig, BareTorchForCausalLM),
}

ANNEALING_DATASETS = {"openr1_math", "finemath_4plus", "cosmopedia_v2"}
GENERAL_RATIOS = {
    "fineweb_edu_100bt": 0.40,
    "stack_dedup": 0.30,
    "dclm_100bt": 0.20,
    "finepdfs_100bt": 0.10,
}
REASONING_RATIOS = {
    "cosmopedia_v2": 0.476,
    "finemath_4plus": 0.476,
    "openr1_math": 0.048,
}


def get_wsd_schedule(
    optimizer,
    num_warmup_steps,
    num_decay_steps,
    num_training_steps,
    min_lr_ratio=0.1,
):
    num_stable_steps = max(
        0, num_training_steps - num_warmup_steps - num_decay_steps
    )

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        else:
            decay_step = current_step - (num_warmup_steps + num_stable_steps)
            if decay_step >= num_decay_steps:
                return min_lr_ratio
            progress = float(decay_step) / float(max(1, num_decay_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )

    return LambdaLR(optimizer, lr_lambda)


class DistributedCurriculumSampler(DistributedSampler):

    def __init__(
        self,
        dataset,
        trainer=None,
        decay_start_step=None,
        num_replicas=None,
        rank=None,
        seed=42,
    ):
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )

        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
        )

        self.dataset = dataset
        self.trainer = trainer
        self.decay_start_step = decay_start_step
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        split_idx = getattr(self.dataset, "split_idx", len(self.dataset))
        gen_indices = list(range(0, split_idx))
        ann_indices = list(range(split_idx, len(self.dataset)))

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        p_gen = (
            torch.randperm(len(gen_indices), generator=g).tolist()
            if gen_indices
            else []
        )
        p_ann = (
            torch.randperm(len(ann_indices), generator=g).tolist()
            if ann_indices
            else []
        )

        gen_shuffled = [gen_indices[i] for i in p_gen]
        ann_shuffled = [ann_indices[i] for i in p_ann]

        gen_rank = gen_shuffled[self.rank :: self.num_replicas]
        ann_rank = ann_shuffled[self.rank :: self.num_replicas]

        gen_ptr = 0
        ann_ptr = 0
        total_samples_per_rank = len(self.dataset) // self.num_replicas

        for _ in range(total_samples_per_rank):
            current_step = 0
            if (
                self.trainer is not None
                and hasattr(self.trainer, "state")
                and self.trainer.state is not None
            ):
                current_step = self.trainer.state.global_step

            use_annealing = (
                self.decay_start_step is not None
                and current_step >= self.decay_start_step
                and len(ann_rank) > 0
            )

            if use_annealing:
                p_val = torch.rand(1, generator=g).item()
                if p_val < 0.70 or len(gen_rank) == 0:
                    yield ann_rank[ann_ptr % len(ann_rank)]
                    ann_ptr += 1
                else:
                    yield gen_rank[gen_ptr % len(gen_rank)]
                    gen_ptr += 1
            else:
                if len(gen_rank) > 0:
                    yield gen_rank[gen_ptr % len(gen_rank)]
                    gen_ptr += 1
                elif len(ann_rank) > 0:
                    yield ann_rank[ann_ptr % len(ann_rank)]
                    ann_ptr += 1

    def __len__(self):
        return len(self.dataset) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class R2CheckpointCallback(TrainerCallback):

    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints_lca",
    ):
        self.bucket_name = bucket_name
        self.remote_name = remote_name
        self.prefix = prefix.strip("/")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            local_ckpt_path = os.path.join(args.output_dir, checkpoint_dir)

            if os.path.exists(local_ckpt_path):
                rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
                target_r2_path = f"{self.remote_name}:{self.bucket_name}/{self.prefix}/{rel_output_dir}/{checkpoint_dir}"

                logger.info(
                    f"\n[R2 Sync] Uploading {checkpoint_dir} to Cloudflare R2"
                    f" ({target_r2_path}) in background..."
                )
                cmd = [
                    "rclone",
                    "copy",
                    local_ckpt_path,
                    target_r2_path,
                    "--transfers",
                    "4",
                    "--s3-chunk-size",
                    "64M",
                ]
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )


class DistillMemmapDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 32768,
        dataset_filter: str = "all",
        max_samples: int = None,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.dataset_filter = dataset_filter
        self.max_samples = max_samples
        self.samples = []
        self.split_idx = 0
        self.reload_shards()

    def reload_shards(self):
        token_files = sorted(
            glob.glob(
                os.path.join(self.data_dir, "**/*_packed_tokens.bin"),
                recursive=True,
            )
        )
        token_files = [f for f in token_files if not f.endswith(".corrupt")]

        if not token_files:
            raise FileNotFoundError(
                f"❌ No valid '*_packed_tokens.bin' files found in '{self.data_dir}'."
            )

        general_samples = []
        annealing_samples = []

        for t_path in token_files:
            prefix = t_path.replace("_packed_tokens.bin", "")
            idx_path = f"{prefix}_teacher_indices.bin"
            val_path = f"{prefix}_teacher_values.bin"

            if not (os.path.exists(idx_path) and os.path.exists(val_path)):
                continue

            is_annealing = any(ds in t_path for ds in ANNEALING_DATASETS)

            if self.dataset_filter == "general" and is_annealing:
                continue
            if self.dataset_filter == "reasoning" and not is_annealing:
                continue

            num_tokens = os.path.getsize(t_path) // 4
            num_seqs = num_tokens // self.seq_len

            if num_seqs > 0:
                target_list = (
                    annealing_samples if is_annealing else general_samples
                )
                for s_i in range(num_seqs):
                    target_list.append((t_path, idx_path, val_path, s_i, num_seqs))

        self.split_idx = len(general_samples)
        self.samples = general_samples + annealing_samples

        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = self.samples[: self.max_samples]

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(
                f"Loaded 32k dataset track '{self.dataset_filter}'"
                f" ({len(self.samples):,} sequences |"
                f" {(len(self.samples) * self.seq_len) / 1e9:.2f}BT) from"
                f" '{self.data_dir}'"
            )

    def _get_worker_mmap(self, path, dtype, num_seqs, extra_dim=None):
        current_pid = os.getpid()

        if (
            not hasattr(self, "_worker_mmap_cache")
            or getattr(self, "_worker_pid", None) != current_pid
        ):
            self._worker_mmap_cache = {}
            self._worker_pid = current_pid

        if path not in self._worker_mmap_cache:
            total_valid_tokens = num_seqs * self.seq_len
            if extra_dim is None:
                shape = (num_seqs, self.seq_len)
                m = np.memmap(
                    path, dtype=dtype, mode="r", shape=(total_valid_tokens,)
                ).reshape(shape)
            else:
                shape = (num_seqs, self.seq_len, extra_dim)
                m = np.memmap(
                    path, dtype=dtype, mode="r", shape=(total_valid_tokens, extra_dim)
                ).reshape(shape)
            m.flags.writeable = False
            self._worker_mmap_cache[path] = m

        return self._worker_mmap_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_idx = idx % len(self.samples)
        t_path, idx_path, val_path, s_i, num_seqs = self.samples[s_idx]

        tok_mmap = self._get_worker_mmap(t_path, np.uint32, num_seqs)
        idx_mmap = self._get_worker_mmap(idx_path, np.uint32, num_seqs, extra_dim=4)
        val_mmap = self._get_worker_mmap(
            val_path, np.float16, num_seqs, extra_dim=4
        )

        tokens = torch.from_numpy(tok_mmap[s_i].astype(np.int64))
        teacher_indices = torch.from_numpy(idx_mmap[s_i].astype(np.int64))
        teacher_values = torch.from_numpy(val_mmap[s_i].astype(np.float32))

        tokens = torch.clamp(tokens, 0, 248076)
        teacher_indices = torch.clamp(teacher_indices, 0, 248076)
        teacher_values = torch.nan_to_num(
            teacher_values, nan=0.0, posinf=0.0, neginf=0.0
        )

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
        }


class DistillTrainer(Trainer):

    def __init__(
        self,
        *args,
        alpha_ce=0.5,
        alpha_kl=0.5,
        temperature=1.0,
        decay_steps=None,
        loss_chunk_size=512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature
        self.decay_steps = decay_steps
        self.loss_chunk_size = loss_chunk_size

        self._tr_loss_ce = 0.0
        self._tr_loss_kl = 0.0
        self._tr_loss_steps = 0

    def _get_train_sampler(self, dataset=None) -> DistributedSampler:
        target_dataset = dataset if dataset is not None else self.train_dataset
        if target_dataset is not None:
            decay_steps = (
                self.decay_steps
                if self.decay_steps is not None and self.decay_steps > 0
                else int(self.args.max_steps * 0.15)
            )
            decay_start_step = max(0, self.args.max_steps - decay_steps)
            return DistributedCurriculumSampler(
                dataset=target_dataset,
                trainer=self,
                decay_start_step=decay_start_step,
                seed=self.args.seed,
            )
        try:
            return super()._get_train_sampler(dataset)
        except TypeError:
            return super()._get_train_sampler()

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer = None,
    ):
        if self.args.lr_scheduler_type in ["wsd", "warmup_stable_decay"]:
            if self.lr_scheduler is None:
                optimizer = self.optimizer if optimizer is None else optimizer
                decay_steps = (
                    self.decay_steps
                    if self.decay_steps is not None and self.decay_steps > 0
                    else int(num_training_steps * 0.15)
                )
                self.lr_scheduler = get_wsd_schedule(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_decay_steps=decay_steps,
                    num_training_steps=num_training_steps,
                )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        import torch.utils.checkpoint as cp

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        teacher_indices = inputs["teacher_indices"]
        teacher_values = inputs["teacher_values"]

        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and hasattr(model, "module"):
            lm_head = getattr(model.module, "lm_head", None)

        last_hidden = hidden_states[-1]
        shift_hidden = last_hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_teacher_indices = teacher_indices[:, :-1, :]
        shift_teacher_values = teacher_values[:, :-1, :]

        p_teacher_full = F.softmax(
            shift_teacher_values / self.temperature, dim=-1
        )

        B, L_shift, _ = shift_hidden.shape
        num_tokens = B * L_shift
        chunk_size = self.loss_chunk_size

        total_loss_ce = 0.0
        total_loss_kl = 0.0

        def compute_chunk(c_hid, c_lbl, c_t_idx, c_t_p):
            target_dtype = (
                lm_head.weight.dtype
                if hasattr(lm_head, "weight")
                else torch.bfloat16
            )
            c_hid_input = (
                c_hid.to(target_dtype) if c_hid.dtype != target_dtype else c_hid
            )
            c_log = lm_head(c_hid_input)
            ce = F.cross_entropy(
                c_log.reshape(-1, c_log.size(-1)),
                c_lbl.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            lps = F.log_softmax(c_log.float() / self.temperature, dim=-1)
            st_top4 = torch.gather(lps, dim=-1, index=c_t_idx)
            kl = F.kl_div(st_top4, c_t_p.float(), reduction="sum")
            return ce, kl

        for c_start in range(0, L_shift, chunk_size):
            c_end = min(c_start + chunk_size, L_shift)
            c_hidden = shift_hidden[:, c_start:c_end, :]
            c_labels = shift_labels[:, c_start:c_end]
            c_teacher_idx = shift_teacher_indices[:, c_start:c_end, :]
            c_p_teacher = p_teacher_full[:, c_start:c_end, :]

            if c_hidden.requires_grad:
                c_loss_ce, c_loss_kl = cp.checkpoint(
                    compute_chunk,
                    c_hidden,
                    c_labels,
                    c_teacher_idx,
                    c_p_teacher,
                    use_reentrant=False,
                )
            else:
                c_loss_ce, c_loss_kl = compute_chunk(
                    c_hidden, c_labels, c_teacher_idx, c_p_teacher
                )

            total_loss_ce += c_loss_ce
            total_loss_kl += c_loss_kl

        loss_ce = total_loss_ce / num_tokens
        loss_kl = (total_loss_kl / num_tokens) * (self.temperature**2)
        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kl * loss_kl)

        if model.training:
            self._tr_loss_ce += loss_ce.detach().item()
            self._tr_loss_kl += loss_kl.detach().item()
            self._tr_loss_steps += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if "loss" in logs and self._tr_loss_steps > 0:
            logs["loss_ce"] = round(self._tr_loss_ce / self._tr_loss_steps, 4)
            logs["loss_kl"] = round(self._tr_loss_kl / self._tr_loss_steps, 4)
            self._tr_loss_ce = 0.0
            self._tr_loss_kl = 0.0
            self._tr_loss_steps = 0
        super().log(logs, *args, **kwargs)


def adapt_model_for_long_context(
    model, config, target_seq_len=32768, target_rope_theta=1000000.0
):
    """Explicitly scales RotaryEmbedding base theta and forces cache regeneration
    across all Transformer blocks for long context adaptation.
    """
    config.max_seq_len = target_seq_len
    config.rope_theta = target_rope_theta
    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = target_seq_len

    adapted_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, RotaryEmbedding) or module.__class__.__name__ in [
            "RotaryEmbedding",
            "RoPE",
        ]:
            # 1. Update internal module metadata
            module.base = float(target_rope_theta)
            module.max_position_embeddings = target_seq_len

            # 2. Recompute inv_freq tensor with the new base theta
            device = module.inv_freq.device
            dim = module.dim
            new_inv_freq = 1.0 / (
                target_rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            module.register_buffer("inv_freq", new_inv_freq, persistent=False)

            # 3. CRITICAL: Force recalculation of cached sinusoidal angles
            # This overrides the default base=10000 cache initialized during model creation
            module._set_cos_sin_cache(
                seq_len=target_seq_len, device=device, dtype=torch.float32
            )

            adapted_layers += 1

    return adapted_layers


def load_pretrained_weights(model, checkpoint_dir):
    """Loads weights from safetensors or PyTorch checkpoints without strict size checks."""
    from safetensors.torch import load_file

    logger.info(f"Loading base weights from checkpoint: {checkpoint_dir}")
    safetensor_files = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))

    state_dict = {}
    if safetensor_files:
        for sf in safetensor_files:
            state_dict.update(load_file(sf, device="cpu"))
    else:
        pt_files = glob.glob(os.path.join(checkpoint_dir, "pytorch_model*.bin"))
        for pf in pt_files:
            state_dict.update(
                torch.load(pf, map_location="cpu", weights_only=False)
            )

    if not state_dict:
        raise FileNotFoundError(
            f"No model weights found in checkpoint directory: {checkpoint_dir}"
        )

    # Filter out positional embedding tables that may have dimension mismatches
    model_state = model.state_dict()
    filtered_sd = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_sd[k] = v
            else:
                logger.warning(
                    f"Shape mismatch on {k}: base has {v.shape}, new model expects"
                    f" {model_state[k].shape}. Re-initializing parameter."
                )

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    logger.info(
        f"Base model loaded successfully. Missing keys: {len(missing)} |"
        f" Unexpected keys: {len(unexpected)}"
    )


def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(
        description="BareTorch Long-Context Adaptation (32K LCA Engine)"
    )

    # Checkpoint & Model
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained checkpoint to resume from (e.g. checkpoint-143052)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baretorch",
        choices=list(MODEL_MAP.keys()),
    )
    parser.add_argument(
        "--layer_sequence",
        type=str,
        default="cs_lrad,cs_lrad,cs_lrad,transformer",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints_lca_2.5B_32k"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B"
    )
    parser.add_argument(
        "--data_cache_dir", type=str, default="./teacher_predictions"
    )
    parser.add_argument("--tie_embeddings", action="store_true", default=False)
    parser.add_argument("--use_qk_norm", action="store_true", default=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)

    # Long Context Hyperparameters
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--loss_chunk_size", type=int, default=512)

    # Distillation Parameters
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Optimization (LCA: low LR, 10B tokens total budget)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="~10,000 steps at 1M tokens/step = ~10BT",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-5,
        help="Context adaptation rate: 5e-5 to 1e-4",
    )
    parser.add_argument("--scheduler", type=str, default="wsd")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--decay_steps", type=int, default=1500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Must be 1 per GPU at 32k context",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="8 GPUs * 1 batch * 4 accum * 32768 = 1,048,576 tokens/step",
    )

    # Architectural Dimensions
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=36)

    # Infrastructure
    parser.add_argument("--r2_sync", action="store_true")
    parser.add_argument("--r2_bucket", type=str, default="baretorch-data")
    parser.add_argument("--r2_prefix", type=str, default="checkpoints_lca")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )
    vocab_size = len(tokenizer)

    config_cls, model_cls = MODEL_MAP[args.model_type]
    config_args = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "rank": args.rank,
        "max_seq_len": args.seq_len,
        "num_kv_heads": max(1, args.num_heads // 4),
        "use_grad_checkpointing": True,
        "use_qk_norm": args.use_qk_norm,
        "tie_word_embeddings": args.tie_embeddings,
    }

    if args.model_type == "baretorch":
        raw_sequence = [
            s.strip().lower()
            for s in args.layer_sequence.split(",")
            if s.strip()
        ]
        config_args["layer_types"] = [
            raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)
        ]

    # 1. Initialize architecture with 32k sequence length
    config = config_cls(**config_args)
    model = model_cls(config)

    # 2. Load pre-trained weights from 75BT checkpoint
    load_pretrained_weights(model, args.base_checkpoint)

    # 3. Adapt RoPE frequencies & force-regenerate sinusoidal cache
    adapted_count = adapt_model_for_long_context(
        model, config, args.seq_len, args.rope_theta
    )

    # 4. Tie embeddings via the native PreTrainedModel hook
    if args.tie_embeddings:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        elif hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
            model.lm_head.weight = model.embed_tokens.weight

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info("🚀 BareTorch Long-Context Adaptation Engine (32K)")
        logger.info(f"   ├─ Base Checkpoint       : {args.base_checkpoint}")
        logger.info(f"   ├─ Sequence Length       : {args.seq_len:,} tokens")
        logger.info(f"   ├─ RoPE Theta Base       : {args.rope_theta:,.0f}")
        logger.info(
            f"   ├─ Adapted RoPE Blocks   : {adapted_count} (Expected: 9)"
        )
        logger.info(f"   ├─ Learning Rate         : {args.learning_rate}")
        logger.info(
            f"   └─ Batch per Step        :"
            f" {8 * args.batch_size * args.grad_accum * args.seq_len:,} tokens"
        )
        logger.info("=" * 70)

    train_dataset = DistillMemmapDataset(
        data_dir=args.data_cache_dir, seq_len=args.seq_len, dataset_filter="all"
    )

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_{args.model_type}",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        max_grad_norm=0.5,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        torch_compile=False,
        gradient_checkpointing=True,
        fsdp="shard_grad_op",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    callbacks = []
    if args.r2_sync:
        callbacks.append(
            R2CheckpointCallback(
                bucket_name=args.r2_bucket, prefix=args.r2_prefix
            )
        )

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        temperature=args.temperature,
        decay_steps=args.decay_steps,
        loss_chunk_size=args.loss_chunk_size,
    )

    # Check for existing LCA checkpoints to allow safe auto-resumption
    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing checkpoint(s) in '{training_args.output_dir}'."
                    " Resuming LCA training automatically..."
                )

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)


if __name__ == "__main__":
    main()