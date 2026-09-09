import argparse
import glob
import logging
import os
import random
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Sampler
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

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

MODEL_MAP = {
    "cs_lrad": (CSLRADConfig, CSLRADForCausalLM),
    "transformer": (TransformerConfig, TransformerForCausalLM),
    "cs_ttt": (CSTTTConfig, CSTTTForCausalLM),
    "cs_lrad_transformer": (CSLRADTransformerConfig, CSLRADTransformerForCausalLM),
    "cs_ttt_transformer": (CSTTTTransformerConfig, CSTTTTransformerForCausalLM),
    "cs_lrad_cs_ttt_transformer": (
        CSLRADCSTTTTransformerConfig,
        CSLRADCSTTTTransformerForCausalLM,
    ),
    "baretorch": (BareTorchConfig, BareTorchForCausalLM),
}

# High-density datasets preserved for WSD Annealing Phase (Phase 2)
ANNEALING_DATASETS = {"openr1_math", "finemath_4plus", "cosmopedia_v2"}

# Scaled quotas matching exact 165B dataset mix (2,000 sequences total / ~4.1M tokens)
GENERAL_QUOTAS = {
    "fineweb_edu_100bt": 667,
    "stack_dedup": 424,
    "dclm_100bt": 303,
    "finepdfs_100bt": 133,
}

REASONING_QUOTAS = {
    "cosmopedia_v2": 340,
    "finemath_4plus": 121,
    "openr1_math": 12,
}


def get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps, num_training_steps, min_lr_ratio=0.0):
    num_stable_steps = max(0, num_training_steps - num_warmup_steps - num_decay_steps)

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
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class DistributedCurriculumSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=42):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.dataset = dataset
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

        if gen_indices:
            p_gen = torch.randperm(len(gen_indices), generator=g).tolist()
            gen_shuffled = [gen_indices[i] for i in p_gen]
        else:
            gen_shuffled = []

        if ann_indices:
            p_ann = torch.randperm(len(ann_indices), generator=g).tolist()
            ann_shuffled = [ann_indices[i] for i in p_ann]
        else:
            ann_shuffled = []

        gen_rank = gen_shuffled[self.rank :: self.num_replicas]
        ann_rank = ann_shuffled[self.rank :: self.num_replicas]

        indices = gen_rank + ann_rank
        return iter(indices)

    def __len__(self):
        return len(self.dataset) // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


class R2CheckpointCallback(TrainerCallback):
    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints",
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
                target_r2_path = (
                    f"{self.remote_name}:{self.bucket_name}/{self.prefix}/{rel_output_dir}/{checkpoint_dir}"
                )

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
    def __init__(self, data_dir: str, seq_len: int = 2048, dataset_filter: str = "all", max_samples: int = None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.dataset_filter = dataset_filter
        self.max_samples = max_samples
        self.samples = []
        self.split_idx = 0
        self.reload_shards()

    def reload_shards(self):
        token_files = sorted(glob.glob(os.path.join(self.data_dir, "**/*_packed_tokens.bin"), recursive=True))
        if not token_files:
            raise FileNotFoundError(
                f"❌ No '*_packed_tokens.bin' files found in '{self.data_dir}'."
                " Run teacher_inference.py first!"
            )

        general_samples = []
        annealing_samples = []
        total_tokens = 0

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
                target_list = annealing_samples if is_annealing else general_samples

                for s_i in range(num_seqs):
                    # Store path strings to allow lazy memmap creation inside worker processes
                    target_list.append((t_path, idx_path, val_path, s_i))

                total_tokens += num_seqs * self.seq_len

        self.split_idx = len(general_samples)
        self.samples = general_samples + annealing_samples

        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = self.samples[: self.max_samples]

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(f"Loaded dataset track '{self.dataset_filter}' ({len(self.samples):,} sequences) from '{self.data_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            raise RuntimeError(f"Dataset for filter '{self.dataset_filter}' is empty.")

        s_idx = idx % len(self.samples)
        t_path, idx_path, val_path, s_i = self.samples[s_idx]

        # Lazy memmap access per worker execution slice
        tok_mmap = np.memmap(t_path, dtype=np.uint32, mode="r", shape=(-1, self.seq_len))
        idx_mmap = np.memmap(idx_path, dtype=np.uint32, mode="r", shape=(-1, self.seq_len, 4))
        val_mmap = np.memmap(val_path, dtype=np.float16, mode="r", shape=(-1, self.seq_len, 4))

        tokens = torch.from_numpy(tok_mmap[s_i].astype(np.int64))
        teacher_indices = torch.from_numpy(idx_mmap[s_i].astype(np.int64))
        teacher_values = torch.from_numpy(val_mmap[s_i].astype(np.float32))

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
        }


def build_proportional_val_dataset(data_dir, quotas, seq_len=2048, seed=42):
    rng = random.Random(seed)
    collected_samples = []

    for ds_name, target_count in quotas.items():
        ds_dir = os.path.join(data_dir, ds_name)
        token_files = sorted(glob.glob(os.path.join(ds_dir, "**/*_packed_tokens.bin"), recursive=True))

        ds_samples = []
        for t_path in token_files:
            prefix = t_path.replace("_packed_tokens.bin", "")
            idx_path = f"{prefix}_teacher_indices.bin"
            val_path = f"{prefix}_teacher_values.bin"

            if not (os.path.exists(idx_path) and os.path.exists(val_path)):
                continue

            num_tokens = os.path.getsize(t_path) // 4
            num_seqs = num_tokens // seq_len

            if num_seqs > 0:
                for s_i in range(num_seqs):
                    ds_samples.append((t_path, idx_path, val_path, s_i))

        if ds_samples:
            rng.shuffle(ds_samples)
            sampled = ds_samples[:target_count]
            collected_samples.extend(sampled)

    dataset = DistillMemmapDataset.__new__(DistillMemmapDataset)
    dataset.data_dir = data_dir
    dataset.seq_len = seq_len
    dataset.samples = collected_samples
    dataset.dataset_filter = "proportional_val"
    dataset.split_idx = len(collected_samples)
    return dataset


class DistillTrainer(Trainer):
    def __init__(self, *args, alpha_ce=0.5, alpha_kl=0.5, temperature=1.0, decay_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature
        self.decay_steps = decay_steps

    def _get_train_sampler(self, dataset=None) -> Sampler:
        target_dataset = dataset if dataset is not None else self.train_dataset
        if target_dataset is not None:
            return DistributedCurriculumSampler(target_dataset, seed=self.args.seed)
        try:
            return super()._get_train_sampler(dataset)
        except TypeError:
            return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.args.lr_scheduler_type in ["wsd", "warmup_stable_decay"]:
            if self.lr_scheduler is None:
                optimizer = self.optimizer if optimizer is None else optimizer
                decay_steps = (
                    self.decay_steps
                    if self.decay_steps is not None and self.decay_steps > 0
                    else int(num_training_steps * 0.10)
                )
                self.lr_scheduler = get_wsd_schedule(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_decay_steps=decay_steps,
                    num_training_steps=num_training_steps,
                )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        teacher_indices = inputs["teacher_indices"]
        teacher_values = inputs["teacher_values"]

        outputs = model(input_ids=input_ids)
        student_logits = outputs.logits  # Shape: (B, L, Vocab)

        # Non-copying view slice
        shift_logits = student_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # 1. Hard Cross-Entropy Loss (flatten via reshape)
        loss_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        # 2. Top-4 KL Divergence Soft Distillation Loss
        shift_teacher_indices = teacher_indices[:, :-1, :]
        shift_teacher_values = teacher_values[:, :-1, :]

        student_top4_logits = torch.gather(shift_logits, dim=-1, index=shift_teacher_indices)

        p_teacher = F.softmax(shift_teacher_values / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_top4_logits / self.temperature, dim=-1)

        # Token-normalized KL divergence to match Cross-Entropy loss scale
        num_tokens = shift_logits.size(0) * shift_logits.size(1)
        loss_kl = (F.kl_div(log_p_student, p_teacher, reduction="sum") / num_tokens) * (self.temperature ** 2)

        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kl * loss_kl)

        return (total_loss, outputs) if return_outputs else total_loss


def prepare_dataset(args):
    train_dir = os.path.join(args.data_cache_dir, "train")
    val_dir = os.path.join(args.data_cache_dir, "val")

    if not os.path.exists(train_dir) and os.path.exists(args.data_cache_dir):
        train_dir = args.data_cache_dir
        val_dir = args.data_cache_dir

    train_dataset = DistillMemmapDataset(data_dir=train_dir, seq_len=args.seq_len, dataset_filter="all")

    val_general_dataset = build_proportional_val_dataset(val_dir, GENERAL_QUOTAS, seq_len=args.seq_len)
    val_reasoning_dataset = build_proportional_val_dataset(val_dir, REASONING_QUOTAS, seq_len=args.seq_len)

    eval_datasets = {
        "general": val_general_dataset,
        "reasoning": val_reasoning_dataset,
    }

    return train_dataset, eval_datasets


def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="BareTorch Knowledge Distillation Engine")

    # Architecture & Data
    parser.add_argument("--model_type", type=str, default="baretorch", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_distill_2.5B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--data_cache_dir", type=str, default="./teacher_predictions")
    parser.add_argument("--tie_embeddings", action="store_true", default=True, help="Tie input & output embedding weights.")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)

    # Distillation Parameters
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Optimization
    parser.add_argument("--max_steps", type=int, default=1258850)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--scheduler", type=str, default="wsd", help="Scheduler type: 'cosine', 'linear', or 'wsd'")
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--decay_steps", type=int, default=None, help="Decay steps for WSD schedule (default: 10% of max_steps)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)

    # Structural Dimensions
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=36)
    parser.add_argument("--seq_len", type=int, default=2048)

    # Cloud Sync & Engine
    parser.add_argument("--r2_sync", action="store_true")
    parser.add_argument("--r2_bucket", type=str, default="baretorch-data")
    parser.add_argument("--r2_prefix", type=str, default="checkpoints")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=25000)
    parser.add_argument("--eval_steps", type=int, default=10000)

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
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
        "use_grad_checkpointing": args.grad_checkpointing,
        "tie_word_embeddings": args.tie_embeddings,
    }

    if args.model_type == "baretorch":
        raw_sequence = [s.strip().lower() for s in args.layer_sequence.split(",") if s.strip()]
        config_args["layer_types"] = [raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)]

    config = config_cls(**config_args)
    model = model_cls(config)

    if args.tie_embeddings:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        elif hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
            model.lm_head.weight = model.embed_tokens.weight

    unique_params_dict = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params_dict.values())
    trainable_params = sum(p.numel() for p in unique_params_dict.values() if p.requires_grad)

    if local_rank == 0:
        logger.info("=" * 70)
        logger.info(f"🤖 Model Architecture Initialized: {args.model_type}")
        logger.info(f"   ├─ Tied Embeddings     : {args.tie_embeddings}")
        logger.info(f"   ├─ Total Parameters     : {total_params / 1e9:.3f} Billion ({total_params:,})")
        logger.info(f"   ├─ Trainable Parameters : {trainable_params / 1e9:.3f} Billion ({trainable_params:,})")
        logger.info(f"   └─ LR Schedule          : {args.scheduler.upper()} (Warmup={args.warmup_steps} steps)")
        logger.info("=" * 70)

    train_dataset, eval_datasets = prepare_dataset(args)

    scheduler_type = args.scheduler.lower()
    if scheduler_type == "wsd":
        scheduler_type = "warmup_stable_decay"

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_{args.model_type}",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        gradient_checkpointing=args.grad_checkpointing,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    callbacks = [R2CheckpointCallback(bucket_name=args.r2_bucket, prefix=args.r2_prefix)] if args.r2_sync else []

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        callbacks=callbacks,
        alpha_ce=args.alpha_ce,
        alpha_kl=alpha_kl,
        temperature=args.temperature,
        decay_steps=args.decay_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()