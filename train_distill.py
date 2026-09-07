import argparse
import glob
import logging
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Subset
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


def get_wsd_schedule(optimizer, num_warmup_steps, num_decay_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Warmup-Stable-Decay (WSD) Learning Rate Schedule.
    1. Warmup: Linear LR ramp up from 0 to 1.0
    2. Stable: Holds constant peak LR (1.0)
    3. Decay: Cosine decay to min_lr_ratio over the final num_decay_steps
    """
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


class R2CheckpointCallback(TrainerCallback):
    """Syncs checkpoints asynchronously to Cloudflare R2 using rclone."""

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
    """
    Zero-copy Dataset mapping triplets of binary files:
    1. packed_tokens (.bin, uint32, shape: [N, seq_len])
    2. teacher_indices (.bin, uint32, shape: [N, seq_len, 4])
    3. teacher_values (.bin, float16, shape: [N, seq_len, 4])
    """

    def __init__(self, data_dir: str, seq_len: int = 2048):
        self.seq_len = seq_len

        token_files = sorted(glob.glob(os.path.join(data_dir, "*_packed_tokens.bin")))
        if not token_files:
            raise FileNotFoundError(
                f"❌ No '*_packed_tokens.bin' files found in '{data_dir}'."
                " Run teacher_inference.py first!"
            )

        self.samples = []
        total_tokens = 0

        for t_path in token_files:
            prefix = t_path.replace("_packed_tokens.bin", "")
            idx_path = f"{prefix}_teacher_indices.bin"
            val_path = f"{prefix}_teacher_values.bin"

            if not (os.path.exists(idx_path) and os.path.exists(val_path)):
                continue

            num_tokens = os.path.getsize(t_path) // 4
            num_seqs = num_tokens // seq_len

            if num_seqs > 0:
                tok_mmap = np.memmap(t_path, dtype=np.uint32, mode="r", shape=(num_seqs, seq_len))
                idx_mmap = np.memmap(idx_path, dtype=np.uint32, mode="r", shape=(num_seqs, seq_len, 4))
                val_mmap = np.memmap(val_path, dtype=np.float16, mode="r", shape=(num_seqs, seq_len, 4))

                for s_i in range(num_seqs):
                    self.samples.append((tok_mmap, idx_mmap, val_mmap, s_i))

                total_tokens += num_seqs * seq_len

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(f"Loaded {len(token_files)} dataset shard triplet(s) from '{data_dir}'")
            logger.info(
                f"   └─ Total Tokens: {total_tokens:,} | Total Sequences (L={seq_len}): {len(self.samples):,}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tok_mmap, idx_mmap, val_mmap, s_i = self.samples[idx]

        tokens = torch.from_numpy(tok_mmap[s_i].astype(np.int64))
        teacher_indices = torch.from_numpy(idx_mmap[s_i].astype(np.int64))
        teacher_values = torch.from_numpy(val_mmap[s_i].astype(np.float32))

        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
        }


class DistillTrainer(Trainer):
    """
    Custom Hugging Face Trainer implementing combined Cross-Entropy, Top-4 KL Divergence Loss,
    and WSD Learning Rate Scheduler support.
    """

    def __init__(self, *args, alpha_ce=0.5, alpha_kl=0.5, temperature=1.0, decay_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce
        self.alpha_kl = alpha_kl
        self.temperature = temperature
        self.decay_steps = decay_steps

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.args.lr_scheduler_type == "wsd":
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
        teacher_indices = inputs["teacher_indices"]  # Shape: (B, L, 4)
        teacher_values = inputs["teacher_values"]    # Shape: (B, L, 4)

        # Student Forward Pass
        outputs = model(input_ids=input_ids)
        student_logits = outputs.logits  # Shape: (B, L, Vocab)

        # 1. Hard Cross-Entropy Loss on Ground-Truth Tokens
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 2. Top-4 KL Divergence Soft Distillation Loss
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_indices = teacher_indices[..., 1:, :].contiguous()
        shift_teacher_values = teacher_values[..., 1:, :].contiguous()

        # Gather student log probabilities corresponding exclusively to teacher's top-4 tokens
        student_top4_logits = torch.gather(shift_student_logits, dim=-1, index=shift_teacher_indices)

        # Softmax over top-4 candidate logits
        p_teacher = F.softmax(shift_teacher_values / self.temperature, dim=-1)
        log_p_student = F.log_softmax(student_top4_logits / self.temperature, dim=-1)

        loss_kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)

        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kl * loss_kl)

        return (total_loss, outputs) if return_outputs else total_loss


def prepare_dataset(args):
    train_dir = os.path.join(args.data_cache_dir, "train")
    val_dir = os.path.join(args.data_cache_dir, "val")

    if not os.path.exists(train_dir) and os.path.exists(args.data_cache_dir):
        train_dir = args.data_cache_dir
        val_dir = args.data_cache_dir

    train_dataset = DistillMemmapDataset(data_dir=train_dir, seq_len=args.seq_len)
    val_dataset = DistillMemmapDataset(data_dir=val_dir, seq_len=args.seq_len)

    if args.max_val_samples is not None and args.max_val_samples > 0:
        if len(val_dataset) > args.max_val_samples:
            val_dataset = Subset(val_dataset, range(args.max_val_samples))

    return train_dataset, val_dataset


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
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.8-27B")
    parser.add_argument("--data_cache_dir", type=str, default="./teacher_predictions")
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--tie_embeddings", action="store_true", default=True, help="Tie input & output embedding weights.")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)

    # Distillation Parameters
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--alpha_kl", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Optimization
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--scheduler", type=str, default="wsd", help="Scheduler type: 'cosine', 'linear', or 'wsd'")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--decay_steps", type=int, default=None, help="Decay steps for WSD schedule (default: 10% of max_steps)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)

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
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)

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

    # Apply Explicit Weight Tying across Embeddings and Output LM Head
    if args.tie_embeddings:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        elif hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
            model.lm_head.weight = model.embed_tokens.weight

    # Accurate Parameter Count (De-duplicating Tied Tensors via Memory Pointers)
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

    train_dataset, val_dataset = prepare_dataset(args)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_{args.model_type}",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
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
    )

    callbacks = [R2CheckpointCallback(bucket_name=args.r2_bucket, prefix=args.r2_prefix)] if args.r2_sync else []

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        alpha_ce=args.alpha_ce,
        alpha_kl=args.alpha_kl,
        temperature=args.temperature,
        decay_steps=args.decay_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()