import os
import glob
import argparse
import logging
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
    TrainerCallback,
)

from baretorch import (
    CSLRADConfig, CSLRADForCausalLM,
    TransformerConfig, TransformerForCausalLM,
    CSTTTConfig, CSTTTForCausalLM,
    CSLRADTransformerConfig, CSLRADTransformerForCausalLM,
    CSTTTTransformerConfig, CSTTTTransformerForCausalLM,
    CSLRADCSTTTTransformerConfig, CSLRADCSTTTTransformerForCausalLM,
    BareTorchConfig, BareTorchForCausalLM,
)

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
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
    "cs_lrad_cs_ttt_transformer": (CSLRADCSTTTTransformerConfig, CSLRADCSTTTTransformerForCausalLM),
    "baretorch": (BareTorchConfig, BareTorchForCausalLM),
}


# ==============================================================================
#                     Cloudflare R2 Background Sync Callback
# ==============================================================================
class R2CheckpointCallback(TrainerCallback):
    """
    Hugging Face Trainer Callback that automatically syncs newly saved 
    checkpoints to Cloudflare R2 asynchronously using rclone.
    Does not block active GPU training execution.
    """
    def __init__(self, bucket_name: str = "baretorch-data", remote_name: str = "r2", prefix: str = "checkpoints"):
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
                
                logger.info(f"\n[R2 Sync] Uploading {checkpoint_dir} to Cloudflare R2 ({target_r2_path}) in background...")
                cmd = [
                    "rclone", "copy",
                    local_ckpt_path,
                    target_r2_path,
                    "--transfers", "4",
                    "--s3-chunk-size", "64M"
                ]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ==============================================================================
#                 Zero-Copy Memory-Mapped Binary Dataset
# ==============================================================================
class MemmapDataset(Dataset):
    """
    High-Performance, zero-copy PyTorch Dataset mapping directly over 
    flat uint16 pre-tokenized binary files using numpy.memmap.
    """
    def __init__(self, bin_dir: str, seq_len: int = 2048):
        self.seq_len = seq_len
        
        # Search for binary shard files
        bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(f"❌ No .bin files found in '{bin_dir}'. Run tokenize_to_bin.py first!")

        self.memmaps = []
        self.file_lengths = []
        total_tokens = 0

        for fpath in bin_files:
            # uint16 = 2 bytes per token
            num_tokens = os.path.getsize(fpath) // 2
            if num_tokens >= seq_len:
                # Direct pointer to NVMe memory space with zero RAM loading
                mmap = np.memmap(fpath, dtype=np.uint16, mode="r")
                self.memmaps.append(mmap)
                
                num_samples = num_tokens // seq_len
                self.file_lengths.append(num_samples)
                total_tokens += num_tokens

        self.cum_samples = np.cumsum(self.file_lengths)
        self.total_samples = int(self.cum_samples[-1]) if len(self.cum_samples) > 0 else 0
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            logger.info(f"Loaded {len(self.memmaps)} binary shard(s) from '{bin_dir}'")
            logger.info(f"   └─ Total Tokens: {total_tokens:,} | Total Sequences (L={seq_len}): {self.total_samples:,}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Locate target shard via binary search
        file_idx = np.searchsorted(self.cum_samples, idx, side="right")
        sample_in_file = idx if file_idx == 0 else idx - self.cum_samples[file_idx - 1]

        start_idx = sample_in_file * self.seq_len
        end_idx = start_idx + self.seq_len

        # Slice sequence window off NVMe disk
        chunk = self.memmaps[file_idx][start_idx:end_idx].astype(np.int64)
        x = torch.from_numpy(chunk)

        return {"input_ids": x, "labels": x.clone()}


def prepare_dataset(args):
    """
    Instantiates memory-mapped datasets directly from pre-tokenized binary directories.
    Optionally caps validation set to args.max_val_samples if configured.
    """
    train_dir = os.path.join(args.data_cache_dir, "train")
    val_dir = os.path.join(args.data_cache_dir, "val")
    
    # Fallback if binary files are placed directly inside data_cache_dir root
    if not os.path.exists(train_dir) and os.path.exists(args.data_cache_dir):
        train_dir = args.data_cache_dir
        val_dir = args.data_cache_dir

    train_dataset = MemmapDataset(bin_dir=train_dir, seq_len=args.seq_len)
    val_dataset = MemmapDataset(bin_dir=val_dir, seq_len=args.seq_len)
    
    # Optional capping of validation samples
    if args.max_val_samples is not None and args.max_val_samples > 0:
        if len(val_dataset) > args.max_val_samples:
            val_dataset = Subset(val_dataset, range(args.max_val_samples))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                capped_tokens = args.max_val_samples * args.seq_len
                logger.info(f"✂️ Capped validation dataset to {args.max_val_samples:,} sequences (~{capped_tokens:,} tokens)")

    return train_dataset, val_dataset


def main():
    # Early Distributed Process Group Initialization for torchrun
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="BareTorch High-Throughput Cluster Pre-training Engine")
    
    # Architecture
    parser.add_argument("--model_type", type=str, default="baretorch", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_500m")
    
    # Dataset & Binary Paths
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/dclm_100BT-shuffled")
    parser.add_argument("--data_cache_dir", type=str, default="./tokenized_bin", help="Directory containing pre-tokenized uint16 .bin files.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum validation sequence samples (L=seq_len). If unset, uses full validation dataset.")
    
    # Optimization & Batching
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    
    # Structural Dimensions
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seq_len", type=int, default=2048)
    
    # Cloud Storage / Sync
    parser.add_argument("--r2_sync", action="store_true", help="Enable background checkpoint syncing to Cloudflare R2 via rclone.")
    parser.add_argument("--r2_bucket", type=str, default="baretorch-data", help="Cloudflare R2 bucket name.")
    parser.add_argument("--r2_prefix", type=str, default="checkpoints", help="Prefix path inside R2 bucket.")
    
    # Engine Configurations
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(f"Initializing BareTorch Engine. Selected Architecture: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    config_cls, model_cls = MODEL_MAP[args.model_type]
    
    config_args = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "rank": args.rank,
        "dropout": args.dropout,
        "use_grad_checkpointing": args.grad_checkpointing,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if args.model_type == "baretorch":
        raw_sequence = [s.strip().lower() for s in args.layer_sequence.split(",") if s.strip()]
        layer_types = [raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)]
        config_args["layer_types"] = layer_types
        if local_rank == 0:
            logger.info(f"Assembled Hybrid Sequence ({args.num_layers} layers): {layer_types}")
        
    if "transformer" in args.model_type or args.model_type == "baretorch":
        config_args["max_seq_len"] = args.seq_len
        config_args["num_kv_heads"] = max(1, args.num_heads // 4)

    config = config_cls(**config_args)
    model = model_cls(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        logger.info(f"Model initialized successfully. Total parameters: {total_params / 1e6:.2f}M")

    # Fast memory-mapped dataset loading
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
        report_to="tensorboard",
        logging_dir=f"./runs/{args.model_type}",
        torch_compile=args.compile,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Detect R2 sync activation via argument or environment variables
    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'")
        callbacks.append(R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix))
    else:
        if local_rank == 0:
            logger.info("R2 sync disabled. Running in local mode (disk checkpoints only).")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # Automatic Checkpoint Resumption Logic
    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(f"Found existing checkpoint in '{training_args.output_dir}'. Resuming training...")

    if local_rank == 0:
        logger.info("Starting pre-training run...")
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    
    if local_rank == 0:
        logger.info("Pre-training run successfully completed!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()