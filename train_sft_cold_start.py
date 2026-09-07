import argparse
import logging
import os
import subprocess
from datasets import Dataset, load_dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from baretorch import BareTorchConfig, BareTorchForCausalLM

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# ==============================================================================
#                        Cloudflare R2 Background Sync Callback
# ==============================================================================
class R2CheckpointCallback(TrainerCallback):
    """Hugging Face Trainer Callback that automatically syncs newly saved
    checkpoints to Cloudflare R2 asynchronously using rclone.
    Does not block active GPU training execution.
    """

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


def format_gsm8k_to_chatml(example):
    """Transforms raw GSM8K question and answer into structured CoT ChatML format:

    <think>
    step-by-step reasoning...
    </think>
    \boxed{final_answer}
    """
    question = example["question"]
    raw_answer = example["answer"]

    if "####" in raw_answer:
        parts = raw_answer.split("####")
        reasoning = parts[0].strip()
        target = parts[1].strip().replace(",", "")
        assistant_content = f"<think>\n{reasoning}\n</think>\n\\boxed{{{target}}}"
    else:
        assistant_content = f"<think>\n{raw_answer.strip()}\n</think>"

    system_prompt = (
        "You are a helpful AI assistant that solves math problems step-by-step. "
        "Put your reasoning inside <think>...</think> tags and write the final"
        " answer inside \\boxed{...}."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]


def pack_cot_chatml_dataset(
    raw_dataset, tokenizer, max_seq_len=1024, local_rank=0
):
    """Packs CoT ChatML dialogues into dense max_seq_len-token blocks.

    Completely eliminates padding tokens, preventing CS-LRAD recurrent state
    transition matrix contamination, guaranteeing static tensor shapes for
    torch.compile, and maximizing VRAM compute efficiency.
    """
    if local_rank == 0:
        logger.info(
            f"📦 Packing CoT dataset into dense {max_seq_len}-token blocks (Zero"
            " Padding)..."
        )

    all_input_ids = []
    all_labels = []
    packed_samples = []

    iterator = tqdm(
        raw_dataset,
        desc="Packing CoT Data",
        disable=(local_rank != 0),
        dynamic_ncols=True,
    )

    for example in iterator:
        if "messages" in example and example["messages"]:
            messages = example["messages"]
        else:
            messages = format_gsm8k_to_chatml(example)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            formatted_turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            tokens = tokenizer.encode(formatted_turn, add_special_tokens=False)

            all_input_ids.extend(tokens)

            # Loss masking: calculate loss exclusively on assistant tokens
            if role == "assistant":
                all_labels.extend(tokens)
            else:
                all_labels.extend([-100] * len(tokens))

        # Slice into uniform max_seq_len chunks
        while len(all_input_ids) >= max_seq_len:
            chunk_input_ids = all_input_ids[:max_seq_len]
            chunk_labels = all_labels[:max_seq_len]

            # Ensure at least one valid assistant loss target exists in shifted labels
            if any(lbl != -100 for lbl in chunk_labels[1:]):
                packed_samples.append({
                    "input_ids": chunk_input_ids,
                    "labels": chunk_labels,
                    "attention_mask": [1] * max_seq_len,
                })

            all_input_ids = all_input_ids[max_seq_len:]
            all_labels = all_labels[max_seq_len:]

    if local_rank == 0:
        logger.info(
            f"✅ Packing complete: Created {len(packed_samples):,} dense"
            f" {max_seq_len}-token sequences."
        )

    return Dataset.from_list(packed_samples)


def main():
    # Distributed Initialization for torchrun
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 3-Prep: Cold-Start CoT SFT Warmup Engine"
    )

    # Checkpoint Paths
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_sft/checkpoint-2759",
        help="Path to Stage 1 base SFT checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_500m_sft_cold_start",
        help="Directory to save Cold-Start SFT weights.",
    )

    # Tokenizer & Dataset Parameters
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/gsm8k",
        help="Hugging Face math dataset path.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="Dataset subset/config name.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Sub-sample N rows for fast warmup. Set to 0 for full dataset.",
    )

    # Hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Warmup SFT epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-GPU batch size."
    )
    parser.add_argument(
        "--grad_accum", type=int, default=2, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Cold-Start SFT learning rate.",
    )
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable targeted torch.compile for CS-LRAD sub-modules.",
    )

    # Cloud Storage / Sync
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        help=(
            "Enable background checkpoint syncing to Cloudflare R2 via rclone."
        ),
    )
    parser.add_argument(
        "--r2_bucket",
        type=str,
        default="baretorch-data",
        help="Cloudflare R2 bucket name.",
    )
    parser.add_argument(
        "--r2_prefix",
        type=str,
        default="checkpoints",
        help="Prefix path inside R2 bucket.",
    )

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. Tokenizer Setup
    if local_rank == 0:
        logger.info("Initializing Tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        if local_rank == 0:
            logger.info(
                f"Loaded existing tokenizer from checkpoint: {args.pretrained_model_path}"
            )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if local_rank == 0:
            logger.info(
                f"Loaded base tokenizer from HF Hub: {args.tokenizer_name}"
            )

    tokenizer.model_max_length = args.seq_len

    desired_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [t for t in desired_tokens if t not in existing_vocab]

    if tokens_to_add:
        if local_rank == 0:
            logger.info(f"Adding new special tokens to tokenizer: {tokens_to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    else:
        if local_rank == 0:
            logger.info(
                "All special tokens already exist in vocabulary. Skipping token"
                " addition."
            )

    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.eos_token = "<|im_end|>"

    # 2. Model Loading & Safe Tensor-Core Padded Embedding Resizing
    if local_rank == 0:
        logger.info(
            f"Loading base SFT model weights from: {args.pretrained_model_path}"
        )

    model = BareTorchForCausalLM.from_pretrained(
        args.pretrained_model_path, torch_dtype=torch.bfloat16
    )

    if tokens_to_add:
        if local_rank == 0:
            logger.info(
                f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)} (Padded to multiple of 64)..."
            )
        # Pad embedding layer to multiple of 64 for GPU Tensor Core memory alignment
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64, mean_resizing=False)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.config.use_cache = False

    # Targeted Sub-Module Compilation specifically for CS-LRAD recurrent layers
    if args.compile:
        if local_rank == 0:
            logger.info(
                "⚡ Applying targeted torch.compile to custom CS-LRAD sub-modules..."
            )
        compiled_blocks = 0
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if "lrad" in cls_name or "lrad" in name.lower():
                module.forward = torch.compile(module.forward)
                compiled_blocks += 1
        if local_rank == 0:
            logger.info(
                f"Successfully compiled {compiled_blocks} CS-LRAD recurrent"
                " sub-module(s)."
            )

    # 3. Load & Process Dataset with Sequence Packing
    if local_rank == 0:
        logger.info(
            f"Loading GSM8K Math Dataset '{args.dataset_name}' (Config:"
            f" {args.dataset_config})..."
        )

    raw_dataset = load_dataset(
        args.dataset_name, args.dataset_config, split="train"
    )

    if args.max_samples > 0 and len(raw_dataset) > args.max_samples:
        if local_rank == 0:
            logger.info(
                f"✂️ Sub-sampling dataset from {len(raw_dataset):,} rows to"
                f" {args.max_samples:,} rows."
            )
        raw_dataset = raw_dataset.select(range(args.max_samples))

    processed_dataset = pack_cot_chatml_dataset(
        raw_dataset, tokenizer, max_seq_len=args.seq_len, local_rank=local_rank
    )

    # Split 5% off for validation
    dataset_split = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]

    if local_rank == 0:
        logger.info(
            f"Dataset split complete: {len(train_data):,} training samples |"
            f" {len(val_data):,} validation samples."
        )

    # 4. Training Configurations
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
        torch_compile=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(
                f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}' |"
                f" Prefix: '{r2_prefix}'"
            )
        callbacks.append(
            R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # 5. Launch Supervised Fine-Tuning
    if local_rank == 0:
        logger.info("🔥 Starting Stage 3-Prep: Cold-Start CoT SFT Warmup...")

    trainer.train()

    if local_rank == 0:
        logger.info(
            f"Saving final Cold-Start SFT checkpoint to '{args.output_dir}'..."
        )
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Cold-Start CoT Warmup completed successfully!")

        if enable_r2_sync:
            rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
            target_r2_path = (
                f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
            )
            logger.info(
                "📤 Syncing final Cold-Start SFT model weights to Cloudflare R2"
                f" ({target_r2_path})..."
            )
            cmd = [
                "rclone",
                "copy",
                args.output_dir,
                target_r2_path,
                "--transfers",
                "8",
                "--s3-chunk-size",
                "64M",
            ]
            subprocess.run(cmd, check=False)
            logger.info(
                "✅ Final Cold-Start SFT weights successfully uploaded to Cloudflare"
                " R2!"
            )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()