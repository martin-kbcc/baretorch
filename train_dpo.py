# /home/martinkb/Desktop/BareTorch_F/train_dpo.py

import argparse
import logging
import os
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

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
#                         Cloudflare R2 Background Sync Callback
# ==============================================================================
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


# ==============================================================================
#                           ChatML Formatting Utility
# ==============================================================================
def format_dpo_sample(example):
    """
    Formats raw dataset rows into strict ChatML strings for DPOTrainer.
    Handles both conversational message lists and raw prompt/chosen/rejected strings.
    """
    prompt = example.get("prompt", example.get("question", example.get("instruction", "")))
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Handle conversational message lists
    if isinstance(prompt, list):
        prompt = "\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in prompt])
    else:
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if isinstance(chosen, list):
        chosen = chosen[-1]["content"] if len(chosen) > 0 else ""
    if isinstance(rejected, list):
        rejected = rejected[-1]["content"] if len(rejected) > 0 else ""

    formatted_chosen = f"{chosen}<|im_end|>\n"
    formatted_rejected = f"{rejected}<|im_end|>\n"

    return {
        "prompt": prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
    }


# ==============================================================================
#                                Main Engine
# ==============================================================================
def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 3: Direct Preference Optimization (DPO) Engine"
    )

    # Checkpoint Paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints_300m_grpo",
        help="Path to SFT/GRPO-aligned BareTorch checkpoint folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_300m_dpo",
        help="Directory to save final DPO aligned weights.",
    )

    # Tokenizer & Dataset
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        help="Hugging Face preference dataset identifier.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset subset/config name.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Sub-sample N rows for fast iteration. Set to 0 for full dataset.",
    )

    # DPO Hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature scaling factor.")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="DPO learning rate.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length (prompt + response).")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num_epochs", type=int, default=1)

    # Cloud Storage / Sync
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        help="Enable background checkpoint syncing to Cloudflare R2 via rclone.",
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
        logger.info(f"Initializing Tokenizer '{args.tokenizer_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Policy Model Loading
    if local_rank == 0:
        logger.info(f"Loading input model weights from: {args.model_path}")

    model = BareTorchForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV cache during training

    # 3. Load & Process Dataset
    if local_rank == 0:
        logger.info(f"Loading Preference Dataset '{args.dataset_name}'...")

    dataset_kwargs = {}
    if args.dataset_config and args.dataset_config.lower() not in ("default", "none", "null"):
        dataset_kwargs["name"] = args.dataset_config

    raw_dataset = load_dataset(args.dataset_name, split="train", **dataset_kwargs)

    if args.max_samples > 0 and len(raw_dataset) > args.max_samples:
        if local_rank == 0:
            logger.info(f"✂️ Sub-sampling dataset to {args.max_samples:,} rows.")
        raw_dataset = raw_dataset.select(range(args.max_samples))

    formatted_dataset = raw_dataset.map(
        format_dpo_sample,
        remove_columns=raw_dataset.column_names,
    )

    dataset_split = formatted_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]

    if local_rank == 0:
        logger.info(
            f"Dataset split complete: {len(train_data):,} training samples | "
            f"{len(val_data):,} validation samples."
        )

    # 4. DPO Config Setup
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
    )

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}'")
        callbacks.append(R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix))

    # 5. Initialize & Run DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Passing None creates an implicit reference copy of the policy model
        args=dpo_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if local_rank == 0:
        logger.info("⚖️ Starting Stage 3: Direct Preference Optimization (DPO)...")

    trainer.train()

    if local_rank == 0:
        logger.info(f"Saving final DPO model weights to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Stage 3: DPO Preference Alignment completed successfully!")

        if enable_r2_sync:
            rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
            target_r2_path = f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
            logger.info(f"📤 Syncing final DPO model weights to Cloudflare R2 ({target_r2_path})...")
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
            logger.info("✅ Final DPO weights successfully uploaded to Cloudflare R2!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()