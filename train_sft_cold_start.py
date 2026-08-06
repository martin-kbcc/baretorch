import argparse
import logging
import os
import subprocess
from datasets import load_dataset
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
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


def preprocess_chatml_example(messages, tokenizer, max_seq_len=2048):
  """Formats multi-turn dialogues into ChatML format with turn-aware truncation.

  Guarantees that no assistant turns are sliced in half and every included turn
  ends with <|im_end|>. Assigns -100 to user/system tokens for loss masking.
  """
  input_ids = []
  labels = []
  has_assistant_tokens = False

  for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")

    # Format ChatML block for this turn
    formatted_turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
    tokens = tokenizer.encode(formatted_turn, add_special_tokens=False)

    # Stop adding turns if this message would exceed max_seq_len
    if len(input_ids) + len(tokens) > max_seq_len:
      break

    input_ids.extend(tokens)

    # Apply loss masking (compute loss ONLY on assistant completions)
    if role == "assistant":
      labels.extend(tokens)
      has_assistant_tokens = True
    else:
      labels.extend([-100] * len(tokens))

  # Fallback safety: guarantee at least one valid assistant target token exists
  if not has_assistant_tokens and len(input_ids) > 0:
    labels[-1] = input_ids[-1]

  return {
      "input_ids": input_ids,
      "labels": labels,
      "attention_mask": [1] * len(input_ids),
  }


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

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": question},
      {"role": "assistant", "content": assistant_content},
  ]
  return messages


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
      default="./checkpoints_500m_simpo/checkpoint-simpo-final",
      help="Path to Stage 2 SimPO checkpoint.",
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
      default=2,
      help="Warmup SFT epochs (1-2 is usually sufficient).",
  )
  parser.add_argument(
      "--batch_size", type=int, default=8, help="Per-GPU batch size."
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
  parser.add_argument("--warmup_steps", type=int, default=50)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--seq_len", type=int, default=1024)
  parser.add_argument(
      "--compile",
      action="store_true",
      help="Enable targeted torch.compile for CS-LRAD sub-modules.",
  )
  parser.add_argument(
      "--grad_checkpointing",
      action="store_true",
      help="Enable gradient checkpointing.",
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

  # 1. Tokenizer Setup with ChatML & Thinking Tokens
  if local_rank == 0:
    logger.info(
        f"Initializing Tokenizer '{args.tokenizer_name}' with ChatML/Thinking"
        " Special Tokens..."
    )

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
  tokenizer.model_max_length = args.seq_len

  special_tokens_dict = {
      "additional_special_tokens": [
          "<|im_start|>",
          "<|im_end|>",
          "<think>",
          "</think>",
      ],
      "pad_token": "<|im_end|>",
  }
  tokenizer.add_special_tokens(special_tokens_dict)

  # 2. Model Loading & Safe Token Embedding Resizing
  if local_rank == 0:
    logger.info(
        f"Loading SimPO model weights from: {args.pretrained_model_path}"
    )

  model = BareTorchForCausalLM.from_pretrained(args.pretrained_model_path)

  # Safely resize token embeddings to include <think> and </think>
  model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

  # Model runtime configurations
  model.config.pad_token_id = tokenizer.pad_token_id
  model.config.use_cache = False  # Disable KV caching during training
  model.config.use_grad_checkpointing = args.grad_checkpointing

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

  # 3. Load & Process Dataset
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

  if local_rank == 0:
    logger.info(
        "Formatting GSM8K problems into CoT <think>...</think> ChatML"
        " structure..."
    )

  def map_fn(example):
    messages = format_gsm8k_to_chatml(example)
    return preprocess_chatml_example(
        messages, tokenizer, max_seq_len=args.seq_len
    )

  processed_dataset = raw_dataset.map(
      map_fn,
      batched=False,
      remove_columns=raw_dataset.column_names,
      num_proc=4,
      desc="Formatting CoT Math SFT Data",
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
      torch_compile=False,  # Explicitly False; targeted compilation applied directly to CS-LRAD above
      gradient_checkpointing=args.grad_checkpointing,
      gradient_checkpointing_kwargs={"use_reentrant": False},
      ddp_find_unused_parameters=False,
      dataloader_num_workers=4,
      dataloader_pin_memory=True,
  )

  # Detect R2 sync activation via argument or environment variables
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
  else:
    if local_rank == 0:
      logger.info(
          "R2 sync disabled. Running in local mode (disk checkpoints only)."
      )

  # Sequence padding collator guaranteed to pad to multiples of 32 for CS-LRAD/CS-TTT
  data_collator = DataCollatorForSeq2Seq(
      tokenizer=tokenizer,
      model=model,
      padding=True,
      pad_to_multiple_of=32,
      label_pad_token_id=-100,
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_data,
      eval_dataset=val_data,
      data_collator=data_collator,
      callbacks=callbacks,
  )

  # 5. Launch Supervised Fine-Tuning
  if local_rank == 0:
    logger.info("🔥 Starting Stage 3-Prep: Cold-Start CoT SFT Warmup...")

  trainer.train()

  # Save final model and tokenizer config
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