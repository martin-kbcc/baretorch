import os
import argparse
import logging
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from baretorch import BareTorchConfig, BareTorchForCausalLM

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def preprocess_chatml_example(example, tokenizer, max_seq_len=2048):
    """
    Formats multi-turn dialogues into ChatML format with turn-aware truncation.
    Guarantees that no assistant turns are sliced in half and every included turn 
    ends with <|im_end|>. Assigns -100 to user/system tokens for loss masking.
    """
    messages = example.get("messages", [])
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
        "attention_mask": [1] * len(input_ids)
    }


def main():
    # Distributed Initialization for torchrun
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(description="BareTorch Stage 1: Supervised Fine-Tuning (SFT) Engine")
    
    # Checkpoint Paths
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pre-trained BareTorch checkpoint folder.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_100m_sft", help="Directory to save fine-tuned SFT weights.")
    
    # Dataset Parameters
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/smoltalk", help="Hugging Face SFT dataset path.")
    parser.add_argument("--dataset_config", type=str, default="all", help="Dataset subset/config name (e.g., 'all' or 'everyday-conversations').")
    parser.add_argument("--max_samples", type=int, default=100000, help="Sub-sample N rows for ultra-dense fast SFT. Set to 0 for full dataset.")
    
    # Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12, help="Per-GPU batch size.")
    parser.add_argument("--grad_accum", type=int, default=3, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="SFT learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. Tokenizer Setup with ChatML Special Tokens
    if local_rank == 0:
        logger.info("Initializing Tokenizer and ChatML Special Tokens...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = args.seq_len  # Explicitly set max model length to 2048
    
    special_tokens_dict = {
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "pad_token": "<|im_end|>"
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # 2. Model Loading & Safe Token Embedding Resizing
    if local_rank == 0:
        logger.info(f"Loading pre-trained model weights from: {args.pretrained_model_path}")

    model = BareTorchForCausalLM.from_pretrained(args.pretrained_model_path)
    
    # Safely resize token embeddings without multivariate covariance estimation
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    # Model runtime configurations
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV caching during training
    model.config.use_grad_checkpointing = args.grad_checkpointing

    # 3. Load & Process Dataset
    if local_rank == 0:
        logger.info(f"Loading SFT Dataset '{args.dataset_name}' (Config: {args.dataset_config})...")

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")

    if args.max_samples > 0 and len(raw_dataset) > args.max_samples:
        if local_rank == 0:
            logger.info(f"✂️ Sub-sampling dataset from {len(raw_dataset):,} rows to {args.max_samples:,} rows.")
        raw_dataset = raw_dataset.select(range(args.max_samples))

    if local_rank == 0:
        logger.info("Formatting multi-turn conversations and building assistant loss masks...")

    processed_dataset = raw_dataset.map(
        lambda example: preprocess_chatml_example(example, tokenizer, max_seq_len=args.seq_len),
        batched=False,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
        desc="Formatting ChatML SFT Data",
    )

    # Split 5% off for validation
    dataset_split = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]

    if local_rank == 0:
        logger.info(f"Dataset split complete: {len(train_data):,} training samples | {len(val_data):,} validation samples.")

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
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="tensorboard",
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Sequence padding collator guaranteed to pad to multiples of 32 for CS-LRAD/CS-TTT
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=32,
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )

    # 5. Launch Supervised Fine-Tuning
    if local_rank == 0:
        logger.info("🔥 Starting Stage 1: Supervised Fine-Tuning (SFT)...")

    trainer.train()

    # Save final model and tokenizer config
    if local_rank == 0:
        logger.info(f"Saving final SFT checkpoint to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Stage 1: Supervised Fine-Tuning completed successfully!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()