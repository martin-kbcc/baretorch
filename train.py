import os
import argparse
import logging
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset

# Import all BareTorch configurations and causal language models
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
#                               Logging Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Mute chatty background connection threads from downloading dataset shards
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Map command-line strings to our custom model architectures
MODEL_MAP = {
    "cs_lrad": (CSLRADConfig, CSLRADForCausalLM),
    "transformer": (TransformerConfig, TransformerForCausalLM),
    "cs_ttt": (CSTTTConfig, CSTTTForCausalLM),
    "cs_lrad_transformer": (CSLRADTransformerConfig, CSLRADTransformerForCausalLM),
    "cs_ttt_transformer": (CSTTTTransformerConfig, CSTTTTransformerForCausalLM),
    "cs_lrad_cs_ttt_transformer": (CSLRADCSTTTTransformerConfig, CSLRADCSTTTTransformerForCausalLM),
    "baretorch": (BareTorchConfig, BareTorchForCausalLM),
}


def tokenize_and_chunk(tokenizer, max_length=1024):
    """
    Groups tokenized streams into raw sequence blocks of exactly 'max_length'.
    This avoids wasteful padding tokens and ensures maximum pre-training efficiency.
    """
    def process_fn(batch):
        tokenized = tokenizer(batch["text"], truncation=False, padding=False)
        concatenated_ids = []
        for ids in tokenized["input_ids"]:
            concatenated_ids.extend(ids)
            
        # Group tokens into equal-sized block chunks
        total_length = len(concatenated_ids)
        total_length = (total_length // max_length) * max_length
        
        result = {
            "input_ids": [
                concatenated_ids[i : i + max_length]
                for i in range(0, total_length, max_length)
            ]
        }
        # Copy inputs to labels for Hugging Face causal autoregressive loss tracking
        result["labels"] = result["input_ids"].copy()
        return result
    return process_fn


def main():
    parser = argparse.ArgumentParser(description="BareTorch Cluster-Scale Pre-training Engine")
    
    # 1. High-Level Architecture Parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="baretorch",
        choices=list(MODEL_MAP.keys()),
        help="The specific BareTorch model architecture configuration to train."
    )
    parser.add_argument(
        "--layer_sequence",
        type=str,
        default="cs_lrad,cs_lrad,cs_lrad,transformer",
        help="Comma-separated sequence of layer types (only used for baretorch model_type)."
    )
    
    # 2. Optimization Parameters
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay parameter.")
    
    # 3. Structural Dimensions
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of query attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Total layer count.")
    parser.add_argument("--chunk_size", type=int, default=32, help="Sequence chunk segmentation size.")
    parser.add_argument("--rank", type=int, default=8, help="Internal low-rank projection parameter.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout percentage.")
    parser.add_argument("--seq_len", type=int, default=1024, help="Maximum training sequence context length.")
    
    # 4. Engine & Workload Configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU device.")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch model compilation.")
    parser.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    
    # 5. Monitoring & Saving Checkpoints
    parser.add_argument("--logging_steps", type=int, default=50, help="Log step intervals.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint step intervals.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run validation step intervals.")
    
    args = parser.parse_args()

    logger.info(f"Initializing BareTorch Engine. Selected Model Type: {args.model_type}")

    # 1. Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # 2. Build Config Args dynamically from parameters
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
    
    # Check if the master hybrid is selected
    if args.model_type == "baretorch":
        # Parse sequence and tile it cleanly up to args.num_layers
        raw_sequence = [s.strip().lower() for s in args.layer_sequence.split(",") if s.strip()]
        layer_types = [raw_sequence[i % len(raw_sequence)] for i in range(args.num_layers)]
        config_args["layer_types"] = layer_types
        logger.info(f"Assembled Hybrid Sequence ({args.num_layers} layers): {layer_types}")
        
    # Inject context bounds and GQA allocations where supported
    if "transformer" in args.model_type or args.model_type == "baretorch":
        config_args["max_seq_len"] = args.seq_len
        # Allocate Grouped-Query Attention (GQA) with 1/4th key-value head divisor
        config_args["num_kv_heads"] = max(1, args.num_heads // 4)

    # Instantiate Config and Model
    config = config_cls(**config_args)
    model = model_cls(config)
    
    # Print parameter count for validation
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized successfully. Total parameters: {total_params / 1e6:.2f}M")

    # 3. Stream the Curated High-Density Pre-training Dataset
    logger.info("Loading streaming shards of HuggingFaceFW/dclm_100BT-shuffled...")
    raw_dataset = load_dataset("HuggingFaceFW/dclm_100BT-shuffled", split="train", streaming=True)

    # Dedicate the first 1,000 raw documents to our static validation set.
    logger.info("Slicing dataset stream into isolated train and validation subsets...")
    raw_val_dataset = raw_dataset.take(1000)
    raw_train_dataset = raw_dataset.skip(1000)

    # Map both datasets through our fast-chunking pipeline
    processed_train_dataset = raw_train_dataset.map(
        tokenize_and_chunk(tokenizer, max_length=args.seq_len),
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )
    
    processed_val_dataset = raw_val_dataset.map(
        tokenize_and_chunk(tokenizer, max_length=args.seq_len),
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    # 4. Configure our unified Hugging Face Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./output_{args.model_type}",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,                          # Utilize native cluster/4090 BF16 precision
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        
        # DYNAMIC EVALUATION CONFIGURATION
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=f"./runs/{args.model_type}",
        torch_compile=args.compile,
        ddp_find_unused_parameters=False,
    )

    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        data_collator=default_data_collator,
    )

    # 6. Execute Training
    logger.info(f"Starting pre-training run...")
    trainer.train()
    logger.info("Pre-training run successfully completed!")

    # 7. Clean Distributed Process Group & Hard Exit
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    import os
    logger.info("Exiting cleanly...")
    os._exit(0)


if __name__ == "__main__":
    main()