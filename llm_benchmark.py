# /home/martinkb/Desktop/BareTorch_F/llm_benchmark.py
import os
import json
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

# Import BareTorch framework
import baretorch
from baretorch import (
    BareTorchConfig,
    BareTorchForCausalLM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BareTorch Universal LLM Evaluation Suite")
    
    # Model & Checkpoint Paths
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to saved HuggingFace checkpoint directory or PyTorch state_dict file."
    )
    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default="HuggingFaceTB/SmolLM2-360M", 
        help="Tokenizer checkpoint name/path to load if not embedded in checkpoint_path."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=None, 
        help="Explicit vocabulary size override (if None, auto-detected from tokenizer)."
    )
    
    # Architecture Flags (Fallback if config.json is missing)
    parser.add_argument("--d_model", type=int, default=1152, help="Model hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention/mixer heads.")
    parser.add_argument("--num_layers", type=int, default=24, help="Total transformer/mixer layers.")
    parser.add_argument(
        "--layer_sequence", 
        type=str, 
        default="cs_lrad,cs_lrad,cs_lrad,transformer", 
        help="Comma-separated layer pattern sequence."
    )
    parser.add_argument("--chunk_size", type=int, default=32, help="CS-LRAD chunk size.")
    parser.add_argument("--rank", type=int, default=8, help="CS-LRAD projection rank.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence context length.")
    
    # Task & Evaluation Flags
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="mmlu,arc_challenge,arc_easy,hellaswag,winogrande",
        help="Comma-separated list of lm-eval tasks."
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot count (0 for zero-shot).")
    parser.add_argument("--limit", type=float, default=None, help="Sample limit per task for smoke testing.")
    
    # Execution & Precision Flags
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (e.g., cuda, cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Path to save output JSON.")
    
    return parser.parse_args()


def build_baretorch_config(args, resolved_vocab_size: int, config_file: str = None) -> BareTorchConfig:
    """Builds BareTorchConfig from file if available, otherwise constructs from CLI args."""
    if config_file and os.path.exists(config_file):
        logger.info(f"Loading BareTorch configuration directly from '{config_file}'...")
        try:
            cfg = AutoConfig.from_pretrained(config_file)
            if resolved_vocab_size is not None:
                cfg.vocab_size = resolved_vocab_size
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load via AutoConfig ({e}). Constructing BareTorchConfig manually from flags...")

    pattern = [s.strip() for t in args.layer_sequence.split(",") if (s := t.strip())]
    repeats = (args.num_layers + len(pattern) - 1) // len(pattern)
    full_layer_types = (pattern * repeats)[:args.num_layers]

    logger.info(
        f"Constructing BareTorchConfig from CLI flags: d_model={args.d_model}, num_layers={args.num_layers}, "
        f"num_heads={args.num_heads}, vocab_size={resolved_vocab_size}, layer_pattern='{args.layer_sequence}'"
    )
    
    return BareTorchConfig(
        vocab_size=resolved_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        layer_types=full_layer_types,
        chunk_size=args.chunk_size,
        rank=args.rank,
        max_seq_len=args.max_seq_len,
    )


def load_baretorch_model(args, resolved_vocab_size: int, device: str, dtype: torch.dtype):
    """Loads model cleanly from checkpoint directory or PyTorch state_dict file."""
    checkpoint_path = args.checkpoint_path
    logger.info(f"Loading BareTorch model from checkpoint: '{checkpoint_path}'")
    
    if os.path.isdir(checkpoint_path):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded model via AutoModelForCausalLM.")
        except Exception as e:
            logger.warning(f"AutoModel load failed ({e}). Falling back to BareTorchForCausalLM load...")
            config_file = os.path.join(checkpoint_path, "config.json")
            config = build_baretorch_config(args, resolved_vocab_size, config_file)
            model = BareTorchForCausalLM.from_pretrained(
                checkpoint_path, 
                config=config, 
                torch_dtype=dtype
            )
            
    elif os.path.isfile(checkpoint_path):
        ckpt_dir = os.path.dirname(checkpoint_path)
        config_file = os.path.join(ckpt_dir, "config.json")
        
        config = build_baretorch_config(args, resolved_vocab_size, config_file)
        model = BareTorchForCausalLM(config)
        
        logger.info(f"Loading state dict from file '{checkpoint_path}'...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys during load: {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys during load: {unexpected[:5]}")
    else:
        raise FileNotFoundError(f"Checkpoint path not found: '{checkpoint_path}'")

    model = model.to(dtype=dtype).to(device).eval()
    return model


def extract_primary_metric(metrics: dict):
    """Safely extracts primary task metric avoiding 0.0 falsy key skips."""
    candidate_keys = [
        "acc_norm,none", "acc,none", "exact_match,none", 
        "acc_norm,flexible", "acc,flexible", "exact_match,flexible"
    ]
    for key in candidate_keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    
    for val in metrics.values():
        if isinstance(val, (int, float)):
            return val
    return "N/A"


def main():
    args = parse_args()
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    eval_dtype = dtype_map[args.dtype]
    
    # 1. Tokenizer Setup
    tokenizer_path = args.checkpoint_path if os.path.isdir(args.checkpoint_path) else args.tokenizer_name
    logger.info(f"Loading tokenizer from: '{tokenizer_path}'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from '{tokenizer_path}' ({e}). Falling back to '{args.tokenizer_name}'.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Resolve Final Vocab Size
    resolved_vocab_size = args.vocab_size if args.vocab_size is not None else len(tokenizer)
    logger.info(f"Target vocab_size resolved to: {resolved_vocab_size}")
    
    # 3. Model Loading
    model = load_baretorch_model(args, resolved_vocab_size, args.device, eval_dtype)
    
    # 4. Wrap with lm-evaluation-harness
    logger.info("Wrapping BareTorch model into lm-evaluation-harness interface...")
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    
    # 5. Execute Evaluation
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    logger.info(f"Starting evaluation across tasks: {task_list}")
    
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    
    print("\n" + "=" * 70)
    print("📊 BARETORCH BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    
    formatted_summary = {}
    if "results" in results:
        for task_name, metrics in results["results"].items():
            primary_metric = extract_primary_metric(metrics)
            if isinstance(primary_metric, (float, int)):
                val_str = f"{primary_metric * 100:.2f}%"
            else:
                val_str = str(primary_metric)
                
            formatted_summary[task_name] = val_str
            print(f"  ├─ {task_name:<20} : {val_str}")
    
    print("=" * 70 + "\n")
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, "w") as f:
        json.dump(results.get("results", {}), f, indent=4, default=str)
        
    logger.info(f"Full benchmark metrics saved to '{args.output_file}'")


if __name__ == "__main__":
    main()