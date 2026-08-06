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
    CSLRADConfig,
    CSLRADForCausalLM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BareTorch LLM Evaluation Suite")
    
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
        default="gpt2", 
        help="Tokenizer checkpoint/name to use (if not found in checkpoint dir)."
    )
    
    # Architecture Overrides (Matching launch_lrad_hybrid.sh defaults for 1B model)
    parser.add_argument("--d_model", type=int, default=1536, help="Model hidden dimension.")
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
    
    # Task Selection
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="gsm8k,mmlu,arc_challenge,hellaswag,winogrande",
        help="Comma-separated list of lm-eval tasks."
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot count (0 for zero-shot).")
    parser.add_argument("--limit", type=float, default=None, help="Sample limit per task for smoke testing.")
    
    # Execution & Precision
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", type=str, default="benchmark_results_1b.json", help="Path to save output JSON.")
    
    return parser.parse_args()


def build_baretorch_config(args, config_file: str = None) -> BareTorchConfig:
    """Builds BareTorchConfig from file if available, otherwise constructs from CLI args."""
    if config_file and os.path.exists(config_file):
        logger.info(f"Loading BareTorch configuration from '{config_file}'...")
        try:
            return AutoConfig.from_pretrained(config_file)
        except Exception as e:
            logger.warning(f"Failed to load via AutoConfig ({e}). Constructing BareTorchConfig manually...")

    # Expand layer sequence pattern across total layers
    pattern = [s.strip() for t in args.layer_sequence.split(",") if (s := t.strip())]
    repeats = (args.num_layers + len(pattern) - 1) // len(pattern)
    full_layer_types = (pattern * repeats)[:args.num_layers]

    logger.info(
        f"Constructing BareTorchConfig: d_model={args.d_model}, num_layers={args.num_layers}, "
        f"num_heads={args.num_heads}, layer_pattern='{args.layer_sequence}'"
    )
    
    return BareTorchConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        layer_types=full_layer_types,
        chunk_size=args.chunk_size,
        rank=args.rank,
    )


def load_baretorch_model(args, device: str, dtype: torch.dtype):
    """
    Robust checkpoint loader supporting HuggingFace directories and raw state_dicts,
    fully compliant with 1B parameter configurations.
    """
    checkpoint_path = args.checkpoint_path
    logger.info(f"Loading BareTorch model from checkpoint: '{checkpoint_path}'")
    
    if os.path.isdir(checkpoint_path):
        # Hugging Face Directory Load
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded model via AutoModelForCausalLM.")
        except Exception as e:
            logger.warning(f"AutoModel load failed ({e}). Attempting BareTorchForCausalLM load...")
            config_file = os.path.join(checkpoint_path, "config.json")
            config = build_baretorch_config(args, config_file)
            
            model = BareTorchForCausalLM.from_pretrained(
                checkpoint_path, 
                config=config, 
                torch_dtype=dtype
            )
            
    elif os.path.isfile(checkpoint_path):
        # Raw PyTorch .pt or .bin state_dict file
        logger.info("Detected raw state_dict file. Initializing model structure...")
        ckpt_dir = os.path.dirname(checkpoint_path)
        config_file = os.path.join(ckpt_dir, "config.json")
        
        config = build_baretorch_config(args, config_file)
        model = BareTorchForCausalLM(config)
        
        logger.info(f"Loading state dict from '{checkpoint_path}'...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Unwrap state_dict if saved inside a wrapper dictionary
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys during load: {missing_keys[:5]} ... (total {len(missing_keys)})")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during load: {unexpected_keys[:5]} ... (total {len(unexpected_keys)})")
            
        model = model.to(dtype=dtype)
    else:
        raise FileNotFoundError(f"Checkpoint path not found: '{checkpoint_path}'")

    model = model.to(device).eval()
    return model


def main():
    args = parse_args()
    
    # 1. Setup Torch Precision
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    eval_dtype = dtype_map[args.dtype]
    
    # 2. Resolve Tokenizer
    tokenizer_path = args.checkpoint_path if os.path.isdir(args.checkpoint_path) else args.tokenizer_name
    logger.info(f"Loading tokenizer from: '{tokenizer_path}'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from '{tokenizer_path}' ({e}). Falling back to '{args.tokenizer_name}'.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load BareTorch Model
    model = load_baretorch_model(args, args.device, eval_dtype)
    
    # 4. Wrap Model into lm-evaluation-harness HFLM Class
    logger.info("Wrapping BareTorch model into lm-evaluation-harness interface...")
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # 5. Parse Tasks
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    logger.info(f"Starting evaluation across tasks: {task_list}")
    
    # 6. Run Evaluation Suite
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    
    # 7. Print Results Table
    print("\n" + "=" * 70)
    print("📊 BARETORCH 1B FOUNDATIONAL BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    
    formatted_summary = {}
    if "results" in results:
        for task_name, metrics in results["results"].items():
            primary_metric = (
                metrics.get("acc,none") 
                or metrics.get("exact_match,none") 
                or metrics.get("acc_norm,none")
                or list(metrics.values())[0]
            )
            if isinstance(primary_metric, float):
                formatted_summary[task_name] = f"{primary_metric * 100:.2f}%"
                print(f"  ├─ {task_name:<20} : {primary_metric * 100:.2f}%")
            else:
                formatted_summary[task_name] = str(primary_metric)
                print(f"  ├─ {task_name:<20} : {primary_metric}")
    
    print("=" * 70 + "\n")
    
    # 8. Save JSON Results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, "w") as f:
        json.dump(results["results"], f, indent=4, default=str)
        
    logger.info(f"Full benchmark metrics saved to '{args.output_file}'")


if __name__ == "__main__":
    main()