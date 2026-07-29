import os
import json
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

# Import BareTorch framework to ensure AutoClasses and model mappings are registered
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
        "--model_type", 
        type=str, 
        default="baretorch", 
        choices=["baretorch", "cs_lrad", "transformer"],
        help="Architecture key used if initializing manually from config."
    )
    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default="gpt2", 
        help="Tokenizer checkpoint to use."
    )
    
    # Task Selection
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="gsm8k,mmlu,arc_challenge,hellaswag,winogrande",
        help="Comma-separated list of lm-eval tasks (e.g., 'gsm8k,mmlu,arc_challenge')."
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0, 
        help="Number of few-shot examples (default: 0 for zero-shot)."
    )
    parser.add_argument(
        "--limit", 
        type=float, 
        default=None, 
        help="Limit number of samples per task (useful for fast smoke-testing, e.g., 50 or 0.1)."
    )
    
    # Execution & Precision
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Path to save output JSON.")
    
    return parser.parse_args()


def load_baretorch_model(checkpoint_path: str, device: str, dtype: torch.dtype):
    """
    Robust checkpoint loader that supports standard HuggingFace load directories 
    or raw state_dict checkpoints.
    """
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
            logger.warning(f"AutoModel load failed ({e}). Falling back to manual config load...")
            config = AutoConfig.from_pretrained(checkpoint_path)
            model = BareTorchForCausalLM.from_pretrained(
                checkpoint_path, 
                config=config, 
                torch_dtype=dtype
            )
    elif os.path.isfile(checkpoint_path):
        # Raw PyTorch .pt or .bin state_dict file
        logger.info("Detected raw state_dict file. Loading weights into BareTorch model...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Look for accompanying config.json in the same directory
        ckpt_dir = os.path.dirname(checkpoint_path)
        config_file = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(config_file):
            config = AutoConfig.from_pretrained(config_file)
        else:
            logger.warning("No config.json found in checkpoint dir. Initializing default BareTorch 100M Hybrid config.")
            config = BareTorchConfig(
                d_model=512,
                num_heads=16,
                num_layers=12,
                layer_types=["cs_lrad", "cs_lrad", "cs_lrad", "transformer"] * 3,
            )
        
        model = BareTorchForCausalLM(config)
        model.load_state_dict(state_dict, strict=False)
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
    
    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer: '{args.tokenizer_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load BareTorch Model
    model = load_baretorch_model(args.checkpoint_path, args.device, eval_dtype)
    
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
    print("📊 BARETORCH BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    
    formatted_summary = {}
    if "results" in results:
        for task_name, metrics in results["results"].items():
            # Filter primary accuracy/match metrics
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