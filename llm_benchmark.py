import os
import json
import argparse
import logging
import numpy as np
import torch
import torch.serialization

# Disable Hugging Face Rust parallelism warnings/deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Enable TF32 for GPU acceleration
torch.set_float32_matmul_precision("high")

# 2. Bypass cuDNN attention backend during Evaluation / no_grad
torch.backends.cuda.enable_cudnn_sdp(False)

# 3. Comprehensive allowlist for NumPy types in PyTorch 2.6+
safe_numpy_types = [np.dtype, np.ndarray]
for mod_path in ["numpy._core.multiarray", "numpy.core.multiarray", "numpy._core.numerictypes"]:
    try:
        mod = __import__(mod_path, fromlist=["scalar", "_reconstruct"])
        if hasattr(mod, "scalar"):
            safe_numpy_types.append(mod.scalar)
        if hasattr(mod, "_reconstruct"):
            safe_numpy_types.append(mod._reconstruct)
    except (ImportError, AttributeError):
        pass
try:
    torch.serialization.add_safe_globals(safe_numpy_types)
except Exception:
    pass

# 4. Universal Fail-Safe: Force trusted local checkpoints to bypass strict weights_only check
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

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
        default="Qwen/Qwen3.5-9B", 
        help="Tokenizer checkpoint name/path to load if not embedded in checkpoint_path."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=None, 
        help="Explicit vocabulary size override (if None, auto-detected from tokenizer)."
    )
    
    # Architecture Flags (Aligned with BareTorch Defaults)
    parser.add_argument("--d_model", type=int, default=2048, help="Model hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention/mixer heads.")
    parser.add_argument("--num_layers", type=int, default=24, help="Total transformer/mixer layers.")
    parser.add_argument(
        "--layer_sequence", 
        type=str, 
        default="cs_lrad,cs_lrad,cs_lrad,transformer", 
        help="Comma-separated layer pattern sequence."
    )
    parser.add_argument("--chunk_size", type=int, default=32, help="CS-LRAD chunk size.")
    parser.add_argument("--rank", type=int, default=16, help="CS-LRAD projection rank.")
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
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (e.g., cuda, cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Path to save output JSON.")
    
    return parser.parse_args()


def build_baretorch_config(args, resolved_vocab_size: int, config_file: str = None) -> BareTorchConfig:
    """Builds BareTorchConfig directly from config.json if available, else constructs from CLI args."""
    if config_file and os.path.exists(config_file):
        logger.info(f"Loading BareTorch configuration directly from '{config_file}'...")
        try:
            cfg = BareTorchConfig.from_json_file(config_file)
            if resolved_vocab_size is not None:
                cfg.vocab_size = resolved_vocab_size
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load via BareTorchConfig.from_json_file ({e}). Constructing from flags...")

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
        tie_word_embeddings=False,
    )


def load_baretorch_model(args, resolved_vocab_size: int, device: str, dtype: torch.dtype):
    """Loads model with explicit FSDP/DDP key cleaning, dynamic tied-embedding binding, and strict error handling."""
    checkpoint_path = args.checkpoint_path
    logger.info(f"Loading BareTorch model from checkpoint: '{checkpoint_path}'")
    
    config_file = os.path.join(checkpoint_path, "config.json") if os.path.isdir(checkpoint_path) else os.path.join(os.path.dirname(checkpoint_path), "config.json")
    config = build_baretorch_config(args, resolved_vocab_size, config_file)
    
    model = BareTorchForCausalLM(config)

    # 1. Locate weight file
    if os.path.isdir(checkpoint_path):
        sf_file = os.path.join(checkpoint_path, "model.safetensors")
        bin_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(sf_file):
            from safetensors.torch import load_file
            state_dict = load_file(sf_file)
        elif os.path.exists(bin_file):
            state_dict = torch.load(bin_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"❌ No 'model.safetensors' or 'pytorch_model.bin' found in '{checkpoint_path}'")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # 2. Clean FSDP / DDP / torch.compile prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["_fsdp_wrapped_module.", "module.", "_orig_mod."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned_state_dict[new_k] = v

    # 2b. Dynamic search & bind for missing lm_head.weight (Tied Word Embeddings)
    if "lm_head.weight" not in cleaned_state_dict:
        candidate_key = None
        
        # Primary search: 2D weight tensors containing embedding/token/wte/word keywords
        for k, v in cleaned_state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                if any(term in k.lower() for term in ["embed", "tok", "wte", "word", "emb"]):
                    candidate_key = k
                    break

        # Fallback search: any 2D tensor whose 1st dimension matches target vocab size
        if not candidate_key:
            for k, v in cleaned_state_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    if v.shape[0] == resolved_vocab_size or v.shape[0] == config.vocab_size:
                        candidate_key = k
                        break

        if candidate_key:
            cleaned_state_dict["lm_head.weight"] = cleaned_state_dict[candidate_key]
            logger.info(f"🔗 Dynamic Tie Success: Mapped 'lm_head.weight' -> '{candidate_key}' (Shape: {list(cleaned_state_dict[candidate_key].shape)})")
        else:
            logger.warning("⚠️ Could not automatically detect token embedding key in checkpoint state_dict.")

    # 3. Load state dict and verify key alignment
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    logger.info("=" * 70)
    logger.info("📦 CHECKPOINT WEIGHT LOADING REPORT:")
    logger.info(f"  ├─ Total keys in file : {len(cleaned_state_dict)}")
    logger.info(f"  ├─ Missing keys        : {len(missing_keys)}")
    logger.info(f"  └─ Unexpected keys     : {len(unexpected_keys)}")
    logger.info("=" * 70)

    # STRICT GUARD: Crash immediately if any parameters fail to map
    if len(missing_keys) > 0:
        raise RuntimeError(
            f"❌ CRITICAL WEIGHT LOADING ERROR: {len(missing_keys)} parameter tensor(s) failed to load!\n"
            f"The checkpoint is missing weights for key architectural components.\n"
            f"Execution halted to prevent running on uninitialized random weights.\n"
            f"Sample missing keys: {missing_keys[:5]}"
        )

    if len(unexpected_keys) > 0:
        logger.warning(f"⚠️ Unexpected keys in checkpoint file (ignored): {unexpected_keys[:5]}")

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
    logger.info(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load tokenizer from '{tokenizer_path}' ({e}). Falling back to '{args.tokenizer_name}'.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enforce right-padding so Token 0 always sits at Index 0 of Chunk 0
    tokenizer.padding_side = "right"

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
        padding_side="right",
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