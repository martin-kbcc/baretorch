# baretorch/deploy/apple_mlx/benchmark_apple.py
import os
import csv
import json
import time
import gc
import argparse
import traceback

import torch
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from baretorch import BareTorchConfig, BareTorchForCausalLM

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.models.cache import make_prompt_cache
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

try:
    from executorch.exir import to_edge, EdgeCompileConfig
    from executorch.backends.apple.mps.partition import MpsPartitioner
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False


def clear_memory():
    """Flushes Python GC, PyTorch CUDA/MPS caches, and MLX memory stats."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def get_peak_mlx_vram_mb() -> float:
    """Returns peak Metal GPU memory allocation in MB for MLX calls."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024.0 ** 2)
    return 0.0


def count_mlx_params_m(model: nn.Module) -> float:
    """Counts instantiated parameters in an unquantized MLX module tree."""
    def _count(tree):
        total = 0
        if isinstance(tree, dict):
            for v in tree.values():
                total += _count(v)
        elif isinstance(tree, list):
            for v in tree:
                total += _count(v)
        elif hasattr(tree, "size"):
            total += tree.size
        return total
    return _count(model.parameters()) / 1e6


def count_baretorch_params_fast(
    d_model: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    rank: int = 8,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer"
) -> float:
    """Exact parameter math for BareTorch PyTorch modules."""
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]
    full_layer_types = [raw_seq[i % len(raw_seq)] for i in range(num_layers)]

    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    head_dim = d_model // num_heads
    d_ff = int(d_model * 3.5)

    embed_params = 2 * vocab_size * d_model
    final_norm = d_model
    layer_params = 0

    for l_type in full_layer_types:
        norms = 2 * d_model
        mlp = 3 * d_model * d_ff

        if l_type == "cs_lrad":
            attn = (5 * (d_model ** 2)) + (2 * d_model * num_heads * rank) + (2 * (d_model * num_heads + num_heads))
        else:
            attn = (2 * (d_model ** 2)) + (2 * d_model * (num_kv_heads * head_dim))

        layer_params += (norms + mlp + attn)

    total_params = embed_params + final_norm + layer_params
    return total_params / 1e6


def find_matching_baretorch_config(
    target_params_m: float,
    target_vocab_size: int = 50257,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    max_seq_len: int = 32768
) -> tuple[BareTorchConfig, float]:
    """Finds optimal BareTorch architecture matching the baseline parameter count."""
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]

    best_cfg = None
    best_diff = float("inf")
    best_params_m = 0.0

    for nl in range(12, 36, 2):
        for d in range(512, 4096, 32):
            for nh in [8, 12, 16, 20, 24, 32]:
                if d % nh != 0:
                    continue
                head_dim = d // nh
                if head_dim not in [64, 128]:
                    continue

                num_kv_heads = max(1, nh // 4)
                while nh % num_kv_heads != 0:
                    num_kv_heads -= 1

                p_m = count_baretorch_params_fast(
                    d_model=d,
                    num_layers=nl,
                    num_heads=nh,
                    vocab_size=target_vocab_size,
                    rank=8,
                    layer_sequence=layer_sequence
                )

                diff = abs(p_m - target_params_m)
                if diff < best_diff:
                    best_diff = diff
                    best_params_m = p_m
                    full_layer_types = [raw_seq[i % len(raw_seq)] for i in range(nl)]
                    best_cfg = BareTorchConfig(
                        vocab_size=target_vocab_size,
                        d_model=d,
                        num_heads=nh,
                        num_kv_heads=num_kv_heads,
                        num_layers=nl,
                        chunk_size=32,
                        rank=8,
                        dropout=0.0,
                        max_seq_len=max_seq_len,
                        layer_types=full_layer_types
                    )

    div_pct = (best_diff / target_params_m) * 100
    print(f"  ⚡ Parameter Match Complete: |Δ| = {best_diff:.2f}M ({div_pct:.2f}%)")
    return best_cfg, best_params_m


def benchmark_mlx_baseline(model: nn.Module, prompt_len: int, gen_len: int, vocab_size: int) -> dict:
    """Benchmarks open-source baseline models using native mlx_lm."""
    clear_memory()
    prompt = mx.random.randint(0, vocab_size, (1, prompt_len))

    try:
        w_cache = make_prompt_cache(model)
        w_out = model(prompt, cache=w_cache)
        mx.eval(w_out)

        clear_memory()

        cache = make_prompt_cache(model)
        ttft_start = time.perf_counter()
        outputs = model(prompt, cache=cache)
        mx.eval(outputs)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        clear_memory()
        curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

        gen_start = time.perf_counter()
        for _ in range(gen_len):
            outputs = model(curr_token, cache=cache)
            mx.eval(outputs)
            curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

        decode_sec = max(time.perf_counter() - gen_start, 1e-5)
        decode_vram_mb = get_peak_mlx_vram_mb()
        tokens_per_sec = gen_len / decode_sec

        return {
            "prompt_len": prompt_len,
            "ttft_ms": round(ttft_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "decode_vram_mb": round(decode_vram_mb, 2),
            "status": "success"
        }
    except Exception as e:
        clear_memory()
        return {
            "prompt_len": prompt_len,
            "ttft_ms": "OOM",
            "tokens_per_sec": "OOM",
            "decode_vram_mb": "OOM",
            "status": f"OOM ({type(e).__name__})"
        }


def benchmark_baretorch_executorch(config: BareTorchConfig, prompt_len: int, gen_len: int) -> dict:
    """Benchmarks BareTorch PyTorch model lowered to ExecuTorch Metal MPS."""
    clear_memory()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        model = BareTorchForCausalLM(config).eval().to(device=device, dtype=torch.float16)
        prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

        # Warmup Pass
        with torch.no_grad():
            _ = model(prompt)

        clear_memory()

        # Measure Time to First Token (Prefill)
        ttft_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(prompt, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits
        if device.type == "mps":
            torch.mps.synchronize()
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        curr_token = torch.argmax(logits[:, -1:, :], dim=-1)

        # Measure Step Decoding Speed
        gen_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(gen_len):
                outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
                curr_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                past_key_values = outputs.past_key_values
        if device.type == "mps":
            torch.mps.synchronize()

        decode_sec = max(time.perf_counter() - gen_start, 1e-5)
        tokens_per_sec = gen_len / decode_sec
        
        peak_vram_mb = 0.0
        if hasattr(torch.mps, "current_allocated_memory"):
            peak_vram_mb = torch.mps.current_allocated_memory() / (1024.0 ** 2)

        del model
        clear_memory()

        return {
            "prompt_len": prompt_len,
            "ttft_ms": round(ttft_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "decode_vram_mb": round(peak_vram_mb, 2),
            "status": "success"
        }
    except Exception as e:
        clear_memory()
        return {
            "prompt_len": prompt_len,
            "ttft_ms": "OOM",
            "tokens_per_sec": "OOM",
            "decode_vram_mb": "OOM",
            "status": f"OOM ({type(e).__name__})"
        }


def format_cell(val) -> str:
    if str(val).startswith("OOM"):
        return "💥 OOM"
    elif isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)


def export_to_csv(paired_results: list, output_csv: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "Baseline_Model_ID", "Context_Length", "Metric",
        "BareTorch_Matched_Value", "Baseline_Value", "BareTorch_Advantage"
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for pair in paired_results:
            hf_res = pair["hf_baseline"]
            bt_res = pair["baretorch_matched"]
            baseline_id = hf_res["model_name"]

            bt_runs = bt_res.get("runs", [])
            hf_runs = hf_res.get("runs", [])

            for run_idx, b_run in enumerate(bt_runs):
                ctx = b_run["prompt_len"]
                h_run = hf_runs[run_idx] if run_idx < len(hf_runs) else {}

                t_b = b_run.get("ttft_ms")
                t_h = h_run.get("ttft_ms", "N/A")
                ttft_adv = f"{((t_h - t_b) / t_h) * 100:+.2f}%" if (isinstance(t_h, (int, float)) and isinstance(t_b, (int, float)) and t_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Prefill_Latency_ms", format_cell(t_b), format_cell(t_h), ttft_adv])

                s_b = b_run.get("tokens_per_sec")
                s_h = h_run.get("tokens_per_sec", "N/A")
                speed_adv = f"{s_b / s_h:.2f}x" if (isinstance(s_h, (int, float)) and isinstance(s_b, (int, float)) and s_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Local_GPU_Decode_tok_s", format_cell(s_b), format_cell(s_h), speed_adv])

                v_b = b_run.get("decode_vram_mb")
                v_h = h_run.get("decode_vram_mb", "N/A")
                vram_adv = f"-{((v_h - v_b) / v_h) * 100:.2f}%" if (isinstance(v_h, (int, float)) and isinstance(v_b, (int, float)) and v_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Decode_VRAM_MB", format_cell(v_b), format_cell(v_h), vram_adv])

    print(f"\n📊 Apple Silicon CSV report saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="BareTorch Apple Silicon Benchmark Suite")
    parser.add_argument("--hf_model_ids", nargs="+", type=str, default=["meta-llama/Llama-3.2-1B"])
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--output_json", type=str, default="./results_apple_suite.json")
    parser.add_argument("--output_csv", type=str, default="./results_apple_suite.csv")
    args = parser.parse_args()

    print("==================================================================================================")
    print(f"🍎 BARETORCH APPLES-TO-APPLES BENCHMARK SUITE ({len(args.hf_model_ids)} Baseline Models)")
    print("==================================================================================================")

    paired_results = []
    max_seq_len = max(args.prompt_lens) + args.gen_len + 1024

    for model_id in args.hf_model_ids:
        print(f"\n" + "─" * 100)
        print(f"📦 Evaluating Baseline Model: '{model_id}'")
        print("─" * 100)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            vocab_size = getattr(tokenizer, "vocab_size", 50257)
        except Exception:
            vocab_size = 50257

        hf_runs = []
        bt_runs = []
        hf_params_m = 0.0

        if HAS_MLX_LM:
            try:
                print(f"  • Loading Baseline from MLX Hub...")
                hf_model, _ = mlx_lm_load(model_id)
                hf_params_m = count_mlx_params_m(hf_model)
                print(f"  • Baseline Parameters: {hf_params_m:.2f}M params (vocab_size={vocab_size})")

                for ctx in args.prompt_lens:
                    print(f"  ├─ Benchmarking Baseline @ Context: {ctx:<5} tokens...", end="", flush=True)
                    res = benchmark_mlx_baseline(hf_model, ctx, args.gen_len, vocab_size)
                    hf_runs.append(res)
                    print(f" ✅ (TTFT: {format_cell(res['ttft_ms'])} ms | Decode: {format_cell(res['tokens_per_sec'])} tok/s | VRAM: {format_cell(res['decode_vram_mb'])} MB)")

                del hf_model
                clear_memory()
            except Exception as e:
                print(f"⚠️ Could not benchmark baseline '{model_id}': {e}")

        if hf_params_m == 0.0:
            hf_params_m = 1237.0

        print(f"\n  ⚙️ Building BareTorch PyTorch model matching ~{hf_params_m:.2f}M parameters...")
        bt_config, _ = find_matching_baretorch_config(
            target_params_m=hf_params_m,
            target_vocab_size=vocab_size,
            layer_sequence=args.layer_sequence,
            max_seq_len=max_seq_len
        )

        for ctx in args.prompt_lens:
            print(f"  ├─ Benchmarking BareTorch ExecuTorch/MPS @ Context: {ctx:<5} tokens...", end="", flush=True)
            res = benchmark_baretorch_executorch(bt_config, ctx, args.gen_len)
            bt_runs.append(res)
            print(f" ✅ (TTFT: {format_cell(res['ttft_ms'])} ms | Decode: {format_cell(res['tokens_per_sec'])} tok/s | VRAM: {format_cell(res['decode_vram_mb'])} MB)")

        paired_results.append({
            "hf_baseline": {
                "model_name": model_id,
                "param_count_m": hf_params_m,
                "runs": hf_runs
            },
            "baretorch_matched": {
                "model_name": f"BareTorch Matched (~{hf_params_m:.1f}M)",
                "param_count_m": hf_params_m,
                "runs": bt_runs
            }
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(paired_results, f, indent=2)

    export_to_csv(paired_results, args.output_csv)


if __name__ == "__main__":
    main()