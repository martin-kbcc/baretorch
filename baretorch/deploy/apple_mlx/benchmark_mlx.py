# baretorch/deploy/apple_mlx/benchmark_mlx.py
import os
import csv
import json
import time
import gc
import argparse
import traceback
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.deploy.apple_mlx.modeling_mlx import BareTorchForCausalLMMLX

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.models.cache import make_prompt_cache
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


def clear_memory():
    """Flushes Python garbage collection and clears MLX cache/memory stats."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def get_peak_vram_mb() -> float:
    """Returns peak Metal GPU memory allocation in MB."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024.0 ** 2)
    return 0.0


def count_mlx_params_m(model: nn.Module) -> float:
    """Counts actual instantiated parameters in an unquantized MLX module tree."""
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


def apply_quantization(model: nn.Module, bits: int = 4, group_size: int = 64) -> nn.Module:
    """Quantizes an MLX Module tree in-place using MLX's native nn.quantize API."""
    if bits in [4, 8]:
        nn.quantize(model, group_size=group_size, bits=bits)
    return model


def count_baretorch_params_fast(
    d_model: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    rank: int = 8,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer"
) -> float:
    """Exact BareTorch parameter math matching instantiated MLX modules."""
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
    """
    Evaluates candidate hyper-parameters and matches target model size.
    Restricts head_dim strictly to [64, 128] to ensure MLX's Metal FlashAttention
    (mx.fast.sdpa) executes without falling back to eager O(L^2) matrices.
    """
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

                # MLX SDPA Kernel requirement: head_dim MUST be 64 or 128
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
    print(f"  ⚡ MLX Vocab-Aware Match completed (|Δ| = {best_diff:.2f}M, {div_pct:.2f}%)")
    return best_cfg, best_params_m


def benchmark_mlx_model(
    model: nn.Module,
    prompt_len: int,
    gen_len: int,
    vocab_size: int,
    is_mlx_lm: bool = False
) -> dict:
    clear_memory()
    prompt = mx.random.randint(0, vocab_size, (1, prompt_len))

    try:
        if is_mlx_lm:
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
            decode_vram_mb = get_peak_vram_mb()
        else:
            w_out, w_cache = model(prompt)
            mx.eval(w_out, w_cache)

            clear_memory()

            ttft_start = time.perf_counter()
            outputs, past_key_values = model(prompt)
            mx.eval(outputs, past_key_values)
            ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

            clear_memory()
            curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

            def decode_step_bt(tok, p_kv):
                logits, n_kv = model(tok, past_key_values=p_kv)
                next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
                return next_tok, n_kv

            compiled_step = mx.compile(decode_step_bt)

            dummy_tok, past_key_values = compiled_step(curr_token, past_key_values)
            mx.eval(dummy_tok, past_key_values)

            gen_start = time.perf_counter()
            for _ in range(gen_len):
                curr_token, past_key_values = compiled_step(curr_token, past_key_values)
                mx.eval(curr_token, past_key_values)

            decode_sec = max(time.perf_counter() - gen_start, 1e-5)
            decode_vram_mb = get_peak_vram_mb()

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

    print(f"\n📊 MLX Multi-model CSV report saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="BareTorch Apples-to-Apples MLX Suite")
    parser.add_argument(
        "--hf_model_ids",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.2-1B"],
        help="Space-separated list of Hugging Face/MLX model IDs to evaluate"
    )
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None, help="Quantize linear layers to 4-bit or 8-bit INT")
    parser.add_argument("--group_size", type=int, default=64, help="Quantization group size (64 or 128)")
    parser.add_argument("--output_json", type=str, default="./results_mlx_suite.json")
    parser.add_argument("--output_csv", type=str, default="./results_mlx_suite.csv")
    args = parser.parse_args()

    quant_suffix = f" (INT{args.quantize})" if args.quantize else " (FP16)"
    print("==================================================================================================")
    print(f"🍎 BARETORCH APPLES-TO-APPLES NATIVE MLX SUITE{quant_suffix} ({len(args.hf_model_ids)} Baseline Models)")
    print("==================================================================================================")

    paired_results = []
    max_seq_len = max(args.prompt_lens) + args.gen_len + 1024

    for model_id in args.hf_model_ids:
        print(f"\n" + "─" * 100)
        print(f"📦 Evaluating Target Baseline Family: '{model_id}'")
        print("─" * 100)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            vocab_size = getattr(tokenizer, "vocab_size", 50257)
        except Exception:
            vocab_size = 50257

        hf_runs = []
        bt_runs = []

        clear_memory()
        hf_params_m = 0.0

        if HAS_MLX_LM:
            try:
                print(f"  • Loading Baseline from MLX/HF Hub...")
                hf_model, _ = mlx_lm_load(model_id)

                # Step 1: Count baseline parameters BEFORE applying quantization
                hf_params_m = count_mlx_params_m(hf_model)
                print(f"  • Baseline Parameters (Full Precision): {hf_params_m:.2f}M params (vocab_size={vocab_size})")

                # Step 2: Apply quantization AFTER parameter counting
                if args.quantize:
                    print(f"  ⚡ Quantizing Baseline MLX model to INT{args.quantize} (group_size={args.group_size})...")
                    hf_model = apply_quantization(hf_model, bits=args.quantize, group_size=args.group_size)

                for ctx in args.prompt_lens:
                    print(f"  ├─ Benchmarking Baseline @ Context: {ctx:<5} tokens...", end="", flush=True)
                    res = benchmark_mlx_model(hf_model, ctx, args.gen_len, vocab_size, is_mlx_lm=True)
                    hf_runs.append(res)
                    print(f" ✅ (TTFT: {format_cell(res['ttft_ms'])} ms | Decode: {format_cell(res['tokens_per_sec'])} tok/s | VRAM: {format_cell(res['decode_vram_mb'])} MB)")

                del hf_model
                clear_memory()
            except Exception as e:
                print(f"⚠️ Could not benchmark baseline model '{model_id}' ({e}).")

        if hf_params_m == 0.0:
            hf_params_m = 1237.0

        print(f"\n  ⚙️ Looking up BareTorch MLX blueprint matching ~{hf_params_m:.2f}M parameters...")
        bt_config, bt_params_m_predicted = find_matching_baretorch_config(
            target_params_m=hf_params_m,
            target_vocab_size=vocab_size,
            layer_sequence=args.layer_sequence,
            max_seq_len=max_seq_len
        )

        bt_model = BareTorchForCausalLMMLX(bt_config)
        bt_model.set_dtype(mx.float16)

        # Step 3: Count BareTorch parameters BEFORE applying quantization
        actual_bt_params_m = count_mlx_params_m(bt_model)

        # Step 4: Apply quantization AFTER parameter counting
        if args.quantize:
            print(f"  ⚡ Quantizing BareTorch MLX model to INT{args.quantize} (group_size={args.group_size})...")
            bt_model = apply_quantization(bt_model, bits=args.quantize, group_size=args.group_size)

        print(f"  🎯 Target Params: {hf_params_m:.2f}M | Predicted Math: {bt_params_m_predicted:.2f}M | Actual MLX Instantiated: {actual_bt_params_m:.2f}M (Δ = {abs(actual_bt_params_m - hf_params_m):.2f}M)")
        print(f"     Config: d_model={bt_config.d_model}, num_layers={bt_config.num_layers}, num_heads={bt_config.num_heads}, head_dim={bt_config.d_model // bt_config.num_heads}")

        for ctx in args.prompt_lens:
            print(f"  ├─ Benchmarking BareTorch @ Context: {ctx:<5} tokens...", end="", flush=True)
            res = benchmark_mlx_model(bt_model, ctx, args.gen_len, vocab_size, is_mlx_lm=False)
            bt_runs.append(res)
            print(f" ✅ (TTFT: {format_cell(res['ttft_ms'])} ms | Decode: {format_cell(res['tokens_per_sec'])} tok/s | VRAM: {format_cell(res['decode_vram_mb'])} MB)")

        del bt_model
        clear_memory()

        paired_results.append({
            "hf_baseline": {
                "model_name": model_id,
                "param_count_m": hf_params_m,
                "runs": hf_runs
            },
            "baretorch_matched": {
                "model_name": f"BareTorch Matched ({actual_bt_params_m:.1f}M)",
                "param_count_m": actual_bt_params_m,
                "runs": bt_runs
            }
        })

    # Summary Report
    print("\n" + "=" * 145)
    print(f"📊 APPLES-TO-APPLES MLX SUITE SUMMARY REPORT{quant_suffix}")
    print("=" * 145)

    for pair in paired_results:
        hf_res = pair["hf_baseline"]
        bt_res = pair["baretorch_matched"]

        print(f"\n🎯 BASELINE: {hf_res['model_name']} ({hf_res['param_count_m']:.1f}M) vs BARETORCH MATCHED ({bt_res['param_count_m']:.1f}M)")
        print(f"{'Context Length':<15} | {'BareTorch TTFT':<16} | {'Baseline TTFT':<16} | {'BareTorch Decode':<18} | {'Baseline Decode':<18} | {'Decode VRAM (BT / HF)':<22}")
        print("-" * 145)

        bt_runs = bt_res.get("runs", [])
        hf_runs = hf_res.get("runs", [])

        for run_idx in range(len(bt_runs)):
            b_run = bt_runs[run_idx]
            h_run = hf_runs[run_idx] if run_idx < len(hf_runs) else {}

            ctx = b_run["prompt_len"]
            bt_ttft = f"{format_cell(b_run.get('ttft_ms'))} ms"
            hf_ttft = f"{format_cell(h_run.get('ttft_ms'))} ms"
            bt_dec = f"{format_cell(b_run.get('tokens_per_sec'))} tok/s"
            hf_dec = f"{format_cell(h_run.get('tokens_per_sec'))} tok/s"
            vram_str = f"{format_cell(b_run.get('decode_vram_mb'))} / {format_cell(h_run.get('decode_vram_mb'))} MB"

            print(f"{ctx:<15} | {bt_ttft:<16} | {hf_ttft:<16} | {bt_dec:<18} | {hf_dec:<18} | {vram_str:<22}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(paired_results, f, indent=2)

    export_to_csv(paired_results, args.output_csv)
    print(f"💾 MLX Multi-model JSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()