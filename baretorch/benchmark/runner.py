import os
import json
import csv
import argparse
import time
import torch
import torch.nn as nn
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM
from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM
from baretorch.benchmark.profiler import (
    LatencyProfiler,
    MemoryProfiler,
    RooflineEstimator,
    clear_gpu_memory
)

# Raise Dynamo recompile limit for variable context length sweeps
torch._dynamo.config.recompile_limit = 64


def count_baretorch_params_fast(
    d_model: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    rank: int = 8,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer"
) -> float:
    """
    Computes exact BareTorch model parameter count in pure Python integer arithmetic.
    Matched 1-to-1 with actual PyTorch module instantiations.
    """
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]
    full_layer_types = [raw_seq[i % len(raw_seq)] for i in range(num_layers)]

    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    head_dim = d_model // num_heads
    d_ff = int(d_model * 3.5)

    # 1. Embeddings (Token Embedding + Un-tied LM Head)
    embed_params = 2 * vocab_size * d_model

    # 2. Final Norm (RMSNorm)
    final_norm = d_model

    # 3. Layer Parameters
    layer_params = 0

    for l_type in full_layer_types:
        # RMSNorms (ln1 + ln2)
        norms = 2 * d_model

        # GatedMLP SwiGLU (w1 + w2 + w3)
        mlp = 3 * d_model * d_ff

        if l_type == "cs_lrad":
            # CS-LRAD Engine (W_q, W_k, W_v, W_swish_gate, W_out + W_u, W_r + W_gate, W_beta_gate)
            attn = (5 * (d_model ** 2)) + (2 * d_model * num_heads * rank) + (2 * (d_model * num_heads + num_heads))
        else:
            # Transformer Attention (W_q, W_k, W_v, W_out)
            attn = (2 * (d_model ** 2)) + (2 * d_model * (num_kv_heads * head_dim))

        layer_params += (norms + mlp + attn)

    total_params = embed_params + final_norm + layer_params
    return total_params / 1e6  # Millions


def find_matching_baretorch_config(
    target_params_m: float,
    target_vocab_size: int = 50257,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    max_seq_len: int = 32768
) -> tuple[BareTorchConfig, float]:
    """Evaluates candidate hyper-parameters via pure Python math using exact target vocab_size."""
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]

    best_cfg = None
    best_diff = float("inf")
    best_params_m = 0.0

    for nl in range(12, 36, 2):
        for d in range(512, 3072, 32):
            for nh in [8, 12, 16, 20, 24, 32]:
                if d % nh != 0:
                    continue
                head_dim = d // nh
                if head_dim < 64 or head_dim > 128:
                    continue
                if head_dim % 2 != 0:  # RoPE requires even head_dim for 2D complex rotations
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
    print(f"  ⚡ Vocab-Aware Match completed (|Δ| = {best_diff:.2f}M, {div_pct:.2f}%)")
    return best_cfg, best_params_m


def profile_single_model(
    model: nn.Module,
    model_name: str,
    param_count_m: float,
    prompt_lens: List[int],
    gen_len: int,
    device: str,
    is_compiled: bool = False
) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    cfg_dict = cfg.to_dict() if (cfg is not None and hasattr(cfg, "to_dict")) else {}
    vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", 50257))

    if is_compiled and device == "cuda":
        print(f"  ⚡ Running PyTorch Inductor / Triton Warmup for {model_name}...")
        try:
            warmup_prompt = torch.randint(0, vocab_size, (1, 128), device=device)
            with torch.no_grad():
                _ = model(warmup_prompt, use_cache=True)
            torch.cuda.synchronize()
        except Exception:
            clear_gpu_memory(device)

    results = {
        "model_name": model_name,
        "param_count_m": round(param_count_m, 2),
        "executorch_arena_mb": "N/A",
        "runs": []
    }

    for ctx_len in prompt_lens:
        print(f"  🔍 Sweeping {model_name} @ Context Length: {ctx_len} tokens...")

        lat = LatencyProfiler.profile_inference(
            model=model,
            prompt_len=ctx_len,
            gen_len=gen_len,
            device=device,
            vocab_size=vocab_size
        )

        if lat["status"] == "OOM":
            vram_mb = "OOM"
            cache_mb = "OOM"
            roofline_fps = {dev: "OOM" for dev in RooflineEstimator.DEVICE_BANDWIDTH_GBPS.keys()}
        else:
            vram_mb = MemoryProfiler.get_peak_vram_mb(device=device)
            cache_bytes = RooflineEstimator.calculate_active_cache_bytes(model, seq_len=ctx_len)
            cache_mb = round(cache_bytes / (1024.0 ** 2), 2)

            roofline_fps = RooflineEstimator.project_throughput(
                param_count_m=param_count_m,
                active_cache_bytes=cache_bytes,
                precision_bytes=2.0
            )

        results["runs"].append({
            "prompt_len": ctx_len,
            "ttft_ms": lat["ttft_ms"],
            "tokens_per_sec": lat["tokens_per_sec"],
            "peak_vram_mb": vram_mb,
            "cache_memory_mb": cache_mb,
            "roofline_fps": roofline_fps
        })

    et_profile = MemoryProfiler.profile_executorch_arena(
        model=model,
        seq_len=128,
        vocab_size=vocab_size
    )
    results["executorch_arena_mb"] = et_profile.get("arena_ram_mb", "N/A")

    return results


def format_cell(val: Any) -> str:
    if val == "OOM":
        return "💥 OOM"
    elif isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)


def print_comparison_pairs(paired_results: List[Dict[str, Any]]):
    print("\n" + "=" * 145)
    print("📊 APPLES-TO-APPLES BENCHMARK SUITE: BareTorch Matched Hybrids vs Open Source Baselines")
    print("=" * 145)

    for pair_idx, pair in enumerate(paired_results, 1):
        hf_res = pair["hf_baseline"]
        bt_res = pair["baretorch_matched"]

        print(f"\n🎯 PAIR {pair_idx}: {hf_res['model_name']}")
        print(f"  • Baseline Params: {hf_res['param_count_m']:.2f}M")
        print(f"  • BareTorch Matched Params: {bt_res['param_count_m']:.2f}M (Δ = {abs(bt_res['param_count_m'] - hf_res['param_count_m']):.2f}M)")
        print("-" * 145)

        prompt_lens = [r["prompt_len"] for r in bt_res["runs"]]

        for run_idx, ctx in enumerate(prompt_lens):
            b_run = bt_res["runs"][run_idx]
            h_run = hf_res["runs"][run_idx]

            print(f" Context: {ctx:<6} tokens | BareTorch Matched ({bt_res['param_count_m']:.1f}M) | Baseline ({hf_res['param_count_m']:.1f}M) | BareTorch Advantage")
            print("-" * 145)

            t_b, t_h = b_run["ttft_ms"], h_run["ttft_ms"]
            ttft_adv = f"{((t_h - t_b) / t_h) * 100:+.1f}% TTFT" if (isinstance(t_h, (int, float)) and isinstance(t_b, (int, float)) and t_h > 0) else "N/A"
            print(f"    Prefill Latency (ms) : {format_cell(t_b):<25} | {format_cell(t_h):<20} | {ttft_adv}")

            s_b, s_h = b_run["tokens_per_sec"], h_run["tokens_per_sec"]
            speed_adv = f"{s_b / s_h:.2f}x Speed" if (isinstance(s_h, (int, float)) and isinstance(s_b, (int, float)) and s_h > 0) else "N/A"
            print(f"    Local GPU (tok/s)    : {format_cell(s_b):<25} | {format_cell(s_h):<20} | {speed_adv}")

            c_b, c_h = b_run["cache_memory_mb"], h_run["cache_memory_mb"]
            cache_adv = f"-{((c_h - c_b) / c_h) * 100:.1f}% RAM" if (isinstance(c_h, (int, float)) and isinstance(c_b, (int, float)) and c_h > 0) else "N/A"
            print(f"    Active Cache (MB)    : {format_cell(c_b):<25} | {format_cell(c_h):<20} | {cache_adv}")

            p_b, p_h = b_run["roofline_fps"]["iphone_16_pro"], h_run["roofline_fps"]["iphone_16_pro"]
            phone_adv = f"{p_b / p_h:.2f}x FPS" if (isinstance(p_h, (int, float)) and isinstance(p_b, (int, float)) and p_h > 0) else "N/A"
            print(f"    iPhone 16 Pro (FPS)  : {format_cell(p_b):<25} | {format_cell(p_h):<20} | {phone_adv}")

            print("-" * 145)


def export_to_csv(paired_results: List[Dict[str, Any]], output_csv: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "Baseline_Model_ID", "Context_Length", "Metric", 
        "BareTorch_Matched_Value", "Baseline_Value", "BareTorch_Advantage"
    ]
    devices = list(RooflineEstimator.DEVICE_BANDWIDTH_GBPS.keys())

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for pair in paired_results:
            hf_res = pair["hf_baseline"]
            bt_res = pair["baretorch_matched"]
            baseline_id = hf_res["model_name"]

            prompt_lens = [r["prompt_len"] for r in bt_res["runs"]]

            for run_idx, ctx in enumerate(prompt_lens):
                b_run = bt_res["runs"][run_idx]
                h_run = hf_res["runs"][run_idx]

                t_b, t_h = b_run["ttft_ms"], h_run["ttft_ms"]
                ttft_adv = f"{((t_h - t_b) / t_h) * 100:+.2f}%" if (isinstance(t_h, (int, float)) and isinstance(t_b, (int, float)) and t_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Prefill_Latency_ms", t_b, t_h, ttft_adv])

                s_b, s_h = b_run["tokens_per_sec"], h_run["tokens_per_sec"]
                speed_adv = f"{s_b / s_h:.2f}x" if (isinstance(s_h, (int, float)) and isinstance(s_b, (int, float)) and s_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Local_GPU_Decode_tok_s", s_b, s_h, speed_adv])

                c_b, c_h = b_run["cache_memory_mb"], h_run["cache_memory_mb"]
                cache_adv = f"-{((c_h - c_b) / c_h) * 100:.2f}%" if (isinstance(c_h, (int, float)) and isinstance(c_b, (int, float)) and c_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Active_Cache_MB", c_b, c_h, cache_adv])

                for dev_key in devices:
                    p_b, p_h = b_run["roofline_fps"][dev_key], h_run["roofline_fps"][dev_key]
                    fps_adv = f"{p_b / p_h:.2f}x" if (isinstance(p_h, (int, float)) and isinstance(p_b, (int, float)) and p_h > 0) else "N/A"
                    writer.writerow([baseline_id, ctx, f"Projected_FPS_{dev_key}", p_b, p_h, fps_adv])

    print(f"📊 Apples-to-Apples Multi-model CSV report saved to: {output_csv}")


def run_comparative_benchmark(
    hf_model_ids: List[str] = ["meta-llama/Llama-3.2-1B"],
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    prompt_lens: List[int] = [2048, 4096, 8192, 16384, 32768],
    gen_len: int = 32,
    compile_model: bool = True,
    output_json: str = "./results_sota_suite.json",
    output_csv: str = "./results_sota_suite.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("\n======================================================================")
    print(f"🚀 BareTorch Apples-to-Apples Vocab-Aware Benchmark Suite [{device.upper()}]")
    print(f"  • Baseline Target Models ({len(hf_model_ids)}) : {', '.join(hf_model_ids)}")
    print("======================================================================\n")

    max_seq_len = max(prompt_lens) + gen_len + 1024
    paired_results = []

    for model_id in hf_model_ids:
        clear_gpu_memory(device)
        print(f"\n📦 Loading Hugging Face baseline model: '{model_id}'...")

        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True
            )
            actual_model_id = model_id
        except Exception as e:
            print(f"⚠️ Could not load '{model_id}' ({e}). Falling back to 'gpt2'...")
            actual_model_id = f"gpt2 (fallback for {model_id})"
            hf_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=dtype,
                device_map=device
            )

        hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6
        cfg = getattr(hf_model, "config", None)
        cfg_dict = cfg.to_dict() if (cfg is not None and hasattr(cfg, "to_dict")) else {}
        target_vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", 50257))

        print(f"  • Baseline Parameters (PyTorch Measured): {hf_params_m:.2f}M params (vocab_size={target_vocab_size})")

        hf_res = profile_single_model(
            model=hf_model,
            model_name=actual_model_id,
            param_count_m=hf_params_m,
            prompt_lens=prompt_lens,
            gen_len=gen_len,
            device=device,
            is_compiled=False
        )

        del hf_model
        clear_gpu_memory(device)

        print(f"\n⚙️ Looking up BareTorch blueprint matching ~{hf_params_m:.2f}M parameters...")
        bt_config, bt_params_m_predicted = find_matching_baretorch_config(
            target_params_m=hf_params_m,
            target_vocab_size=target_vocab_size,
            layer_sequence=layer_sequence,
            max_seq_len=max_seq_len
        )

        bt_model = BareTorchForCausalLM(bt_config).to(device=device, dtype=dtype)
        actual_bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

        print(f"  🎯 Target Params: {hf_params_m:.2f}M | Predicted Math: {bt_params_m_predicted:.2f}M | Actual BareTorch Instantiated: {actual_bt_params_m:.2f}M (Δ = {abs(actual_bt_params_m - hf_params_m):.2f}M)")
        print(f"     Config: d_model={bt_config.d_model}, num_layers={bt_config.num_layers}, num_heads={bt_config.num_heads}")

        bt_model_name = f"BareTorch Matched ({actual_bt_params_m:.1f}M)"

        if compile_model and device == "cuda":
            print(f"⚡ Fusing {bt_model_name} CS-LRAD kernels via torch.compile(dynamic=True)...")
            try:
                bt_model = torch.compile(bt_model, dynamic=True)
            except Exception as comp_err:
                print(f"⚠️ torch.compile failed ({comp_err}). Falling back to PyTorch eager execution...")

        bt_res = profile_single_model(
            model=bt_model,
            model_name=bt_model_name,
            param_count_m=actual_bt_params_m,
            prompt_lens=prompt_lens,
            gen_len=gen_len,
            device=device,
            is_compiled=compile_model
        )

        del bt_model
        clear_gpu_memory(device)

        paired_results.append({
            "hf_baseline": hf_res,
            "baretorch_matched": bt_res
        })

    print_comparison_pairs(paired_results)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(paired_results, f, indent=2)

    export_to_csv(paired_results, output_csv)
    print(f"💾 Apples-to-Apples JSON report saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="BareTorch Vocab-Aware Benchmark Suite")
    parser.add_argument(
        "--hf_model_ids",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.2-1B"],
        help="Space-separated list of Hugging Face model IDs to evaluate"
    )
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile kernel fusion")
    parser.add_argument("--output_json", type=str, default="./results_sota_suite.json")
    parser.add_argument("--output_csv", type=str, default="./results_sota_suite.csv")

    args = parser.parse_args()

    run_comparative_benchmark(
        hf_model_ids=args.hf_model_ids,
        layer_sequence=args.layer_sequence,
        prompt_lens=args.prompt_lens,
        gen_len=args.gen_len,
        compile_model=not args.no_compile,
        output_json=args.output_json,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()