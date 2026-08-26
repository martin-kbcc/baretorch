# baretorch/benchmarks/nvidia/benchmark_torch.py
import os
import gc
import csv
import json
import time
import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

# Raise Dynamo recompile limit for variable context length sweeps
torch._dynamo.config.recompile_limit = 64


def clear_gpu_memory(device: str = "cuda"):
    """Flushes Python garbage collection and CUDA cache, resetting memory stats."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def count_baretorch_params_fast(
    d_model: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    rank: int = 8,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer"
) -> float:
    """Computes exact BareTorch parameter count matching PyTorch module instantiations."""
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
    """Evaluates candidate hyper-parameters via pure Python math matching target vocab_size."""
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
                if head_dim % 2 != 0:
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


def profile_inference(
    model: nn.Module,
    prompt_len: int = 2048,
    gen_len: int = 32,
    device: str = "cuda",
    vocab_size: int = 50257
) -> Dict[str, Any]:
    """Profiles TTFT, tok/s, and Steady-State Decode VRAM symmetrically with Apple MLX."""
    model.eval()
    
    raw_model = getattr(model, "_orig_mod", model)
    cfg = getattr(raw_model, "config", None)
    if cfg is not None:
        cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", vocab_size))

    try:
        prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

        # 1. Timed Prefill Phase (TTFT)
        start_prefill = time.perf_counter()
        with torch.no_grad():
            try:
                outputs = model(prompt, use_cache=True, num_logits_to_keep=1)
            except TypeError:
                outputs = model(prompt, use_cache=True)

        if device == "cuda":
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - start_prefill) * 1000.0

        past_key_values = getattr(outputs, "past_key_values", None)
        if hasattr(outputs, "logits"):
            curr_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        else:
            curr_token = outputs[0][:, -1:, :].argmax(dim=-1)

        # Free prefill output buffers before decode VRAM tracking
        del outputs, prompt
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 2. Timed Decode Phase (Autoregressive Token Generation)
        start_decode = time.perf_counter()
        with torch.no_grad():
            for _ in range(gen_len):
                outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = getattr(outputs, "past_key_values", None)
                if hasattr(outputs, "logits"):
                    curr_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                else:
                    curr_token = outputs[0][:, -1:, :].argmax(dim=-1)

        if device == "cuda":
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start_decode
        tokens_per_sec = gen_len / decode_time if decode_time > 0 else 0.0

        # Capture Steady-State Decode Peak VRAM in MB
        decode_vram_mb = 0.0
        if device == "cuda" and torch.cuda.is_available():
            decode_vram_mb = round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 2)

        return {
            "prompt_len": prompt_len,
            "ttft_ms": round(ttft_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "decode_vram_mb": decode_vram_mb,
            "status": "success"
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e).lower()
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_str:
            clear_gpu_memory(device)
            return {
                "prompt_len": prompt_len,
                "ttft_ms": "OOM",
                "tokens_per_sec": "OOM",
                "decode_vram_mb": "OOM",
                "status": "OOM"
            }
        raise e


def profile_single_model(
    model: nn.Module,
    model_name: str,
    param_count_m: float,
    prompt_lens: List[int],
    gen_len: int,
    device: str,
    is_compiled: bool = False
) -> Dict[str, Any]:
    raw_model = getattr(model, "_orig_mod", model)
    cfg = getattr(raw_model, "config", None)
    cfg_dict = cfg.to_dict() if (cfg is not None and hasattr(cfg, "to_dict")) else {}
    vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", 50257))

    # FULL JIT WARMUP: Execute complete Prefill + 32 Decode steps so Triton codegen finishes BEFORE timer
    if is_compiled and device == "cuda":
        warmup_ctx = prompt_lens[0]
        print(f"  ⚡ Executing full Inductor JIT warmup for {model_name} (Prefill {warmup_ctx} + {gen_len} Decode steps)...")
        try:
            _ = profile_inference(
                model=model,
                prompt_len=warmup_ctx,
                gen_len=gen_len,
                device=device,
                vocab_size=vocab_size
            )
        except Exception as w_err:
            print(f"  ⚠️ Warmup pass warning ({w_err})")
        clear_gpu_memory(device)

    runs = []
    for ctx_len in prompt_lens:
        print(f"  ├─ Benchmarking {model_name} @ Context: {ctx_len:<5} tokens...", end="", flush=True)

        clear_gpu_memory(device)

        lat = profile_inference(
            model=model,
            prompt_len=ctx_len,
            gen_len=gen_len,
            device=device,
            vocab_size=vocab_size
        )
        runs.append(lat)

        ttft_str = f"{format_cell(lat['ttft_ms'])} ms" if lat['ttft_ms'] != "OOM" else "💥 OOM"
        dec_str = f"{format_cell(lat['tokens_per_sec'])} tok/s" if lat['tokens_per_sec'] != "OOM" else "💥 OOM"
        vram_str = f"{format_cell(lat['decode_vram_mb'])} MB" if lat['decode_vram_mb'] != "OOM" else "💥 OOM"
        print(f" ✅ (TTFT: {ttft_str} | Decode: {dec_str} | VRAM: {vram_str})")

    return {
        "model_name": model_name,
        "param_count_m": round(param_count_m, 2),
        "runs": runs
    }


def format_cell(val: Any) -> str:
    if val == "OOM":
        return "💥 OOM"
    elif isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)


def export_to_csv(paired_results: List[Dict[str, Any]], output_csv: str):
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

                t_b, t_h = b_run.get("ttft_ms"), h_run.get("ttft_ms", "N/A")
                ttft_adv = f"{((t_h - t_b) / t_h) * 100:+.2f}%" if (isinstance(t_h, (int, float)) and isinstance(t_b, (int, float)) and t_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Prefill_Latency_ms", format_cell(t_b), format_cell(t_h), ttft_adv])

                s_b, s_h = b_run.get("tokens_per_sec"), h_run.get("tokens_per_sec", "N/A")
                speed_adv = f"{s_b / s_h:.2f}x" if (isinstance(s_h, (int, float)) and isinstance(s_b, (int, float)) and s_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Local_GPU_Decode_tok_s", format_cell(s_b), format_cell(s_h), speed_adv])

                v_b, v_h = b_run.get("decode_vram_mb"), h_run.get("decode_vram_mb", "N/A")
                vram_adv = f"-{((v_h - v_b) / v_h) * 100:.2f}%" if (isinstance(v_h, (int, float)) and isinstance(v_b, (int, float)) and v_h > 0) else "N/A"
                writer.writerow([baseline_id, ctx, "Decode_VRAM_MB", format_cell(v_b), format_cell(v_h), vram_adv])

    print(f"\n📊 NVIDIA PyTorch Multi-model CSV report saved to: {output_csv}")


def print_summary_report(paired_results: List[Dict[str, Any]]):
    print("\n" + "=" * 145)
    print("📊 APPLES-TO-APPLES NVIDIA PYTORCH SUITE SUMMARY REPORT")
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


def run_comparative_benchmark(
    hf_model_ids: List[str] = ["meta-llama/Llama-3.2-1B"],
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    prompt_lens: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
    gen_len: int = 32,
    compile_model: bool = True,
    output_json: str = "./results_nvidia_suite.json",
    output_csv: str = "./results_nvidia_suite.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("======================================================================")
    print(f"🚀 BARETORCH APPLES-TO-APPLES NVIDIA PYTORCH SUITE [{device.upper()}]")
    print(f"  • Baseline Target Models ({len(hf_model_ids)}) : {', '.join(hf_model_ids)}")
    print(f"  • torch.compile Mode: {'ENABLED (Symmetric)' if compile_model else 'DISABLED'}")
    print("======================================================================\n")

    max_seq_len = max(prompt_lens) + gen_len + 1024
    paired_results = []

    for model_id in hf_model_ids:
        print("─" * 100)
        print(f"📦 Evaluating Target Baseline Family: '{model_id}'")
        print("─" * 100)

        clear_gpu_memory(device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            vocab_size = getattr(tokenizer, "vocab_size", 50257)
        except Exception:
            vocab_size = 50257

        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            actual_model_id = model_id
        except Exception as e:
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True
                )
                actual_model_id = model_id
            except Exception as e_fallback:
                print(f"⚠️ Could not load '{model_id}' ({e_fallback}). Falling back to 'gpt2'...")
                actual_model_id = f"gpt2 (fallback for {model_id})"
                hf_model = AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                    torch_dtype=dtype,
                    device_map=device
                )

        hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6
        print(f"  • Baseline Parameters (PyTorch Measured): {hf_params_m:.2f}M params (vocab_size={vocab_size})")

        if compile_model and device == "cuda":
            print(f"  ⚡ Fusing Hugging Face baseline ({actual_model_id}) kernels via torch.compile(dynamic=True)...")
            try:
                hf_model = torch.compile(hf_model, dynamic=True)
            except Exception as comp_err:
                print(f"⚠️ torch.compile failed for baseline ({comp_err}). Falling back to Eager execution...")

        hf_res = profile_single_model(
            model=hf_model,
            model_name=actual_model_id,
            param_count_m=hf_params_m,
            prompt_lens=prompt_lens,
            gen_len=gen_len,
            device=device,
            is_compiled=compile_model
        )

        del hf_model
        clear_gpu_memory(device)

        print(f"\n  ⚙️ Looking up BareTorch blueprint matching ~{hf_params_m:.2f}M parameters...")
        bt_config, bt_params_m_predicted = find_matching_baretorch_config(
            target_params_m=hf_params_m,
            target_vocab_size=vocab_size,
            layer_sequence=layer_sequence,
            max_seq_len=max_seq_len
        )

        bt_model = BareTorchForCausalLM(bt_config).to(device=device, dtype=dtype)
        actual_bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

        print(f"  🎯 Target Params: {hf_params_m:.2f}M | Predicted Math: {bt_params_m_predicted:.2f}M | Actual BareTorch Instantiated: {actual_bt_params_m:.2f}M (Δ = {abs(actual_bt_params_m - hf_params_m):.2f}M)")
        print(f"     Config: d_model={bt_config.d_model}, num_layers={bt_config.num_layers}, num_heads={bt_config.num_heads}")

        bt_model_name = f"BareTorch Matched ({actual_bt_params_m:.1f}M)"

        if compile_model and device == "cuda":
            print(f"  ⚡ Fusing {bt_model_name} CS-LRAD kernels via torch.compile(dynamic=True)...")
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

    print_summary_report(paired_results)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(paired_results, f, indent=2)

    export_to_csv(paired_results, output_csv)
    print(f"💾 Apples-to-Apples JSON report saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="BareTorch NVIDIA PyTorch Benchmark Suite")
    parser.add_argument(
        "--hf_model_ids",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.2-1B"],
        help="Space-separated list of Hugging Face model IDs to evaluate"
    )
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile kernel fusion")
    parser.add_argument("--output_json", type=str, default="./results_nvidia_suite.json")
    parser.add_argument("--output_csv", type=str, default="./results_nvidia_suite.csv")

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