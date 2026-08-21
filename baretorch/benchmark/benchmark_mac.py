# baretorch/benchmark/benchmark_mac.py
import os
import time
import gc
import argparse
import traceback
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM
from baretorch.export.export_executorch import export_single_model_to_pte, find_matching_baretorch_config

try:
    from executorch.runtime import Runtime
    HAS_EXECUTORCH_RUNTIME = True
except ImportError:
    HAS_EXECUTORCH_RUNTIME = False


def clear_memory():
    """Flushes Python garbage collection and MPS VRAM allocation."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_vram_mb() -> float:
    """Returns actual Apple Silicon GPU VRAM allocated on MPS in MB."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / (1024.0 ** 2)
        except AttributeError:
            try:
                return torch.mps.current_allocated_memory() / (1024.0 ** 2)
            except AttributeError:
                pass
    process = psutil.Process()
    return process.memory_info().rss / (1024.0 ** 2)


def benchmark_pte_execution(
    pte_path: str,
    ctx_len: int,
    vocab_size: int,
    runs: int = 5
) -> dict:
    """
    Executes an exported ExecuTorch .pte file through native C++ pybindings.
    Passes matching (1, ctx_len) input tensor matching static memory arena layout.
    """
    if not os.path.exists(pte_path):
        return {"latency_ms": "File Missing", "peak_vram_mb": "N/A"}

    if not HAS_EXECUTORCH_RUNTIME:
        return {"latency_ms": "Runtime Missing", "peak_vram_mb": "N/A"}

    try:
        runtime = Runtime.get()
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")

        input_tensor = torch.randint(0, vocab_size, (1, ctx_len), dtype=torch.long)

        # 1. Warmup Pass
        _ = method.execute([input_tensor])

        # 2. Timed Forward Execution Runs
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = method.execute([input_tensor])
        elapsed_sec = (time.perf_counter() - start_time) / runs

        peak_vram_mb = get_vram_mb()

        return {
            "ctx_len": ctx_len,
            "latency_ms": round(elapsed_sec * 1000.0, 2),
            "peak_vram_mb": round(peak_vram_mb, 2)
        }
    except Exception as e:
        err_type = type(e).__name__
        print(f"\n    ⚠️ ExecuTorch runtime execution error ({e})")
        return {"latency_ms": f"Exec Error ({err_type})", "peak_vram_mb": "N/A"}


def main():
    parser = argparse.ArgumentParser(description="BareTorch Standardized Apple Silicon ExecuTorch (.pte) Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--quant_type", type=str, choices=["int4", "int8", "fp32"], default="int4")
    parser.add_argument("--backend", type=str, choices=["none", "coreml", "xnnpack"], default="none")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--output_dir", type=str, default="./pte_models")
    args = parser.parse_args()

    print("==================================================================================================")
    print(f"🍎 BARETORCH EXECUTORCH (.PTE) BENCHMARK (Quant: {args.quant_type.upper()} | Backend: {args.backend.upper()})")
    print("==================================================================================================")

    sanitized_name = args.hf_model_id.replace("/", "_").replace("-", "_").lower()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
    except Exception:
        vocab_size = 50257

    # Load models once for metadata extraction
    clear_memory()
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, torch_dtype=torch.float32, trust_remote_code=True)
    hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6
    hf_vocab_size = getattr(hf_model.config, "vocab_size", vocab_size)

    bt_config, _ = find_matching_baretorch_config(
        target_params_m=hf_params_m,
        target_vocab_size=hf_vocab_size,
        layer_sequence="cs_lrad,cs_lrad,cs_lrad,transformer",
        max_seq_len=16384
    )
    bt_model = BareTorchForCausalLM(bt_config).to(dtype=torch.float32)
    actual_bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

    hf_results = {}
    bt_results = {}

    for ctx in args.prompt_lens:
        print(f"\n📦 Exporting & Lowering .pte Graphs @ Context Length: {ctx} tokens...")

        # 1. Export Baseline for ctx
        hf_pte_path = os.path.join(args.output_dir, f"baseline_{sanitized_name}_{ctx}_{args.quant_type}_{args.backend}.pte")
        hf_export_info = export_single_model_to_pte(
            model=hf_model,
            model_name=args.hf_model_id,
            vocab_size=hf_vocab_size,
            seq_len=ctx,
            output_pte_path=hf_pte_path,
            quant_type=args.quant_type,
            backend_delegate=args.backend
        )

        # 2. Export BareTorch for ctx
        bt_pte_path = os.path.join(args.output_dir, f"baretorch_{sanitized_name}_{ctx}_{args.quant_type}_{args.backend}.pte")
        bt_export_info = export_single_model_to_pte(
            model=bt_model,
            model_name=f"BareTorch Matched ({actual_bt_params_m:.1f}M)",
            vocab_size=hf_vocab_size,
            seq_len=ctx,
            output_pte_path=bt_pte_path,
            quant_type=args.quant_type,
            backend_delegate=args.backend
        )

        # 3. Benchmark .pte execution
        print(f"  ├─ Benchmarking Baseline .pte @ {ctx} tokens...", end="", flush=True)
        hf_results[ctx] = benchmark_pte_execution(hf_pte_path, ctx, hf_vocab_size)
        print(f" ✅ (Latency: {hf_results[ctx]['latency_ms']} ms | PTE Size: {hf_export_info.get('pte_file_size_mb', 'N/A')} MB)")

        print(f"  ├─ Benchmarking BareTorch .pte @ {ctx} tokens...", end="", flush=True)
        bt_results[ctx] = benchmark_pte_execution(bt_pte_path, ctx, hf_vocab_size)
        print(f" ✅ (Latency: {bt_results[ctx]['latency_ms']} ms | PTE Size: {bt_export_info.get('pte_file_size_mb', 'N/A')} MB)")

    del hf_model, bt_model
    clear_memory()

    # 4. Comparative Report
    print("\n" + "=" * 110)
    print(f"📊 EXECUTORCH (.PTE) GRAPH LATENCY REPORT: BARETORCH vs. LLAMA 3.2 1B (Quant: {args.quant_type.upper()})")
    print("=" * 110)
    print(f"{'Context Length':<15} | {'BareTorch Latency':<22} | {'Llama 3.2 Latency':<22} | {'BareTorch Advantage':<20}")
    print("-" * 110)

    for ctx in args.prompt_lens:
        bt = bt_results.get(ctx, {})
        hf = hf_results.get(ctx, {})

        t_b = bt.get("latency_ms", "N/A")
        t_h = hf.get("latency_ms", "N/A")

        if isinstance(t_b, (int, float)) and isinstance(t_h, (int, float)) and t_h > 0:
            diff_pct = ((t_h - t_b) / t_h) * 100
            diff_str = f"{diff_pct:+.1f}% Latency"
        else:
            diff_str = "N/A"

        b_str = f"{t_b} ms" if isinstance(t_b, (int, float)) else str(t_b)
        h_str = f"{t_h} ms" if isinstance(t_h, (int, float)) else str(t_h)

        print(f"{ctx:<15} | {b_str:<22} | {h_str:<22} | {diff_str:<20}")

    print("=" * 110)


if __name__ == "__main__":
    main()