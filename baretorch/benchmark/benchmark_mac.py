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


def benchmark_pte_inference_standardized(
    pte_prefill_path: str,
    pte_decode_path: str,
    prompt_len: int,
    gen_len: int,
    vocab_size: int
) -> dict:
    """
    Executes exported ExecuTorch (.pte) graphs via native C++ pybindings.
    Standardized 2-phase benchmark matching NVIDIA CUDA profiler:
      1. Prefill Phase: Runs prefill .pte (seq_len = prompt_len) -> TTFT (ms)
      2. Decode Phase: Runs 1-token decode .pte (seq_len = 1) gen_len times -> tok/s
    """
    if not HAS_EXECUTORCH_RUNTIME:
        return {"ttft_ms": "Runtime Missing", "tokens_per_sec": "N/A", "peak_vram_mb": "N/A"}

    if not (os.path.exists(pte_prefill_path) and os.path.exists(pte_decode_path)):
        return {"ttft_ms": "PTE Missing", "tokens_per_sec": "N/A", "peak_vram_mb": "N/A"}

    try:
        runtime = Runtime.get()

        prog_prefill = runtime.load_program(pte_prefill_path)
        method_prefill = prog_prefill.load_method("forward")

        prog_decode = runtime.load_program(pte_decode_path)
        method_decode = prog_decode.load_method("forward")

        prompt_input = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
        single_token_input = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)

        # 1. Warmup Passes
        _ = method_prefill.execute([prompt_input])
        for _ in range(3):
            _ = method_decode.execute([single_token_input])

        # 2. Prefill Phase (TTFT ms)
        ttft_start = time.perf_counter()
        _ = method_prefill.execute([prompt_input])
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        # 3. Autoregressive Decode Phase (gen_len tokens)
        gen_start = time.perf_counter()
        for _ in range(gen_len):
            _ = method_decode.execute([single_token_input])
        decode_sec = max(time.perf_counter() - gen_start, 1e-5)

        tokens_per_sec = gen_len / decode_sec
        peak_vram_mb = get_vram_mb()

        return {
            "prompt_len": prompt_len,
            "ttft_ms": round(ttft_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2)
        }
    except Exception as e:
        err_type = type(e).__name__
        print(f"\n    ⚠️ ExecuTorch runtime execution error ({e})")
        return {"ttft_ms": f"Exec Error ({err_type})", "tokens_per_sec": "N/A", "peak_vram_mb": "N/A"}


def main():
    parser = argparse.ArgumentParser(description="BareTorch Standardized Apple Silicon ExecuTorch (.pte) Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--quant_type", type=str, choices=["int4", "int8", "fp32"], default="int4")
    parser.add_argument("--backend", type=str, choices=["none", "coreml", "xnnpack"], default="none")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--gen_len", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./pte_models")
    args = parser.parse_args()

    print("==================================================================================================")
    print(f"🍎 BARETORCH STANDARDIZED EXECUTORCH (.PTE) BENCHMARK (Quant: {args.quant_type.upper()} | Backend: {args.backend.upper()})")
    print("==================================================================================================")

    sanitized_name = args.hf_model_id.replace("/", "_").replace("-", "_").lower()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
    except Exception:
        vocab_size = 50257

    clear_memory()
    print(f"\n📦 Loading Hugging Face Baseline: '{args.hf_model_id}'...")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, torch_dtype=torch.float32, trust_remote_code=True)
    hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6
    hf_vocab_size = getattr(hf_model.config, "vocab_size", vocab_size)

    print(f"\n⚙️ Finding BareTorch blueprint matching ~{hf_params_m:.2f}M parameters...")
    bt_config, _ = find_matching_baretorch_config(
        target_params_m=hf_params_m,
        target_vocab_size=hf_vocab_size,
        layer_sequence="cs_lrad,cs_lrad,cs_lrad,transformer",
        max_seq_len=16384
    )
    bt_model = BareTorchForCausalLM(bt_config).to(dtype=torch.float32)
    actual_bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

    print("\n📦 Exporting single-token decode .pte models (seq_len=1)...")
    hf_decode_path = os.path.join(args.output_dir, f"baseline_{sanitized_name}_dec1_{args.quant_type}_{args.backend}.pte")
    export_single_model_to_pte(
        model=hf_model,
        model_name=f"{args.hf_model_id} (Decode-1)",
        vocab_size=hf_vocab_size,
        seq_len=1,
        output_pte_path=hf_decode_path,
        quant_type=args.quant_type,
        backend_delegate=args.backend
    )

    bt_decode_path = os.path.join(args.output_dir, f"baretorch_{sanitized_name}_dec1_{args.quant_type}_{args.backend}.pte")
    export_single_model_to_pte(
        model=bt_model,
        model_name=f"BareTorch (Decode-1)",
        vocab_size=hf_vocab_size,
        seq_len=1,
        output_pte_path=bt_decode_path,
        quant_type=args.quant_type,
        backend_delegate=args.backend
    )

    hf_results = {}
    bt_results = {}

    for ctx in args.prompt_lens:
        print(f"\n📦 Exporting & Benchmarking .pte Graphs @ Context Length: {ctx} tokens...")

        hf_prefill_path = os.path.join(args.output_dir, f"baseline_{sanitized_name}_pref{ctx}_{args.quant_type}_{args.backend}.pte")
        export_single_model_to_pte(
            model=hf_model,
            model_name=f"{args.hf_model_id} (Prefill-{ctx})",
            vocab_size=hf_vocab_size,
            seq_len=ctx,
            output_pte_path=hf_prefill_path,
            quant_type=args.quant_type,
            backend_delegate=args.backend
        )

        bt_prefill_path = os.path.join(args.output_dir, f"baretorch_{sanitized_name}_pref{ctx}_{args.quant_type}_{args.backend}.pte")
        export_single_model_to_pte(
            model=bt_model,
            model_name=f"BareTorch (Prefill-{ctx})",
            vocab_size=hf_vocab_size,
            seq_len=ctx,
            output_pte_path=bt_prefill_path,
            quant_type=args.quant_type,
            backend_delegate=args.backend
        )

        print(f"  ├─ Benchmarking Baseline .pte @ {ctx} tokens...", end="", flush=True)
        hf_results[ctx] = benchmark_pte_inference_standardized(
            pte_prefill_path=hf_prefill_path,
            pte_decode_path=hf_decode_path,
            prompt_len=ctx,
            gen_len=args.gen_len,
            vocab_size=hf_vocab_size
        )
        print(f" ✅ (TTFT: {hf_results[ctx]['ttft_ms']} ms | Decode: {hf_results[ctx]['tokens_per_sec']} tok/s)")

        print(f"  ├─ Benchmarking BareTorch .pte @ {ctx} tokens...", end="", flush=True)
        bt_results[ctx] = benchmark_pte_inference_standardized(
            pte_prefill_path=bt_prefill_path,
            pte_decode_path=bt_decode_path,
            prompt_len=ctx,
            gen_len=args.gen_len,
            vocab_size=hf_vocab_size
        )
        print(f" ✅ (TTFT: {bt_results[ctx]['ttft_ms']} ms | Decode: {bt_results[ctx]['tokens_per_sec']} tok/s)")

    del hf_model, bt_model
    clear_memory()

    print("\n" + "=" * 135)
    print(f"📊 EXECUTORCH (.PTE) GRAPH REPORT: BARETORCH vs. LLAMA 3.2 1B (Quant: {args.quant_type.upper()})")
    print("=" * 135)
    print(f"{'Context Length':<15} | {'BareTorch TTFT':<18} | {'Llama TTFT':<18} | {'BareTorch Decode':<20} | {'Llama Decode':<20}")
    print("-" * 135)

    for ctx in args.prompt_lens:
        bt = bt_results.get(ctx, {})
        hf = hf_results.get(ctx, {})

        bt_ttft = f"{bt.get('ttft_ms', 'N/A')} ms" if isinstance(bt.get('ttft_ms'), (int, float)) else str(bt.get('ttft_ms'))
        hf_ttft = f"{hf.get('ttft_ms', 'N/A')} ms" if isinstance(hf.get('ttft_ms'), (int, float)) else str(hf.get('ttft_ms'))
        bt_dec = f"{bt.get('tokens_per_sec', 'N/A')} tok/s" if isinstance(bt.get('tokens_per_sec'), (int, float)) else str(bt.get('tokens_per_sec'))
        hf_dec = f"{hf.get('tokens_per_sec', 'N/A')} tok/s" if isinstance(hf.get('tokens_per_sec'), (int, float)) else str(hf.get('tokens_per_sec'))

        print(f"{ctx:<15} | {bt_ttft:<18} | {hf_ttft:<18} | {bt_dec:<20} | {hf_dec:<20}")

    print("=" * 135)


if __name__ == "__main__":
    main()