# baretorch/deploy/apple_mlx/benchmark_mlx.py
import time
import gc
import argparse
import traceback
import psutil
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.deploy.apple_mlx.modeling_mlx import BareTorchForCausalLMMLX

try:
    from mlx_lm import load as mlx_lm_load
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


def clear_memory():
    """Triggers Python garbage collection and clears MLX cache allocations."""
    gc.collect()
    mx.eval()


def get_ram_mb() -> float:
    """Returns total active system memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024.0 ** 2)


def benchmark_mlx_inference_standardized(
    model: nn.Module,
    prompt_len: int,
    gen_len: int,
    vocab_size: int
) -> dict:
    """
    Standardized MLX benchmark matching NVIDIA CUDA profiler:
    1. Un-timed warmup pass
    2. Prefill phase (TTFT ms for prompt_len)
    3. Decode phase (steady-state tok/s over gen_len=32)
    """
    prompt = mx.random.randint(0, vocab_size, (1, prompt_len))
    curr_token = mx.random.randint(0, vocab_size, (1, 1))

    # 1. Warmup Pass
    w_out, w_cache = model(prompt)
    mx.eval(w_out)
    for _ in range(3):
        w_out, w_cache = model(curr_token, past_key_values=w_cache)
        mx.eval(w_out)

    # 2. Prefill Phase (TTFT ms)
    ttft_start = time.perf_counter()
    outputs, past_key_values = model(prompt)
    mx.eval(outputs)
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

    # 3. Decode Phase (gen_len tokens)
    gen_start = time.perf_counter()
    for _ in range(gen_len):
        outputs, past_key_values = model(curr_token, past_key_values=past_key_values)
        mx.eval(outputs)
        curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

    decode_sec = max(time.perf_counter() - gen_start, 1e-5)
    peak_ram_mb = get_ram_mb()
    tokens_per_sec = gen_len / decode_sec

    return {
        "prompt_len": prompt_len,
        "ttft_ms": round(ttft_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_ram_mb": round(peak_ram_mb, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="BareTorch Native Apple MLX Standardized Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--gen_len", type=int, default=32)
    args = parser.parse_args()

    print("==================================================================================================")
    print("🍎 BARETORCH NATIVE APPLE SILICON MLX CONTEXT SCALING BENCHMARK")
    print("==================================================================================================")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
    except Exception:
        vocab_size = 50257

    hf_results = {}
    bt_results = {}

    # 1. Benchmark Native MLX Baseline (Llama 3.2 1B)
    clear_memory()
    print(f"\n📦 Loading Native MLX Baseline: '{args.hf_model_id}'...")
    if HAS_MLX_LM:
        try:
            hf_model, _ = mlx_lm_load(args.hf_model_id)
            hf_vocab_size = vocab_size

            for ctx in args.prompt_lens:
                print(f"  ├─ Benchmarking Baseline (MLX) @ Context: {ctx:<5} tokens...", end="", flush=True)
                try:
                    hf_results[ctx] = benchmark_mlx_inference_standardized(hf_model, ctx, args.gen_len, hf_vocab_size)
                    print(f" ✅ (TTFT: {hf_results[ctx]['ttft_ms']} ms | Decode: {hf_results[ctx]['tokens_per_sec']} tok/s | RAM: {hf_results[ctx]['peak_ram_mb']} MB)")
                except Exception as e_step:
                    print(f" ❌ Failed ({type(e_step).__name__})")
                    traceback.print_exc()
                    hf_results[ctx] = {"ttft_ms": "OOM/Fail", "tokens_per_sec": "OOM/Fail", "peak_ram_mb": "N/A"}

            del hf_model
            clear_memory()
        except Exception as e:
            print(f"⚠️ Failed to load baseline model via mlx-lm ({e}). Skipping HF baseline run.")
    else:
        print("⚠️ 'mlx-lm' package not found (`pip install mlx-lm`). Skipping HF baseline run.")

    # 2. Benchmark Native MLX BareTorch Hybrid Model
    clear_memory()
    print(f"\n⚙️ Instantiating Native BareTorch MLX Blueprint (~1237M params)...")
    bt_config = BareTorchConfig(
        vocab_size=vocab_size,
        d_model=1888,
        num_heads=16,
        num_layers=14,
        chunk_size=32,
        rank=8,
        max_seq_len=16384,
        layer_types=["cs_lrad", "cs_lrad", "cs_lrad", "transformer"] * 3 + ["cs_lrad", "cs_lrad"]
    )
    bt_model = BareTorchForCausalLMMLX(bt_config)

    for ctx in args.prompt_lens:
        print(f"  ├─ Benchmarking BareTorch (MLX) @ Context: {ctx:<5} tokens...", end="", flush=True)
        try:
            bt_results[ctx] = benchmark_mlx_inference_standardized(bt_model, ctx, args.gen_len, vocab_size)
            print(f" ✅ (TTFT: {bt_results[ctx]['ttft_ms']} ms | Decode: {bt_results[ctx]['tokens_per_sec']} tok/s | RAM: {bt_results[ctx]['peak_ram_mb']} MB)")
        except Exception as e_step:
            print(f" ❌ Failed ({type(e_step).__name__})")
            traceback.print_exc()
            bt_results[ctx] = {"ttft_ms": "Fail", "tokens_per_sec": "Fail", "peak_ram_mb": "N/A"}

    del bt_model
    clear_memory()

    # 3. Comparative Summary Report
    print("\n" + "=" * 130)
    print("📊 STANDARDIZED BARETORCH vs. LLAMA 3.2 1B CONTEXT REPORT (NATIVE APPLE SILICON MLX)")
    print("=" * 130)
    print(f"{'Context Length':<15} | {'BareTorch TTFT':<16} | {'Llama TTFT':<16} | {'BareTorch Decode':<18} | {'Llama Decode':<18} | {'RAM (BT / HF)':<20}")
    print("-" * 130)

    for ctx in args.prompt_lens:
        bt = bt_results.get(ctx, {})
        hf = hf_results.get(ctx, {})

        bt_ttft = f"{bt.get('ttft_ms', 'N/A')} ms"
        hf_ttft = f"{hf.get('ttft_ms', 'N/A')} ms"
        bt_dec = f"{bt.get('tokens_per_sec', 'N/A')} tok/s"
        hf_dec = f"{hf.get('tokens_per_sec', 'N/A')} tok/s"
        ram_str = f"{bt.get('peak_ram_mb', 'N/A')} / {hf.get('peak_ram_mb', 'N/A')} MB"

        print(f"{ctx:<15} | {bt_ttft:<16} | {hf_ttft:<16} | {bt_dec:<18} | {hf_dec:<18} | {ram_str:<20}")

    print("=" * 130)


if __name__ == "__main__":
    main()