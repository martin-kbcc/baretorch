# baretorch/deploy/apple_mlx/benchmark_mlx.py
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
    """Flushes Python garbage collection and resets MLX Metal peak memory tracking."""
    gc.collect()
    
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def get_peak_vram_mb() -> float:
    """Returns actual Metal GPU peak allocated memory in MB."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024.0 ** 2)
    elif hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024.0 ** 2)
    return 0.0


def benchmark_mlx_model(
    model: nn.Module,
    prompt_len: int,
    gen_len: int,
    vocab_size: int,
    is_mlx_lm: bool = False
) -> dict:
    """
    Standardized FP16 MLX benchmark matching CUDA profiler:
    1. Un-timed warmup pass
    2. Prefill phase (TTFT ms for prompt_len)
    3. Decode phase (steady-state tok/s with pre-compiled kernel warmup)
    """
    clear_memory()

    prompt = mx.random.randint(0, vocab_size, (1, prompt_len))
    curr_token = mx.random.randint(0, vocab_size, (1, 1))

    if is_mlx_lm:
        # 1. Warmup Pass (mlx_lm)
        w_cache = make_prompt_cache(model)
        w_out = model(prompt, cache=w_cache)
        mx.eval(w_out)
        w_curr = mx.argmax(w_out[:, -1:, :], axis=-1)
        for _ in range(3):
            w_out = model(w_curr, cache=w_cache)
            mx.eval(w_out)

        clear_memory()

        # 2. Prefill Phase (TTFT ms)
        cache = make_prompt_cache(model)
        ttft_start = time.perf_counter()
        outputs = model(prompt, cache=cache)
        mx.eval(outputs)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

        # 3. Decode Phase
        gen_start = time.perf_counter()
        for _ in range(gen_len):
            outputs = model(curr_token, cache=cache)
            mx.eval(outputs)
            curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

        decode_sec = max(time.perf_counter() - gen_start, 1e-5)
    else:
        # 1. Warmup Pass (BareTorch MLX)
        w_out, w_cache = model(prompt)
        mx.eval(w_out)
        w_curr = mx.argmax(w_out[:, -1:, :], axis=-1)
        for _ in range(3):
            w_out, w_cache = model(w_curr, past_key_values=w_cache)
            mx.eval(w_out)

        clear_memory()

        # 2. Prefill Phase (TTFT ms)
        ttft_start = time.perf_counter()
        outputs, past_key_values = model(prompt)
        mx.eval(outputs)
        ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

        curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)

        # 3. Compiled Decode Phase with JIT Kernel Warmup
        def decode_step_bt(tok, p_kv):
            logits, n_kv = model(tok, past_key_values=p_kv)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
            return next_tok, n_kv

        compiled_step = mx.compile(decode_step_bt)

        # JIT Warmup pass to eliminate Metal compilation overhead from timer
        dummy_tok, past_key_values = compiled_step(curr_token, past_key_values)
        mx.eval(dummy_tok)

        gen_start = time.perf_counter()
        for _ in range(gen_len):
            curr_token, past_key_values = compiled_step(curr_token, past_key_values)
            mx.eval(curr_token)

        decode_sec = max(time.perf_counter() - gen_start, 1e-5)

    peak_vram_mb = get_peak_vram_mb()
    tokens_per_sec = gen_len / decode_sec

    return {
        "prompt_len": prompt_len,
        "ttft_ms": round(ttft_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="BareTorch Native Apple MLX Standardized Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--gen_len", type=int, default=32)
    args = parser.parse_args()

    print("==================================================================================================")
    print("🍎 BARETORCH NATIVE APPLE SILICON MLX CONTEXT SCALING BENCHMARK (FP16)")
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
                print(f"  ├─ Benchmarking Baseline (MLX FP16) @ Context: {ctx:<5} tokens...", end="", flush=True)
                try:
                    hf_results[ctx] = benchmark_mlx_model(hf_model, ctx, args.gen_len, hf_vocab_size, is_mlx_lm=True)
                    print(f" ✅ (TTFT: {hf_results[ctx]['ttft_ms']} ms | Decode: {hf_results[ctx]['tokens_per_sec']} tok/s | VRAM: {hf_results[ctx]['peak_vram_mb']} MB)")
                except Exception as e_step:
                    print(f" ❌ Failed ({type(e_step).__name__})")
                    traceback.print_exc()
                    hf_results[ctx] = {"ttft_ms": "OOM/Fail", "tokens_per_sec": "OOM/Fail", "peak_vram_mb": "N/A"}

            del hf_model
            clear_memory()
        except Exception as e:
            print(f"⚠️ Failed to load baseline model via mlx-lm ({e}). Skipping HF baseline run.")
    else:
        print("⚠️ 'mlx-lm' package not found (`pip install mlx-lm`). Skipping HF baseline run.")

    # 2. Benchmark Native MLX BareTorch Hybrid Model (FP16)
    clear_memory()
    print(f"\n⚙️ Instantiating Native BareTorch MLX Blueprint (~1237M params in FP16)...")
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
    bt_model.set_dtype(mx.float16)

    for ctx in args.prompt_lens:
        print(f"  ├─ Benchmarking BareTorch (MLX FP16) @ Context: {ctx:<5} tokens...", end="", flush=True)
        try:
            bt_results[ctx] = benchmark_mlx_model(bt_model, ctx, args.gen_len, vocab_size, is_mlx_lm=False)
            print(f" ✅ (TTFT: {bt_results[ctx]['ttft_ms']} ms | Decode: {bt_results[ctx]['tokens_per_sec']} tok/s | VRAM: {bt_results[ctx]['peak_vram_mb']} MB)")
        except Exception as e_step:
            print(f" ❌ Failed ({type(e_step).__name__})")
            traceback.print_exc()
            bt_results[ctx] = {"ttft_ms": "Fail", "tokens_per_sec": "Fail", "peak_vram_mb": "N/A"}

    del bt_model
    clear_memory()

    # 3. Comparative Summary Report
    print("\n" + "=" * 130)
    print("📊 STANDARDIZED BARETORCH vs. LLAMA 3.2 1B CONTEXT REPORT (NATIVE APPLE SILICON MLX FP16)")
    print("=" * 130)
    print(f"{'Context Length':<15} | {'BareTorch TTFT':<16} | {'Llama TTFT':<16} | {'BareTorch Decode':<18} | {'Llama Decode':<18} | {'VRAM (BT / HF)':<20}")
    print("-" * 130)

    for ctx in args.prompt_lens:
        bt = bt_results.get(ctx, {})
        hf = hf_results.get(ctx, {})

        bt_ttft = f"{bt.get('ttft_ms', 'N/A')} ms"
        hf_ttft = f"{hf.get('ttft_ms', 'N/A')} ms"
        bt_dec = f"{bt.get('tokens_per_sec', 'N/A')} tok/s"
        hf_dec = f"{hf.get('tokens_per_sec', 'N/A')} tok/s"
        vram_str = f"{bt.get('peak_vram_mb', 'N/A')} / {hf.get('peak_vram_mb', 'N/A')} MB"

        print(f"{ctx:<15} | {bt_ttft:<16} | {hf_ttft:<16} | {bt_dec:<18} | {hf_dec:<18} | {vram_str:<20}")

    print("=" * 130)


if __name__ == "__main__":
    main()