# benchmark_context_mlx_4bit.py
import argparse
import csv
import gc
import json
import logging
import os
import platform
import time

# Prevent macOS multi-threaded weight loading issues
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration_mlx.modeling_baretorch import (
    BareTorchForCausalLMMLX,
    extract_baretorch_cache_arrays,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_mlx_4bit")


def flush_mlx_cache():
    """Flushes unused buffers from the Metal cache WITHOUT resetting peak memory stats."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def reset_mlx_memory_stats():
    """Flushes cache and resets MLX peak memory high-water mark at the start of a trial."""
    flush_mlx_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def get_mlx_peak_memory_gb() -> float:
    """Returns peak memory allocated by MLX in Gigabytes."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024**3)
    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024**3)
    return 0.0


def extract_mlx_cache_states(cache):
    """Extracts underlying MLX arrays from mlx_lm cache objects for explicit mx.eval()."""
    if cache is None:
        return []
    states = []
    for c in cache:
        if hasattr(c, "state") and c.state is not None:
            if isinstance(c.state, (list, tuple)):
                states.extend([s for s in c.state if s is not None])
            else:
                states.append(c.state)
        elif isinstance(c, (list, tuple)):
            states.extend([s for s in c if s is not None])
    return states


def enable_last_logit_only_mlx_qwen(model):
    """Ensures mlx-lm's Qwen model only projects the final token's hidden state into
    vocabulary logits during prefill, matching BareTorch's num_logits_to_keep=1.
    """
    inner_model = getattr(model, "model", None)
    if inner_model is not None and not getattr(inner_model, "_is_efficient_patched", False):
        orig_call = inner_model.__call__.__func__

        def efficient_inner_call(self_inner, inputs, *args, **kwargs):
            h = orig_call(self_inner, inputs, *args, **kwargs)
            if isinstance(h, mx.array) and h.ndim == 3 and h.shape[1] > 1:
                h = h[:, -1:, :]
            return h

        inner_model.__class__ = type(
            inner_model.__class__.__name__ + "LastLogitPatched",
            (inner_model.__class__,),
            {"__call__": efficient_inner_call},
        )
        inner_model._is_efficient_patched = True


def quantize_mlx_model_in_place(model: nn.Module, bits: int = 4, group_size: int = 64):
    """Applies native MLX quantization to all compatible Linear/Embedding layers in chunks."""
    if bits in (4, 8):
        def class_predicate(path, module):
            if hasattr(module, "weight"):
                return module.weight.ndim == 2 and (module.weight.shape[-1] % group_size == 0)
            return False

        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=class_predicate)

    # Evaluate layer-by-layer to prevent single giant Metal command buffer timeout
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            mx.eval(layer.parameters())
            mx.synchronize()
    mx.eval(model.parameters())
    mx.synchronize()
    flush_mlx_cache()


def load_and_quantize_baretorch_mlx(
    checkpoint_path: str,
    bits: int = 4,
    group_size: int = 64,
    dtype=mx.float16,
) -> BareTorchForCausalLMMLX:
    model = BareTorchForCausalLMMLX.from_pretrained(
        checkpoint_path, dtype=dtype, fuse=True
    )
    if bits in (4, 8):
        logger.info(
            f"Quantizing fused BareTorch to {bits}-bit (group_size={group_size}) via mlx.nn.quantize..."
        )
        quantize_mlx_model_in_place(model, bits=bits, group_size=group_size)
    reset_mlx_memory_stats()
    return model


def benchmark_baretorch_mlx_at_context(
    model: nn.Module,
    synth_input: mx.array,
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
):
    """Measures TTFT, Prefill Throughput, Decode Velocity, and Peak RAM for BareTorch on MLX."""
    seq_len = synth_input.shape[1]

    # 1. Per-Context Untimed Warmup
    for _ in range(warmup):
        logits, past_kv = model(synth_input, use_cache=True, num_logits_to_keep=1)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(tok, extract_baretorch_cache_arrays(past_kv))

        for _ in range(min(4, gen_tokens)):
            logits, past_kv = model(tok, past_key_values=past_kv, use_cache=True)
            tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(tok, extract_baretorch_cache_arrays(past_kv))
        del logits, past_kv, tok
    reset_mlx_memory_stats()

    prefill_latencies = []
    decode_latencies = []
    peak_memories = []

    # 2. Timing Trials
    for _ in range(trials):
        reset_mlx_memory_stats()

        # --- A. Measure Prefill Phase (TTFT) ---
        mx.synchronize()
        t0 = time.perf_counter()

        logits, past_kv = model(synth_input, use_cache=True, num_logits_to_keep=1)
        mx.eval(logits, extract_baretorch_cache_arrays(past_kv))
        mx.synchronize()

        ttft_ms = (time.perf_counter() - t0) * 1000.0
        prefill_latencies.append(ttft_ms)

        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_tok)
        del logits

        prefill_peak_gb = get_mlx_peak_memory_gb()
        flush_mlx_cache()

        # --- B. Measure Autoregressive Decoding Phase ---
        mx.synchronize()
        t_start_dec = time.perf_counter()

        for _ in range(gen_tokens):
            logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(next_tok, extract_baretorch_cache_arrays(past_kv))

        mx.synchronize()
        dec_time_ms = (time.perf_counter() - t_start_dec) * 1000.0
        decode_latencies.append(dec_time_ms)

        decode_peak_gb = get_mlx_peak_memory_gb()
        peak_memories.append(max(prefill_peak_gb, decode_peak_gb))

        del past_kv, next_tok, logits
        reset_mlx_memory_stats()

    avg_ttft_ms = float(np.mean(prefill_latencies))
    avg_prefill_tok_s = (seq_len / (avg_ttft_ms / 1000.0)) if avg_ttft_ms > 0 else 0.0
    avg_decode_ms = float(np.mean(decode_latencies))
    avg_decode_tok_s = (gen_tokens / (avg_decode_ms / 1000.0)) if avg_decode_ms > 0 else 0.0
    avg_peak_ram = float(np.mean(peak_memories))

    return {
        "context_len": seq_len,
        "ttft_ms": avg_ttft_ms,
        "prefill_tok_s": avg_prefill_tok_s,
        "decode_tok_s": avg_decode_tok_s,
        "peak_ram_gb": avg_peak_ram,
    }


def benchmark_qwen_mlx_at_context(
    model: nn.Module,
    synth_input: mx.array,
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
    prefill_step_size: int = 4096,
):
    """Measures TTFT, Prefill Throughput, Decode Velocity, and Peak RAM for Qwen3.5 on Native MLX."""
    from mlx_lm.models.cache import make_prompt_cache

    seq_len = synth_input.shape[1]

    def run_chunked_prefill(inp_tokens, cache_obj):
        total_len = inp_tokens.shape[1]
        logits_out = None
        for start in range(0, total_len, prefill_step_size):
            end = min(start + prefill_step_size, total_len)
            logits_out = model(inp_tokens[:, start:end], cache=cache_obj)
            if end < total_len:
                mx.eval(extract_mlx_cache_states(cache_obj))
        return logits_out

    # 1. Per-Context Untimed Warmup
    for _ in range(warmup):
        cache = make_prompt_cache(model)
        logits = run_chunked_prefill(synth_input, cache)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(tok, extract_mlx_cache_states(cache))

        for _ in range(min(4, gen_tokens)):
            logits = model(tok, cache=cache)
            tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(tok, extract_mlx_cache_states(cache))
        del logits, cache, tok
    reset_mlx_memory_stats()

    prefill_latencies = []
    decode_latencies = []
    peak_memories = []

    # 2. Timing Trials
    for _ in range(trials):
        reset_mlx_memory_stats()
        cache = make_prompt_cache(model)

        # --- A. Measure Prefill Phase (TTFT) ---
        mx.synchronize()
        t0 = time.perf_counter()

        logits = run_chunked_prefill(synth_input, cache)
        mx.eval(logits, extract_mlx_cache_states(cache))
        mx.synchronize()

        ttft_ms = (time.perf_counter() - t0) * 1000.0
        prefill_latencies.append(ttft_ms)

        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_tok)
        del logits

        # Record prefill high-water mark BEFORE flushing dead buffers for decode
        prefill_peak_gb = get_mlx_peak_memory_gb()
        flush_mlx_cache()

        # --- B. Measure Autoregressive Decoding Phase ---
        mx.synchronize()
        t_start_dec = time.perf_counter()

        for _ in range(gen_tokens):
            logits = model(next_tok, cache=cache)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(next_tok, extract_mlx_cache_states(cache))

        mx.synchronize()
        dec_time_ms = (time.perf_counter() - t_start_dec) * 1000.0
        decode_latencies.append(dec_time_ms)

        decode_peak_gb = get_mlx_peak_memory_gb()
        peak_memories.append(max(prefill_peak_gb, decode_peak_gb))

        del cache, next_tok, logits
        reset_mlx_memory_stats()

    avg_ttft_ms = float(np.mean(prefill_latencies))
    avg_prefill_tok_s = (seq_len / (avg_ttft_ms / 1000.0)) if avg_ttft_ms > 0 else 0.0
    avg_decode_ms = float(np.mean(decode_latencies))
    avg_decode_tok_s = (gen_tokens / (avg_decode_ms / 1000.0)) if avg_decode_ms > 0 else 0.0
    avg_peak_ram = float(np.mean(peak_memories))

    return {
        "context_len": seq_len,
        "ttft_ms": avg_ttft_ms,
        "prefill_tok_s": avg_prefill_tok_s,
        "decode_tok_s": avg_decode_tok_s,
        "peak_ram_gb": avg_peak_ram,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BareTorch MLX vs. Qwen3.5 MLX Native Quantized (4-bit / 8-bit) Apple Silicon Benchmark"
    )
    parser.add_argument(
        "--baretorch_path",
        type=str,
        default="./checkpoint-132000",
        help="Local path to BareTorch checkpoint directory",
    )
    parser.add_argument(
        "--qwen_model_id",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="Hugging Face model ID for Qwen baseline (quantized on-the-fly via MLX)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Quantization bit-width applied natively via mlx.nn.quantize (default: 4)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=64,
        help="Quantization group size for mlx.nn.quantize (default: 64)",
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Context window lengths to benchmark",
    )
    parser.add_argument(
        "--gen_tokens",
        type=int,
        default=64,
        help="Tokens to decode autoregressively per trial",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Measurement trials per context length",
    )
    parser.add_argument(
        "--benchmark_qwen",
        action="store_true",
        default=True,
        help="Benchmark Qwen3.5-2B in Native MLX (default: True)",
    )
    parser.add_argument(
        "--skip_qwen",
        action="store_false",
        dest="benchmark_qwen",
        help="Skip Qwen baseline benchmark",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmark_apple_silicon_4bit.csv",
        help="Output CSV filepath",
    )
    args = parser.parse_args()

    soc_name = platform.processor() or "Apple Silicon"
    quant_tag = f"{args.bits}-bit" if args.bits in (4, 8) else "FP16"
    bt_label = f"BareTorch 2.5B (MLX {quant_tag})"
    qw_label = f"Qwen3.5-2B (MLX {quant_tag})"

    print("=" * 85)
    print(f"🍏 BARETORCH MLX vs. QWEN3.5 MLX {quant_tag.upper()} BENCHMARK ({soc_name.upper()})")
    print(f"BareTorch Path : {args.baretorch_path}")
    print(f"Qwen Model ID  : {args.qwen_model_id}")
    print(f"Quantization   : {quant_tag} (group_size={args.group_size})")
    print(f"Contexts       : {args.context_lengths}")
    print(f"Decode Steps   : {args.gen_tokens} tokens")
    print(f"Trials         : {args.trials} per context length")
    print(f"Benchmark Qwen : {args.benchmark_qwen}")
    print(f"CSV Output     : {args.csv_path}")
    print("=" * 85)

    csv_rows = []
    baretorch_results = {}
    qwen_results = {}

    # --------------------------------------------------------------------------
    # 1. Benchmark BareTorch in Native MLX (CPU-stream batched load + quantize)
    # --------------------------------------------------------------------------
    if os.path.exists(args.baretorch_path):
        logger.info(f"Loading & quantizing BareTorch from '{args.baretorch_path}'...")
        bt_model = load_and_quantize_baretorch_mlx(
            checkpoint_path=args.baretorch_path,
            bits=args.bits,
            group_size=args.group_size,
            dtype=mx.float16,
        )

        logger.info(f"Starting {bt_label} context evaluation loop...")
        for ctx in args.context_lengths:
            print(f"  --> Benchmarking {bt_label} @ {ctx:,} tokens...")
            synth_input = mx.random.randint(100, 10000, shape=(1, ctx))
            mx.eval(synth_input)

            try:
                metrics = benchmark_baretorch_mlx_at_context(
                    model=bt_model,
                    synth_input=synth_input,
                    gen_tokens=args.gen_tokens,
                    warmup=1,
                    trials=args.trials,
                )
                baretorch_results[ctx] = metrics
                csv_rows.append({
                    "model": bt_label,
                    "backend": f"MLX {quant_tag}",
                    "context_len": ctx,
                    "ttft_ms": round(metrics["ttft_ms"], 2),
                    "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
                    "decode_tok_s": round(metrics["decode_tok_s"], 2),
                    "peak_ram_gb": round(metrics["peak_ram_gb"], 2),
                    "status": "SUCCESS",
                })
            except Exception as e:
                print(f"  ❌ {bt_label} failed @ {ctx:,} tokens: {e}")
                baretorch_results[ctx] = {"error": str(e)}
                csv_rows.append({
                    "model": bt_label,
                    "backend": f"MLX {quant_tag}",
                    "context_len": ctx,
                    "ttft_ms": None,
                    "prefill_tok_s": None,
                    "decode_tok_s": None,
                    "peak_ram_gb": None,
                    "status": "FAIL",
                })
                reset_mlx_memory_stats()
            del synth_input

        del bt_model
        reset_mlx_memory_stats()
        logger.info("Unloaded BareTorch from unified memory.\n")
    else:
        logger.warning(f"BareTorch path '{args.baretorch_path}' not found. Skipping.")

    # --------------------------------------------------------------------------
    # 2. Benchmark Qwen3.5-2B in Native MLX (Quantized identically via nn.quantize)
    # --------------------------------------------------------------------------
    if args.benchmark_qwen:
        logger.info(f"Loading Qwen baseline ('{args.qwen_model_id}') in Native MLX...")
        try:
            from mlx_lm import load as mlx_load

            qwen_model, _ = mlx_load(args.qwen_model_id)
            enable_last_logit_only_mlx_qwen(qwen_model)

            if args.bits in (4, 8):
                logger.info(
                    f"Quantizing Qwen3.5-2B to {args.bits}-bit (group_size={args.group_size}) via mlx.nn.quantize..."
                )
                quantize_mlx_model_in_place(qwen_model, bits=args.bits, group_size=args.group_size)
            reset_mlx_memory_stats()

            logger.info(f"Starting {qw_label} context evaluation loop...")
            for ctx in args.context_lengths:
                print(f"  --> Benchmarking {qw_label} @ {ctx:,} tokens...")
                synth_input = mx.random.randint(100, 10000, shape=(1, ctx))
                mx.eval(synth_input)

                try:
                    metrics = benchmark_qwen_mlx_at_context(
                        model=qwen_model,
                        synth_input=synth_input,
                        gen_tokens=args.gen_tokens,
                        warmup=1,
                        trials=args.trials,
                    )
                    qwen_results[ctx] = metrics
                    csv_rows.append({
                        "model": "Qwen3.5-2B",
                        "backend": f"MLX {quant_tag}",
                        "context_len": ctx,
                        "ttft_ms": round(metrics["ttft_ms"], 2),
                        "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
                        "decode_tok_s": round(metrics["decode_tok_s"], 2),
                        "peak_ram_gb": round(metrics["peak_ram_gb"], 2),
                        "status": "SUCCESS",
                    })
                except Exception as e:
                    print(f"  ❌ {qw_label} failed @ {ctx:,} tokens: {e}")
                    qwen_results[ctx] = {"error": str(e)}
                    csv_rows.append({
                        "model": "Qwen3.5-2B",
                        "backend": f"MLX {quant_tag}",
                        "context_len": ctx,
                        "ttft_ms": None,
                        "prefill_tok_s": None,
                        "decode_tok_s": None,
                        "peak_ram_gb": None,
                        "status": "FAIL",
                    })
                    reset_mlx_memory_stats()
                del synth_input

            del qwen_model
            reset_mlx_memory_stats()
            logger.info("Unloaded Qwen from unified memory.\n")
        except Exception as e:
            logger.error(
                f"Failed to load Qwen in Native MLX ({e}). Ensure `pip install -U mlx-lm` is installed on your Mac."
            )

    # --------------------------------------------------------------------------
    # 3. Save Structured CSV Results
    # --------------------------------------------------------------------------
    fieldnames = [
        "model",
        "backend",
        "context_len",
        "ttft_ms",
        "prefill_tok_s",
        "decode_tok_s",
        "peak_ram_gb",
        "status",
    ]
    with open(args.csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    logger.info(f"✅ Benchmark metrics saved to '{args.csv_path}'")

    # --------------------------------------------------------------------------
    # 4. Formatted Markdown Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"📊 COMPARATIVE {quant_tag.upper()} MLX BENCHMARK REPORT ({soc_name})")
    print("=" * 95)
    header = (
        "| Context | Model (Backend) | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak RAM (GB) |"
    )
    sep = "| :---: | :--- | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)

    for ctx in args.context_lengths:
        if ctx in baretorch_results:
            bt = baretorch_results[ctx]
            if "error" in bt:
                print(f"| {ctx:,} | **{bt_label}** | 💥 {bt['error']} | — | — | — |")
            else:
                print(
                    f"| {ctx:,} | **{bt_label}** | {bt['ttft_ms']:.1f} ms | "
                    f"{bt['prefill_tok_s']:,.1f} tok/s | {bt['decode_tok_s']:.2f} tok/s | "
                    f"{bt['peak_ram_gb']:.2f} GB |"
                )

        if ctx in qwen_results:
            qw = qwen_results[ctx]
            if "error" in qw:
                print(f"| {ctx:,} | {qw_label} | 💥 {qw['error']} | — | — | — |")
            else:
                print(
                    f"| {ctx:,} | {qw_label} | {qw['ttft_ms']:.1f} ms | "
                    f"{qw['prefill_tok_s']:,.1f} tok/s | {qw['decode_tok_s']:.2f} tok/s | "
                    f"{qw['peak_ram_gb']:.2f} GB |"
                )
        print("| --- | --- | --- | --- | --- | --- |")

    print("=" * 95)


if __name__ == "__main__":
    main()