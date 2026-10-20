# /home/martinkb/Desktop/BareTorch_F/benchmark_context_mlx.py
import argparse
import csv
import gc
import logging
import os
import platform
import time

# Prevent macOS MPS multi-threaded weight loading segmentation faults
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from baretorch.integration_mlx.modeling_baretorch import BareTorchForCausalLMMLX

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_mlx")


def flush_mlx_cache():
    """Flushes unused buffers from the Metal cache WITHOUT resetting peak memory stats."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def reset_mlx_memory_stats():
    """Flushes cache and resets MLX peak memory high-water mark at the start of a trial."""
    flush_mlx_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def get_mlx_peak_memory_gb() -> float:
    """Returns peak memory allocated by MLX in Gigabytes."""
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024**3)
    return 0.0


def clear_mps_memory():
    """Flushes garbage collection and clears PyTorch MPS cache on Apple Silicon."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def get_mps_allocated_gb() -> float:
    """Returns current allocated memory on PyTorch MPS in Gigabytes."""
    if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024**3)
    return 0.0


def enable_last_logit_only_torch(model):
    """Ensures PyTorch model's lm_head only projects the final token's hidden state
    into vocabulary logits during prefill, matching benchmark_context.py.
    """
    head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else getattr(model, "lm_head", None)
    )

    if (
        head is not None
        and isinstance(head, torch.nn.Linear)
        and not getattr(head, "_is_efficient_patched", False)
    ):
        orig_forward = head.forward

        def efficient_head_forward(hidden_states, *args, **kwargs):
            if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
                hidden_states = hidden_states[:, -1:, :]
            return orig_forward(hidden_states, *args, **kwargs)

        head.forward = efficient_head_forward
        head._is_efficient_patched = True


def benchmark_mlx_at_context(
    model: nn.Module,
    synth_input: mx.array,
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
):
    """Measures TTFT, Prefill Throughput, Decode Velocity, and Peak RAM on MLX.
    Aligned 1-to-1 with benchmark_model_at_context in benchmark_context.py.
    """
    seq_len = synth_input.shape[1]

    # 1. Per-Context Untimed Warmup
    for _ in range(warmup):
        logits, past_kv = model(synth_input, use_cache=True, num_logits_to_keep=1)
        tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(tok, past_kv)

        for _ in range(min(4, gen_tokens)):
            logits, past_kv = model(tok, past_key_values=past_kv, use_cache=True)
            tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(tok, past_kv)
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
        mx.eval(logits, past_kv)
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
            logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(next_tok, past_kv)

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


def benchmark_qwen_mps(
    model_id: str,
    context_lengths: list[int],
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
):
    """Evaluates Qwen3.5 baseline on Apple Silicon using PyTorch MPS."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Loading Qwen baseline on {device.type.upper()}...")

    results = {}
    try:
        qwen = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        qwen.eval()

        enable_last_logit_only_torch(qwen)
        logger.info("Loaded Qwen on MPS with last-logit-only prefill patch applied.")
    except Exception as e:
        logger.error(f"Failed to initialize Qwen on MPS: {e}")
        return {ctx: {"error": f"INIT_FAIL: {e}"} for ctx in context_lengths}

    for ctx in context_lengths:
        print(f"  --> Benchmarking Qwen3.5-2B (PyTorch MPS) @ {ctx:,} tokens...")
        inp = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)

        prefill_latencies = []
        decode_latencies = []
        peak_memories = []

        try:
            # 1. Context-specific warmup
            with torch.no_grad():
                for _ in range(warmup):
                    out = qwen(inp, use_cache=True)
                    p_kv = out.past_key_values
                    tok = out.logits[:, -1:].argmax(dim=-1)
                    for _ in range(min(4, gen_tokens)):
                        step = qwen(tok, past_key_values=p_kv, use_cache=True)
                        p_kv = step.past_key_values
                        tok = step.logits[:, -1:].argmax(dim=-1)
                    del out, p_kv, tok, step
            clear_mps_memory()

            # 2. Timing Trials
            for _ in range(trials):
                clear_mps_memory()

                # --- A. Measure Prefill Phase (TTFT) ---
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = qwen(inp, use_cache=True)
                torch.mps.synchronize()
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                prefill_latencies.append(ttft_ms)

                past_kv = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
                prefill_mem_gb = get_mps_allocated_gb()
                del out

                clear_mps_memory()

                # --- B. Measure Autoregressive Decoding Phase ---
                torch.mps.synchronize()
                t_dec = time.perf_counter()
                with torch.no_grad():
                    for _ in range(gen_tokens):
                        step = qwen(tok, past_key_values=past_kv, use_cache=True)
                        past_kv = step.past_key_values
                        tok = step.logits[:, -1:].argmax(dim=-1)
                torch.mps.synchronize()
                dec_ms = (time.perf_counter() - t_dec) * 1000.0
                decode_latencies.append(dec_ms)

                decode_mem_gb = get_mps_allocated_gb()
                peak_memories.append(max(prefill_mem_gb, decode_mem_gb))

                del past_kv, tok, step
                clear_mps_memory()

            avg_ttft = float(np.mean(prefill_latencies))
            avg_dec = float(np.mean(decode_latencies))
            avg_peak = float(np.mean(peak_memories))

            results[ctx] = {
                "context_len": ctx,
                "ttft_ms": avg_ttft,
                "prefill_tok_s": (ctx / (avg_ttft / 1000.0)) if avg_ttft > 0 else 0.0,
                "decode_tok_s": (gen_tokens / (avg_dec / 1000.0)) if avg_dec > 0 else 0.0,
                "peak_ram_gb": avg_peak,
            }
        except Exception as e:
            logger.warning(f"Qwen failed @ {ctx:,} tokens on MPS: {e}")
            results[ctx] = {"error": "MPS_FAIL_OR_OOM"}
            clear_mps_memory()
        finally:
            del inp
            clear_mps_memory()

    del qwen
    clear_mps_memory()
    logger.info("Unloaded Qwen from MPS memory.\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="BareTorch MLX vs. Qwen Apple Silicon Comparative Benchmark"
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
        help="Hugging Face model ID for Qwen baseline",
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
        help="Benchmark Qwen3.5-2B via PyTorch MPS (default: True)",
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
        default="benchmark_apple_silicon.csv",
        help="Output CSV filepath",
    )
    args = parser.parse_args()

    soc_name = platform.processor() or "Apple Silicon"
    print("=" * 85)
    print(f"🍏 BARETORCH MLX vs. QWEN3.5 BENCHMARK ({soc_name.upper()})")
    print(f"BareTorch Path : {args.baretorch_path}")
    print(f"Qwen Model ID  : {args.qwen_model_id}")
    print(f"Contexts       : {args.context_lengths}")
    print(f"Decode Steps   : {args.gen_tokens} tokens")
    print(f"Trials         : {args.trials} per context length")
    print(f"Benchmark Qwen : {args.benchmark_qwen}")
    print(f"CSV Output     : {args.csv_path}")
    print("=" * 85)

    csv_rows = []
    baretorch_results = {}
    qwen_results = {}

    bt_label = "BareTorch 2.5B (MLX Eager)"

    # --------------------------------------------------------------------------
    # 1. Benchmark BareTorch MLX
    # --------------------------------------------------------------------------
    if os.path.exists(args.baretorch_path):
        logger.info(f"Loading BareTorch from '{args.baretorch_path}' in FP16...")
        model = BareTorchForCausalLMMLX.from_pretrained(
            args.baretorch_path, dtype=mx.float16
        )

        logger.info("Starting BareTorch MLX context evaluation loop...")
        for ctx in args.context_lengths:
            print(f"  --> Benchmarking {bt_label} @ {ctx:,} tokens...")
            synth_input = mx.random.randint(100, 10000, shape=(1, ctx))
            mx.eval(synth_input)

            try:
                metrics = benchmark_mlx_at_context(
                    model=model,
                    synth_input=synth_input,
                    gen_tokens=args.gen_tokens,
                    warmup=1,
                    trials=args.trials,
                )
                baretorch_results[ctx] = metrics
                csv_rows.append({
                    "model": bt_label,
                    "backend": "MLX",
                    "context_len": ctx,
                    "ttft_ms": round(metrics["ttft_ms"], 2),
                    "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
                    "decode_tok_s": round(metrics["decode_tok_s"], 2),
                    "peak_ram_gb": round(metrics["peak_ram_gb"], 2),
                    "status": "SUCCESS",
                })
            except Exception as e:
                print(f"  ❌ BareTorch MLX failed @ {ctx:,} tokens: {e}")
                baretorch_results[ctx] = {"error": str(e)}
                csv_rows.append({
                    "model": bt_label,
                    "backend": "MLX",
                    "context_len": ctx,
                    "ttft_ms": None,
                    "prefill_tok_s": None,
                    "decode_tok_s": None,
                    "peak_ram_gb": None,
                    "status": "FAIL",
                })
                reset_mlx_memory_stats()
            del synth_input

        del model
        reset_mlx_memory_stats()
        logger.info("Unloaded BareTorch from unified memory.\n")
    else:
        logger.warning(f"BareTorch path '{args.baretorch_path}' not found. Skipping.")

    # --------------------------------------------------------------------------
    # 2. Benchmark Qwen3.5-2B (PyTorch MPS)
    # --------------------------------------------------------------------------
    if args.benchmark_qwen:
        logger.info("Initiating Qwen baseline evaluation...")
        qwen_results = benchmark_qwen_mps(
            model_id=args.qwen_model_id,
            context_lengths=args.context_lengths,
            gen_tokens=args.gen_tokens,
            warmup=1,
            trials=args.trials,
        )
        for ctx in args.context_lengths:
            if ctx in qwen_results:
                res = qwen_results[ctx]
                if "error" not in res:
                    csv_rows.append({
                        "model": "Qwen3.5-2B",
                        "backend": "PyTorch MPS",
                        "context_len": ctx,
                        "ttft_ms": round(res["ttft_ms"], 2),
                        "prefill_tok_s": round(res["prefill_tok_s"], 2),
                        "decode_tok_s": round(res["decode_tok_s"], 2),
                        "peak_ram_gb": round(res["peak_ram_gb"], 2),
                        "status": "SUCCESS",
                    })
                else:
                    csv_rows.append({
                        "model": "Qwen3.5-2B",
                        "backend": "PyTorch MPS",
                        "context_len": ctx,
                        "ttft_ms": None,
                        "prefill_tok_s": None,
                        "decode_tok_s": None,
                        "peak_ram_gb": None,
                        "status": res["error"],
                    })

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
    print(f"📊 COMPARATIVE BENCHMARK REPORT ({soc_name})")
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
                print(f"| {ctx:,} | Qwen3.5-2B (PyTorch MPS) | 💥 {qw['error']} | — | — | — |")
            else:
                print(
                    f"| {ctx:,} | Qwen3.5-2B (PyTorch MPS) | {qw['ttft_ms']:.1f} ms | "
                    f"{qw['prefill_tok_s']:,.1f} tok/s | {qw['decode_tok_s']:.2f} tok/s | "
                    f"{qw['peak_ram_gb']:.2f} GB |"
                )
        print("| --- | --- | --- | --- | --- | --- |")

    print("=" * 95)


if __name__ == "__main__":
    main()