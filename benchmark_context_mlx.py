# /home/martinkb/Desktop/BareTorch_F/benchmark_context_mlx.py
import argparse
import csv
import gc
import logging
import os
import platform
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from baretorch.integration_mlx.modeling_baretorch import BareTorchForCausalLMMLX

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_mlx")


def clear_mlx_memory():
  """Forces garbage collection and clears Apple Silicon unified memory allocator cache."""
  gc.collect()
  if mx.metal.is_available():
    mx.metal.clear_cache()
    mx.metal.reset_peak_memory()


def get_peak_memory_gb() -> float:
  """Returns peak memory allocated on Apple Silicon in Gigabytes."""
  if mx.metal.is_available():
    return mx.metal.get_peak_memory() / (1024**3)
  return 0.0


def benchmark_mlx_at_context(
    model: nn.Module,
    synth_input: mx.array,
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
):
  """Measures TTFT (Prefill Latency), Prefill Velocity, Autoregressive Decode

  Velocity, and Peak RAM on Apple Silicon using strict synchronization.
  """
  seq_len = synth_input.shape[1]

  # 1. Per-Context Warmup (Materializes Metal execution pipelines for exact context shape)
  for _ in range(warmup):
    out = model(synth_input, use_cache=True, num_logits_to_keep=1)
    past_kv = out.past_key_values
    tok = mx.argmax(out.logits[:, -1:, :], axis=-1)
    for _ in range(min(4, gen_tokens)):
      step_out = model(tok, past_key_values=past_kv, use_cache=True)
      past_kv = step_out.past_key_values
      tok = mx.argmax(step_out.logits[:, -1:, :], axis=-1)
      mx.eval(tok)
    mx.eval(past_kv)
    del out, past_kv, step_out, tok
  clear_mlx_memory()

  prefill_latencies = []
  decode_latencies = []
  peak_memories = []

  # 2. Timing Trials
  for _ in range(trials):
    clear_mlx_memory()

    # --- A. Measure Prefill Phase (Time to First Token) ---
    mx.synchronize()
    t0 = time.perf_counter()

    out = model(synth_input, use_cache=True, num_logits_to_keep=1)
    mx.eval(out.logits, out.past_key_values)
    mx.synchronize()

    ttft_ms = (time.perf_counter() - t0) * 1000.0
    prefill_latencies.append(ttft_ms)

    past_kv = out.past_key_values
    next_tok = mx.argmax(out.logits[:, -1:, :], axis=-1)
    del out

    # --- B. Measure Autoregressive Decoding Phase ---
    mx.synchronize()
    t_start_dec = time.perf_counter()

    for _ in range(gen_tokens):
      step_out = model(next_tok, past_key_values=past_kv, use_cache=True)
      past_kv = step_out.past_key_values
      next_tok = mx.argmax(step_out.logits[:, -1:, :], axis=-1)
      mx.eval(next_tok)

    mx.synchronize()
    dec_time_ms = (time.perf_counter() - t_start_dec) * 1000.0
    decode_latencies.append(dec_time_ms)

    peak_mem_gb = get_peak_memory_gb()
    peak_memories.append(peak_mem_gb)

    del past_kv, step_out, next_tok
    clear_mlx_memory()

  avg_ttft_ms = float(np.mean(prefill_latencies))
  avg_prefill_tok_s = (
      (seq_len / (avg_ttft_ms / 1000.0)) if avg_ttft_ms > 0 else 0.0
  )
  avg_decode_ms = float(np.mean(decode_latencies))
  avg_decode_tok_s = (
      (gen_tokens / (avg_decode_ms / 1000.0)) if avg_decode_ms > 0 else 0.0
  )
  avg_peak_ram = float(np.mean(peak_memories))

  return {
      "context_len": seq_len,
      "ttft_ms": avg_ttft_ms,
      "prefill_tok_s": avg_prefill_tok_s,
      "decode_tok_s": avg_decode_tok_s,
      "peak_ram_gb": avg_peak_ram,
  }


def benchmark_qwen_mps_fallback(
    model_id: str,
    context_lengths: list[int],
    gen_tokens: int = 64,
    trials: int = 3,
):
  """Evaluates Qwen baseline on Apple Silicon via PyTorch MPS to benchmark

  kernel-dependent architectures against BareTorch's kernel-free pure GEMMs.
  """
  import torch
  from transformers import AutoModelForCausalLM

  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  logger.info(f"Loading Qwen baseline on {device.type.upper()}...")

  results = {}
  try:
    qwen = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True,
    ).eval()
  except Exception as e:
    logger.error(
        f"Qwen baseline cannot load on Apple Silicon MPS (missing native"
        f" kernel): {e}"
    )
    return {ctx: {"error": "FAILED_INIT"} for ctx in context_lengths}

  for ctx in context_lengths:
    print(f"  --> Benchmarking Qwen MPS Baseline @ {ctx:,} tokens...")
    inp = torch.randint(100, 10000, (1, ctx), device=device)
    prefill_latencies = []
    decode_latencies = []

    try:
      for _ in range(trials):
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

        # Prefill
        t0 = time.perf_counter()
        with torch.no_grad():
          out = qwen(inp, use_cache=True)
          past_kv = out.past_key_values
          tok = out.logits[:, -1:].argmax(dim=-1)
        torch.mps.synchronize()
        prefill_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Decode
        t_dec = time.perf_counter()
        with torch.no_grad():
          for _ in range(gen_tokens):
            step = qwen(tok, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            tok = step.logits[:, -1:].argmax(dim=-1)
        torch.mps.synchronize()
        decode_latencies.append((time.perf_counter() - t_dec) * 1000.0)

        del out, past_kv, tok, step
        gc.collect()
        torch.mps.empty_cache()

      avg_ttft = float(np.mean(prefill_latencies))
      avg_dec = float(np.mean(decode_latencies))
      results[ctx] = {
          "context_len": ctx,
          "ttft_ms": avg_ttft,
          "prefill_tok_s": (
              (ctx / (avg_ttft / 1000.0)) if avg_ttft > 0 else 0.0
          ),
          "decode_tok_s": (
              (gen_tokens / (avg_dec / 1000.0)) if avg_dec > 0 else 0.0
          ),
          "peak_ram_gb": (
              torch.mps.current_allocated_memory() / (1024**3)
              if hasattr(torch.mps, "current_allocated_memory")
              else 0.0
          ),
      }
    except Exception as e:
      logger.warning(f"Qwen failed @ {ctx:,} tokens on MPS: {e}")
      results[ctx] = {"error": "MPS_FALLBACK_FAIL"}
    del inp

  del qwen
  gc.collect()
  torch.mps.empty_cache()
  return results


def main():
  parser = argparse.ArgumentParser(
      description=(
          "BareTorch MLX vs. Qwen Apple Silicon Benchmark (M-Series Unified"
          " Memory)"
      )
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
      help="Hugging Face ID for Qwen baseline comparison",
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
      "--compile",
      action="store_true",
      default=True,
      help="Enable mx.compile optimization graph",
  )
  parser.add_argument(
      "--no_compile",
      action="store_false",
      dest="compile",
      help="Disable mx.compile graph fusion",
  )
  parser.add_argument(
      "--benchmark_qwen",
      action="store_true",
      default=False,
      help="Benchmark Qwen via PyTorch MPS fallback for direct comparison",
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
  print(f"🍏 BARETORCH MLX APPLE SILICON BENCHMARK ({soc_name.upper()})")
  print(f"Framework     : MLX (Native Metal Unified Memory)")
  print(f"Checkpoint    : {args.baretorch_path}")
  print(f"Contexts      : {args.context_lengths}")
  print(f"Decode Steps  : {args.gen_tokens} tokens")
  print(f"Trials        : {args.trials} per context length")
  print(f"mx.compile    : {args.compile}")
  print(f"CSV Output    : {args.csv_path}")
  print("=" * 85)

  csv_rows = []
  baretorch_results = {}
  qwen_results = {}

  bt_label = (
      "BareTorch 2.5B (MLX Compiled)"
      if args.compile
      else "BareTorch 2.5B (MLX Eager)"
  )

  # --------------------------------------------------------------------------
  # 1. Benchmark BareTorch MLX
  # --------------------------------------------------------------------------
  if os.path.exists(args.baretorch_path):
    logger.info(
        f"Loading BareTorch MLX from '{args.baretorch_path}' in FP16..."
    )
    model = BareTorchForCausalLMMLX.from_pretrained(
        args.baretorch_path, dtype=mx.float16
    )

    if args.compile:
      logger.info(
          "⚡ Applying @mx.compile across model forward step transformations..."
      )
      model = mx.compile(model)

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
        clear_mlx_memory()
      del synth_input

    del model
    clear_mlx_memory()
    logger.info("Unloaded BareTorch from unified memory.\n")
  else:
    logger.warning(
        f"BareTorch checkpoint path '{args.baretorch_path}' not found."
        " Skipping."
    )

  # --------------------------------------------------------------------------
  # 2. Benchmark Qwen Baseline (Optional MPS Comparison)
  # --------------------------------------------------------------------------
  if args.benchmark_qwen:
    logger.info("Initiating Qwen baseline evaluation...")
    qwen_results = benchmark_qwen_mps_fallback(
        model_id=args.qwen_model_id,
        context_lengths=args.context_lengths,
        gen_tokens=args.gen_tokens,
        trials=args.trials,
    )
    for ctx, res in qwen_results.items():
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
  # 3. Export CSV
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
  print(f"📊 APPLE SILICON COMPARATIVE BENCHMARK REPORT ({soc_name})")
  print("=" * 95)
  header = (
      "| Context | Model (Backend) | TTFT (ms) | Prefill (tok/s) | Decode"
      " (tok/s) | Peak RAM (GB) |"
  )
  sep = "| :---: | :--- | :---: | :---: | :---: | :---: |"
  print(header)
  print(sep)

  for ctx in args.context_lengths:
    if ctx in baretorch_results:
      bt = baretorch_results[ctx]
      if "error" in bt:
        print(
            f"| {ctx:,} | **{bt_label}** | 💥 {bt['error']} | — | — | — |"
        )
      else:
        print(
            f"| {ctx:,} | **{bt_label}** | {bt['ttft_ms']:.1f} ms |"
            f" {bt['prefill_tok_s']:,.1f} tok/s | {bt['decode_tok_s']:.2f}"
            f" tok/s | {bt['peak_ram_gb']:.2f} GB |"
        )

    if ctx in qwen_results:
      qw = qwen_results[ctx]
      if "error" in qw:
        print(f"| {ctx:,} | Qwen3.5-2B (PyTorch MPS) | 💥 {qw['error']} | — | — | — |")
      else:
        print(
            f"| {ctx:,} | Qwen3.5-2B (PyTorch MPS) | {qw['ttft_ms']:.1f} ms |"
            f" {qw['prefill_tok_s']:,.1f} tok/s | {qw['decode_tok_s']:.2f}"
            f" tok/s | {qw['peak_ram_gb']:.2f} GB |"
        )
    print("| --- | --- | --- | --- | --- | --- |")

  print("=" * 95)


if __name__ == "__main__":
  main()