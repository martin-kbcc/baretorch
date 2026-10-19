# /home/martinkb/Desktop/BareTorch_F/benchmark_context_mlx.py
import argparse
import csv
import gc
import logging
import os
import platform
import time

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from baretorch.integration_mlx.modeling_baretorch import BareTorchForCausalLMMLX
from baretorch.modeling_mlx.cs_lrad import LRADDecoderBlock
from baretorch.modeling_mlx.transformer import TransformerDecoderBlock

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_mlx")


def clear_mlx_memory():
  gc.collect()
  if hasattr(mx, "clear_cache"):
    mx.clear_cache()
  if hasattr(mx, "reset_peak_memory"):
    mx.reset_peak_memory()


def get_peak_memory_gb() -> float:
  if hasattr(mx, "get_peak_memory"):
    return mx.get_peak_memory() / (1024**3)
  return 0.0


def profile_single_decode_step(model, next_tok, past_kv):
  """Measures exact per-module latency for 1 decode step to isolate bottlenecks."""
  mx.synchronize()

  # 1. Embedding
  t0 = time.perf_counter()
  h = model.model.token_embedding(next_tok)
  mx.eval(h)
  mx.synchronize()
  t_emb = (time.perf_counter() - t0) * 1000.0

  t_transformer_total = 0.0
  t_lrad_total = 0.0
  transformer_count = 0
  lrad_count = 0

  # 2. Per-layer dispatch
  for i, layer in enumerate(model.model.layers):
    p_state = past_kv[i]
    t_layer_start = time.perf_counter()

    if isinstance(layer, TransformerDecoderBlock):
      h, _ = layer(h, past_kv=p_state)
      mx.eval(h)
      mx.synchronize()
      t_transformer_total += (time.perf_counter() - t_layer_start) * 1000.0
      transformer_count += 1
    else:
      h, _ = layer(h, past_state=p_state, use_cache=True)
      mx.eval(h)
      mx.synchronize()
      t_lrad_total += (time.perf_counter() - t_layer_start) * 1000.0
      lrad_count += 1

  # 3. Head
  t0 = time.perf_counter()
  h_norm = model.model.final_norm(h)
  logits = model.lm_head(h_norm)
  tok = mx.argmax(logits[:, -1:, :], axis=-1)
  mx.eval(tok)
  mx.synchronize()
  t_head = (time.perf_counter() - t0) * 1000.0

  print(
      f"      [Breakdown] Embed: {t_emb:.2f}ms | {lrad_count} LRAD Layers:"
      f" {t_lrad_total:.2f}ms ({t_lrad_total/lrad_count:.2f}ms/ea) |"
      f" {transformer_count} Trans Layers: {t_transformer_total:.2f}ms"
      f" ({t_transformer_total/transformer_count:.2f}ms/ea) | Head:"
      f" {t_head:.2f}ms"
  )


def benchmark_mlx_at_context(
    model: nn.Module,
    synth_input: mx.array,
    gen_tokens: int = 64,
    warmup: int = 1,
    trials: int = 3,
):
  seq_len = synth_input.shape[1]

  # 1. Warmup
  for _ in range(warmup):
    logits, past_kv = model(synth_input, use_cache=True, num_logits_to_keep=1)
    tok = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(tok, past_kv)

    for _ in range(min(4, gen_tokens)):
      logits, past_kv = model(tok, past_key_values=past_kv, use_cache=True)
      tok = mx.argmax(logits[:, -1:, :], axis=-1)
      mx.eval(tok, past_kv)
    del logits, past_kv, tok
  clear_mlx_memory()

  prefill_latencies = []
  decode_latencies = []
  peak_memories = []

  # 2. Timing Trials
  for trial_idx in range(trials):
    clear_mlx_memory()

    # --- A. Prefill ---
    mx.synchronize()
    t0 = time.perf_counter()

    logits, past_kv = model(synth_input, use_cache=True, num_logits_to_keep=1)
    next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(next_tok, past_kv)
    mx.synchronize()

    ttft_ms = (time.perf_counter() - t0) * 1000.0
    prefill_latencies.append(ttft_ms)
    del logits

    # Print component profile during trial 1
    if trial_idx == 0:
      profile_single_decode_step(model, next_tok, past_kv)

    # --- B. Autoregressive Decode ---
    mx.synchronize()
    t_start_dec = time.perf_counter()

    for _ in range(gen_tokens):
      logits, past_kv = model(next_tok, past_key_values=past_kv, use_cache=True)
      next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
      mx.eval(next_tok, past_kv)

    mx.synchronize()
    dec_time_ms = (time.perf_counter() - t_start_dec) * 1000.0
    decode_latencies.append(dec_time_ms)

    peak_mem_gb = get_peak_memory_gb()
    peak_memories.append(peak_mem_gb)

    del past_kv, next_tok
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


def main():
  parser = argparse.ArgumentParser(
      description="BareTorch MLX Diagnostic Benchmark"
  )
  parser.add_argument(
      "--baretorch_path",
      type=str,
      default="./checkpoint-132000",
      help="Local path to checkpoint directory",
  )
  parser.add_argument(
      "--context_lengths",
      type=int,
      nargs="+",
      default=[512, 2048, 8192, 16384, 32768],
      help="Context lengths to benchmark",
  )
  parser.add_argument(
      "--gen_tokens",
      type=int,
      default=64,
      help="Tokens to decode autoregressively",
  )
  parser.add_argument(
      "--trials",
      type=int,
      default=2,
      help="Measurement trials per context length",
  )
  parser.add_argument(
      "--csv_path",
      type=str,
      default="benchmark_apple_silicon_diagnostic.csv",
      help="Output CSV path",
  )
  args = parser.parse_args()

  soc_name = platform.processor() or "Apple Silicon"
  print("=" * 85)
  print(f"🍏 BARETORCH MLX DIAGNOSTIC BENCHMARK ({soc_name.upper()})")
  print(f"Checkpoint    : {args.baretorch_path}")
  print(f"Contexts      : {args.context_lengths}")
  print(f"Decode Steps  : {args.gen_tokens} tokens")
  print(f"Trials        : {args.trials} per context length")
  print("=" * 85)

  csv_rows = []
  baretorch_results = {}
  bt_label = "BareTorch 2.5B (MLX Eager)"

  if os.path.exists(args.baretorch_path):
    logger.info(f"Loading BareTorch from '{args.baretorch_path}' in FP16...")
    model = BareTorchForCausalLMMLX.from_pretrained(
        args.baretorch_path, dtype=mx.float16
    )

    for ctx in args.context_lengths:
      print(f"\n  --> Benchmarking {bt_label} @ {ctx:,} tokens...")
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
        print(f"  ❌ Failed @ {ctx:,} tokens: {e}")
        baretorch_results[ctx] = {"error": str(e)}
      del synth_input

    del model
    clear_mlx_memory()
  else:
    logger.warning(f"Path '{args.baretorch_path}' not found.")

  print("\n" + "=" * 95)
  print(f"📊 SUMMARY REPORT ({soc_name})")
  print("=" * 95)
  print(
      "| Context | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak"
      " RAM (GB) |"
  )
  print("| :---: | :--- | :---: | :---: | :---: | :---: |")

  for ctx in args.context_lengths:
    if ctx in baretorch_results:
      bt = baretorch_results[ctx]
      if "error" in bt:
        print(f"| {ctx:,} | **{bt_label}** | 💥 {bt['error']} | — | — | — |")
      else:
        print(
            f"| {ctx:,} | **{bt_label}** | {bt['ttft_ms']:.1f} ms |"
            f" {bt['prefill_tok_s']:,.1f} tok/s | {bt['decode_tok_s']:.2f}"
            f" tok/s | {bt['peak_ram_gb']:.2f} GB |"
        )
  print("=" * 95)


if __name__ == "__main__":
  main()