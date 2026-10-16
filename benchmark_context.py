import argparse
import csv
import gc
import inspect
import logging
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baretorch import BareTorchConfig, BareTorchForCausalLM

# Set tensor precision and backend policies
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_cudnn_sdp(False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")


def clear_cuda():
  """Flushes garbage collection and resets CUDA memory allocator tracking."""
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()


def enable_last_logit_only(model):
  """Ensures that the model's output projection head (lm_head) only projects the

  final token's hidden state into vocabulary logits during prefill. This
  eliminates multi-gigabyte logits allocations across all models (including
  Qwen)
  regardless of whether the underlying model file exposes `num_logits_to_keep`.
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
      # During prefill (sequence length > 1), slice only the final position
      if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
        hidden_states = hidden_states[:, -1:, :]
      return orig_forward(hidden_states, *args, **kwargs)

    head.forward = efficient_head_forward
    head._is_efficient_patched = True


def get_forward_kwargs(model, use_cache=True, num_logits_to_keep=1):
  """Injects num_logits_to_keep if the model explicitly declares it in its signature."""
  kwargs = {"use_cache": use_cache}
  try:
    sig = inspect.signature(model.forward)
    if "num_logits_to_keep" in sig.parameters:
      kwargs["num_logits_to_keep"] = num_logits_to_keep
  except Exception:
    pass
  return kwargs


def benchmark_model_at_context(
    model, input_ids, gen_tokens=64, warmup=1, trials=3
):
  """Measures Time to First Token (TTFT), Prefill Throughput, Decode Velocity,

  and Peak VRAM across multiple runs using hardware-synchronized CUDA events.
  """
  seq_len = input_ids.shape[1]
  device = input_ids.device
  prefill_kwargs = get_forward_kwargs(
      model, use_cache=True, num_logits_to_keep=1
  )

  # 1. Per-Context Untimed Warmup (Triggers JIT compilation on specific context shapes)
  with torch.no_grad():
    for _ in range(warmup):
      out = model(input_ids, **prefill_kwargs)
      past_kv = out.past_key_values
      next_token = out.logits[:, -1:].argmax(dim=-1)
      for _ in range(min(4, gen_tokens)):
        step_out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        next_token = step_out.logits[:, -1:].argmax(dim=-1)
      del out, past_kv, step_out
  clear_cuda()

  prefill_latencies = []
  decode_latencies = []
  peak_memories = []

  # 2. Timing Trials
  for _ in range(trials):
    clear_cuda()

    # --- A. Measure Prefill Phase (TTFT) ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
      out = model(input_ids, **prefill_kwargs)
    end_event.record()
    torch.cuda.synchronize()

    ttft_ms = start_event.elapsed_time(end_event)
    prefill_latencies.append(ttft_ms)

    past_kv = out.past_key_values
    next_token = out.logits[:, -1:].argmax(dim=-1)
    del out

    # --- B. Measure Autoregressive Decoding Phase ---
    dec_start_event = torch.cuda.Event(enable_timing=True)
    dec_end_event = torch.cuda.Event(enable_timing=True)

    dec_start_event.record()
    with torch.no_grad():
      for _ in range(gen_tokens):
        step_out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        next_token = step_out.logits[:, -1:].argmax(dim=-1)
    dec_end_event.record()
    torch.cuda.synchronize()

    decode_ms = dec_start_event.elapsed_time(dec_end_event)
    decode_latencies.append(decode_ms)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_memories.append(peak_vram)

    del past_kv, step_out, next_token
    clear_cuda()

  avg_ttft_ms = float(np.mean(prefill_latencies))
  avg_prefill_tok_s = (
      (seq_len / (avg_ttft_ms / 1000.0)) if avg_ttft_ms > 0 else 0.0
  )
  avg_decode_ms = float(np.mean(decode_latencies))
  avg_decode_tok_s = (
      (gen_tokens / (avg_decode_ms / 1000.0)) if avg_decode_ms > 0 else 0.0
  )
  avg_peak_vram = float(np.mean(peak_memories))

  return {
      "context_len": seq_len,
      "ttft_ms": avg_ttft_ms,
      "prefill_tok_s": avg_prefill_tok_s,
      "decode_tok_s": avg_decode_tok_s,
      "peak_vram_gb": avg_peak_vram,
  }


def main():
  parser = argparse.ArgumentParser(
      description=(
          "BareTorch vs. Qwen3.5 Long-Context Inference Benchmark (Strict"
          " Apples-to-Apples)"
      )
  )
  parser.add_argument(
      "--baretorch_path",
      type=str,
      default="./checkpoint-90000",
      help="Local path to BareTorch 2.5B checkpoint folder",
  )
  parser.add_argument(
      "--qwen_model_id",
      type=str,
      default="Qwen/Qwen3.5-2B",
      help="Hugging Face model ID for Qwen3.5-2B",
  )
  parser.add_argument(
      "--context_lengths",
      type=int,
      nargs="+",
      default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
      help="List of context window lengths to benchmark",
  )
  parser.add_argument(
      "--gen_tokens",
      type=int,
      default=64,
      help="Number of tokens to decode autoregressively per trial",
  )
  parser.add_argument(
      "--trials",
      type=int,
      default=3,
      help="Number of measurement runs per context length",
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cuda:0",
      help="Target CUDA GPU device",
  )
  parser.add_argument(
      "--compile",
      action="store_true",
      default=True,
      help=(
          "Enable torch.compile optimization on CS-LRAD recurrent blocks"
          " (default: True)"
      ),
  )
  parser.add_argument(
      "--no_compile",
      action="store_false",
      dest="compile",
      help="Disable torch.compile and execute in eager PyTorch mode",
  )
  parser.add_argument(
      "--csv_path",
      type=str,
      default="benchmark_results_strict.csv",
      help="Path where structured benchmark metrics are exported as CSV",
  )
  args = parser.parse_args()

  device = torch.device(args.device)
  device_name = torch.cuda.get_device_name(device)

  print("=" * 85)
  print("🏎️  BARETORCH 2.5B vs. QWEN3.5-2B STRICT APPLES-TO-APPLES BENCHMARK")
  print(f"Device        : {device_name}")
  print(f"Contexts      : {args.context_lengths}")
  print(f"Decode Steps  : {args.gen_tokens} tokens")
  print(f"Trials        : {args.trials} per context length")
  print(f"torch.compile : {args.compile}")
  print(f"CSV Output    : {args.csv_path}")
  print("=" * 85)

  csv_rows = []
  baretorch_results = {}
  qwen_results = {}

  bt_label = (
      "BareTorch 2.5B (Compiled)" if args.compile else "BareTorch 2.5B (Eager)"
  )

  # --------------------------------------------------------------------------
  # 1. Benchmark BareTorch 2.5B
  # --------------------------------------------------------------------------
  if os.path.exists(args.baretorch_path):
    logger.info(f"Loading BareTorch 2.5B from '{args.baretorch_path}'...")
    bt_model = BareTorchForCausalLM.from_pretrained(
        args.baretorch_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    bt_model.eval()

    # Enforce last-logit-only projection
    enable_last_logit_only(bt_model)

    if args.compile:
      logger.info("⚡ Compiling CS-LRAD recurrent sub-modules...")
      compiled_blocks = 0
      for name, module in bt_model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "lrad" in cls_name or "lrad" in name.lower():
          module.forward = torch.compile(module.forward)
          compiled_blocks += 1
      logger.info(
          f"Successfully compiled {compiled_blocks} CS-LRAD module(s)."
      )

      # Global warmup across decoding steps to ensure JIT graphs are captured
      logger.info(
          "Running JIT compiler warm-up across initial decode steps..."
      )
      dummy_prefill = torch.randint(100, 5000, (1, 64), device=device)
      with torch.no_grad():
        init_out = bt_model(dummy_prefill, **get_forward_kwargs(bt_model))
        p_kv = init_out.past_key_values
        tok = init_out.logits[:, -1:].argmax(dim=-1)
        for _ in range(32):
          step_out = bt_model(tok, past_key_values=p_kv, use_cache=True)
          p_kv = step_out.past_key_values
          tok = step_out.logits[:, -1:].argmax(dim=-1)
        del dummy_prefill, init_out, p_kv, tok, step_out
      clear_cuda()
      logger.info("JIT warm-up complete.")

    logger.info("Starting BareTorch context evaluation loop...")
    for ctx in args.context_lengths:
      print(f"  --> Benchmarking {bt_label} @ {ctx:,} tokens...")
      synth_input = torch.randint(
          100, 10000, (1, ctx), dtype=torch.long, device=device
      )
      try:
        metrics = benchmark_model_at_context(
            bt_model,
            synth_input,
            gen_tokens=args.gen_tokens,
            trials=args.trials,
        )
        baretorch_results[ctx] = metrics
        csv_rows.append({
            "model": bt_label,
            "compiled": args.compile,
            "context_len": ctx,
            "ttft_ms": round(metrics["ttft_ms"], 2),
            "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
            "decode_tok_s": round(metrics["decode_tok_s"], 2),
            "peak_vram_gb": round(metrics["peak_vram_gb"], 2),
            "status": "SUCCESS",
        })
      except torch.cuda.OutOfMemoryError:
        print(f"  ❌ {bt_label} Out-Of-Memory @ {ctx:,} tokens!")
        baretorch_results[ctx] = {"error": "OOM"}
        csv_rows.append({
            "model": bt_label,
            "compiled": args.compile,
            "context_len": ctx,
            "ttft_ms": None,
            "prefill_tok_s": None,
            "decode_tok_s": None,
            "peak_vram_gb": None,
            "status": "OOM",
        })
        clear_cuda()
      del synth_input

    del bt_model
    clear_cuda()
    logger.info("Unloaded BareTorch from GPU memory.\n")
  else:
    logger.warning(
        f"BareTorch checkpoint path '{args.baretorch_path}' not found."
        " Skipping."
    )

  # --------------------------------------------------------------------------
  # 2. Benchmark Qwen3.5-2B
  # --------------------------------------------------------------------------
  try:
    logger.info(f"Loading Qwen3.5-2B from '{args.qwen_model_id}'...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    qwen_model.eval()

    # Enforce identical last-logit-only projection on Qwen
    enable_last_logit_only(qwen_model)
    logger.info("Enabled last-logit-only prefill on Qwen3.5-2B.")

    logger.info("Starting Qwen3.5-2B context evaluation loop...")
    for ctx in args.context_lengths:
      print(f"  --> Benchmarking Qwen3.5-2B @ {ctx:,} tokens...")
      synth_input = torch.randint(
          100, 10000, (1, ctx), dtype=torch.long, device=device
      )
      try:
        metrics = benchmark_model_at_context(
            qwen_model,
            synth_input,
            gen_tokens=args.gen_tokens,
            trials=args.trials,
        )
        qwen_results[ctx] = metrics
        csv_rows.append({
            "model": "Qwen3.5-2B",
            "compiled": False,
            "context_len": ctx,
            "ttft_ms": round(metrics["ttft_ms"], 2),
            "prefill_tok_s": round(metrics["prefill_tok_s"], 2),
            "decode_tok_s": round(metrics["decode_tok_s"], 2),
            "peak_vram_gb": round(metrics["peak_vram_gb"], 2),
            "status": "SUCCESS",
        })
      except torch.cuda.OutOfMemoryError:
        print(f"  ❌ Qwen3.5-2B Out-Of-Memory @ {ctx:,} tokens!")
        qwen_results[ctx] = {"error": "OOM"}
        csv_rows.append({
            "model": "Qwen3.5-2B",
            "compiled": False,
            "context_len": ctx,
            "ttft_ms": None,
            "prefill_tok_s": None,
            "decode_tok_s": None,
            "peak_vram_gb": None,
            "status": "OOM",
        })
        clear_cuda()
      del synth_input

    del qwen_model
    clear_cuda()
    logger.info("Unloaded Qwen3.5-2B from GPU memory.\n")
  except Exception as e:
    logger.error(
        f"Failed to load or benchmark Qwen model '{args.qwen_model_id}':"
        f" {e}"
    )

  # --------------------------------------------------------------------------
  # 3. Export Metrics to CSV
  # --------------------------------------------------------------------------
  fieldnames = [
      "model",
      "compiled",
      "context_len",
      "ttft_ms",
      "prefill_tok_s",
      "decode_tok_s",
      "peak_vram_gb",
      "status",
  ]
  with open(args.csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
      writer.writerow(row)
  logger.info(f"✅ Benchmark metrics saved to '{args.csv_path}'")

  # --------------------------------------------------------------------------
  # 4. Print Formatted Markdown Summary Table
  # --------------------------------------------------------------------------
  print("\n" + "=" * 95)
  print(f"📊 COMPARATIVE BENCHMARK REPORT ({device_name})")
  print("=" * 95)
  header = (
      "| Context | Model | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |"
      " Peak VRAM (GB) |"
  )
  sep = "| :---: | :--- | :---: | :---: | :---: | :---: |"
  print(header)
  print(sep)

  for ctx in args.context_lengths:
    if ctx in baretorch_results:
      bt = baretorch_results[ctx]
      if "error" in bt:
        print(
            f"| {ctx:,} | **{bt_label}** | 💥 OOM | 💥 OOM | 💥 OOM | 💥"
            " OOM |"
        )
      else:
        print(
            f"| {ctx:,} | **{bt_label}** | {bt['ttft_ms']:.1f} ms |"
            f" {bt['prefill_tok_s']:,.1f} tok/s |"
            f" {bt['decode_tok_s']:.2f} tok/s | {bt['peak_vram_gb']:.2f}"
            " GB |"
        )

    if ctx in qwen_results:
      qw = qwen_results[ctx]
      if "error" in qw:
        print(
            f"| {ctx:,} | Qwen3.5-2B | 💥 OOM | 💥 OOM | 💥 OOM | 💥 OOM |"
        )
      else:
        print(
            f"| {ctx:,} | Qwen3.5-2B | {qw['ttft_ms']:.1f} ms |"
            f" {qw['prefill_tok_s']:,.1f} tok/s |"
            f" {qw['decode_tok_s']:.2f} tok/s | {qw['peak_vram_gb']:.2f}"
            " GB |"
        )
    print("| --- | --- | --- | --- | --- | --- |")

  print("=" * 95)


if __name__ == "__main__":
  main()