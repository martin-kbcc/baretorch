# /content/baretorch/benchmark_context_tpu.py
import os

# Enable TorchTPU's DeferAndFuse mode for automatic op fusion and lower HBM usage
os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"

import argparse
import csv
import gc
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

BARETORCH_ID = "model-rampage/BareTorch-2.5B-Base"
QWEN_ID = "Qwen/Qwen3.5-2B"


def sync_tpu():
    """Synchronizes the native TorchTPU ('tpu') backend."""
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
        torch.accelerator.synchronize()
    elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
        torch.tpu.synchronize()


def get_tpu_memory_gb() -> float:
    """Queries allocated TPU HBM memory in GB if exposed by the runtime."""
    try:
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "max_memory_allocated"):
            return float(torch.tpu.max_memory_allocated()) / (1024**3)
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "memory_allocated"):
            return float(torch.tpu.memory_allocated()) / (1024**3)
    except Exception:
        pass
    return 0.0


def reset_tpu_memory():
    gc.collect()
    try:
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "empty_cache"):
            torch.tpu.empty_cache()
        if hasattr(torch, "tpu") and hasattr(torch.tpu, "reset_peak_memory_stats"):
            torch.tpu.reset_peak_memory_stats()
    except Exception:
        pass
    sync_tpu()


def enable_last_logit_only(model):
    """Ensures both models only project the final token through the 248k-vocab lm_head during prefill."""
    head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else getattr(model, "lm_head", None)
    )
    if head is not None and not getattr(head, "_last_logit_patched", False):
        orig_forward = head.forward

        def efficient_head_forward(hidden_states, *args, **kwargs):
            if hidden_states.dim() == 3 and hidden_states.size(1) > 1:
                hidden_states = hidden_states[:, -1:, :]
            return orig_forward(hidden_states, *args, **kwargs)

        head.forward = efficient_head_forward
        head._last_logit_patched = True


def benchmark_model_on_tpu(
    model_id: str,
    label: str,
    device: torch.device,
    context_lengths: list[int],
    gen_tokens: int = 64,
    trials: int = 3,
    use_compile: bool = False,
):
    print(f"\n📥 Loading {label} ('{model_id}') onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    enable_last_logit_only(model)

    if use_compile:
        print(f"  🔥 Compiling {label} with torch.compile(backend='tpu')...")
        if "baretorch" in model_id.lower():
            for module in model.modules():
                if "lrad" in module.__class__.__name__.lower():
                    module.forward = torch.compile(module.forward, backend="tpu")
        else:
            model = torch.compile(model, backend="tpu")

    sync_tpu()

    results = {}
    for ctx in context_lengths:
        print(f"  --> Benchmarking {label} @ {ctx:,} tokens...")
        inp = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)
        sync_tpu()

        try:
            # 1. Full Untimed Warmup: Compiles prefill + ALL `gen_tokens` decode step shapes
            # so zero XLA graph compilations occur during the timed measurement trials.
            t_warmup_start = time.perf_counter()
            with torch.no_grad():
                out = model(inp, use_cache=True)
                p_kv = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
                sync_tpu()

                for _ in range(gen_tokens):
                    step = model(tok, past_key_values=p_kv, use_cache=True)
                    p_kv = step.past_key_values
                    tok = step.logits[:, -1:].argmax(dim=-1)
                    sync_tpu()
            warmup_s = time.perf_counter() - t_warmup_start
            del out, p_kv, tok, step
            reset_tpu_memory()

            # 2. Timed Hardware Execution Trials (100% XLA Cache Hit)
            prefill_latencies = []
            decode_latencies = []
            peak_memories = []

            for _ in range(trials):
                reset_tpu_memory()

                # --- A. Measure Prefill Phase (TTFT) ---
                sync_tpu()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(inp, use_cache=True)
                    p_kv = out.past_key_values
                    tok = out.logits[:, -1:].argmax(dim=-1)
                sync_tpu()
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                prefill_latencies.append(ttft_ms)
                prefill_mem = get_tpu_memory_gb()
                del out

                # --- B. Measure Autoregressive Decode Phase (64 tokens) ---
                sync_tpu()
                t1 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(gen_tokens):
                        step = model(tok, past_key_values=p_kv, use_cache=True)
                        p_kv = step.past_key_values
                        tok = step.logits[:, -1:].argmax(dim=-1)
                sync_tpu()
                dec_ms = (time.perf_counter() - t1) * 1000.0
                decode_latencies.append(dec_ms)
                decode_mem = get_tpu_memory_gb()
                peak_memories.append(max(prefill_mem, decode_mem))

                del p_kv, tok, step
                reset_tpu_memory()

            avg_ttft = float(np.mean(prefill_latencies))
            avg_prefill_tps = ctx / (avg_ttft / 1000.0)
            avg_dec_ms = float(np.mean(decode_latencies))
            avg_dec_tps = gen_tokens / (avg_dec_ms / 1000.0)
            avg_mem = float(np.mean(peak_memories))

            print(
                f"      TTFT: {avg_ttft:,.1f} ms | Prefill: {avg_prefill_tps:,.1f} tok/s | "
                f"Decode: {avg_dec_tps:.2f} tok/s | Warmup/Compile: {warmup_s:.1f}s"
            )
            results[ctx] = {
                "context_len": ctx,
                "ttft_ms": round(avg_ttft, 2),
                "prefill_tok_s": round(avg_prefill_tps, 2),
                "decode_tok_s": round(avg_dec_tps, 2),
                "peak_hbm_gb": round(avg_mem, 2),
                "warmup_compile_s": round(warmup_s, 2),
                "status": "SUCCESS",
            }
        except Exception as e:
            print(f"  ❌ {label} failed @ {ctx:,} tokens on TPU: {e}")
            results[ctx] = {"context_len": ctx, "status": f"FAIL: {e}"}
        finally:
            del inp
            reset_tpu_memory()

    del model
    reset_tpu_memory()
    return results


def main():
    parser = argparse.ArgumentParser(description="BareTorch vs Qwen3.5 Native TorchTPU Benchmark")
    parser.add_argument("--baretorch_id", type=str, default=BARETORCH_ID)
    parser.add_argument("--qwen_id", type=str, default=QWEN_ID)
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--gen_tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--csv_path", type=str, default="benchmark_tpu_colab.csv")
    args = parser.parse_args()

    device = torch.device("tpu")
    print("=" * 95)
    print(f"⚡ GOOGLE TORCHTPU BENCHMARK: BareTorch-2.5B vs. Qwen3.5-2B (device='{device}')")
    print(f"Contexts     : {args.context_lengths}")
    print(f"Decode Steps : {args.gen_tokens} tokens")
    print(f"Trials       : {args.trials} per context length")
    print("=" * 95)

    bt_res = benchmark_model_on_tpu(
        args.baretorch_id,
        "BareTorch-2.5B (TorchTPU)",
        device,
        args.context_lengths,
        args.gen_tokens,
        args.trials,
        use_compile=args.compile,
    )
    qw_res = benchmark_model_on_tpu(
        args.qwen_id,
        "Qwen3.5-2B (TorchTPU)",
        device,
        args.context_lengths,
        args.gen_tokens,
        args.trials,
        use_compile=args.compile,
    )

    print("\n" + "=" * 105)
    print("📊 GOOGLE COLAB TORCHTPU BENCHMARK REPORT (BF16)")
    print("=" * 105)
    print("| Context | Model (Backend) | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Warmup/Compile (s) |")
    print("| :---: | :--- | :---: | :---: | :---: | :---: |")

    csv_rows = []
    for ctx in args.context_lengths:
        for label, res_dict in [
            ("BareTorch-2.5B (TorchTPU)", bt_res),
            ("Qwen3.5-2B (TorchTPU)", qw_res),
        ]:
            if ctx in res_dict:
                r = res_dict[ctx]
                if r["status"] == "SUCCESS":
                    bold = "**" if "BareTorch" in label else ""
                    print(
                        f"| {bold}{ctx:,}{bold} | {bold}{label}{bold} | "
                        f"{bold}{r['ttft_ms']:,.1f} ms{bold} | {bold}{r['prefill_tok_s']:,.1f} tok/s{bold} | "
                        f"{bold}{r['decode_tok_s']:.2f} tok/s{bold} | {r['warmup_compile_s']:.1f}s |"
                    )
                    csv_rows.append({"model": label, **r})
                else:
                    print(f"| {ctx:,} | {label} | 💥 {r['status']} | — | — | — |")
        print("| --- | --- | --- | --- | --- | --- |")

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "context_len",
                "ttft_ms",
                "prefill_tok_s",
                "decode_tok_s",
                "peak_hbm_gb",
                "warmup_compile_s",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()