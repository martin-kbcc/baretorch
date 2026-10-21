# benchmark_context_tpu.py
import csv
import gc
import time
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM

BARETORCH_ID = "model-rampage/BareTorch-2.5B-Base"
QWEN_ID = "Qwen/Qwen3.5-2B"
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
GEN_TOKENS = 32
TRIALS = 3


def sync_tpu():
    """Flushes pending lazy XLA graph operations and waits for TPU hardware completion."""
    xm.mark_step()
    xm.wait_device_ops()


def get_tpu_memory_gb(device) -> float:
    try:
        info = xm.get_memory_info(device)
        used_bytes = info.get("bytes_used", info.get("kb_used", 0) * 1024)
        return float(used_bytes) / (1024**3)
    except Exception:
        return 0.0


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
    gen_tokens: int = 32,
    trials: int = 3,
):
    print(f"\n📥 Loading {label} ('{model_id}') onto TPU ({device})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    enable_last_logit_only(model)
    sync_tpu()

    results = {}
    for ctx in context_lengths:
        print(f"  --> Benchmarking {label} @ {ctx:,} tokens (Compiling XLA graph on warmup)...")
        inp = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)
        sync_tpu()

        try:
            # 1. Untimed XLA Compilation Warmup (Compiles exact prefill + decode step HLO graphs)
            with torch.no_grad():
                t_compile_start = time.perf_counter()
                out = model(inp, use_cache=True)
                p_kv = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
                sync_tpu()

                for _ in range(gen_tokens):
                    step = model(tok, past_key_values=p_kv, use_cache=True)
                    p_kv = step.past_key_values
                    tok = step.logits[:, -1:].argmax(dim=-1)
                    sync_tpu()
                compile_s = time.perf_counter() - t_compile_start
                del out, p_kv, tok, step
                gc.collect()
                sync_tpu()

            # 2. Timed Hardware Execution Trials (XLA cached)
            prefill_latencies = []
            decode_latencies = []
            peak_memories = []

            for _ in range(trials):
                gc.collect()
                sync_tpu()

                # --- A. Measure Prefill (TTFT) ---
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(inp, use_cache=True)
                    p_kv = out.past_key_values
                    tok = out.logits[:, -1:].argmax(dim=-1)
                sync_tpu()
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                prefill_latencies.append(ttft_ms)
                prefill_mem = get_tpu_memory_gb(device)
                del out

                # --- B. Measure Autoregressive Decode ---
                t1 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(gen_tokens):
                        step = model(tok, past_key_values=p_kv, use_cache=True)
                        p_kv = step.past_key_values
                        tok = step.logits[:, -1:].argmax(dim=-1)
                        sync_tpu()
                dec_ms = (time.perf_counter() - t1) * 1000.0
                decode_latencies.append(dec_ms)
                decode_mem = get_tpu_memory_gb(device)
                peak_memories.append(max(prefill_mem, decode_mem))

                del p_kv, tok, step
                gc.collect()
                sync_tpu()

            avg_ttft = float(np.mean(prefill_latencies))
            avg_prefill_tps = ctx / (avg_ttft / 1000.0)
            avg_dec_ms = float(np.mean(decode_latencies))
            avg_dec_tps = gen_tokens / (avg_dec_ms / 1000.0)
            avg_mem = float(np.mean(peak_memories))

            print(
                f"      TTFT: {avg_ttft:,.1f} ms | Prefill: {avg_prefill_tps:,.1f} tok/s | "
                f"Decode: {avg_dec_tps:.2f} tok/s | HBM: {avg_mem:.2f} GB (XLA warmup: {compile_s:.1f}s)"
            )
            results[ctx] = {
                "context_len": ctx,
                "ttft_ms": avg_ttft,
                "prefill_tok_s": avg_prefill_tps,
                "decode_tok_s": avg_dec_tps,
                "peak_hbm_gb": avg_mem,
                "xla_compile_s": compile_s,
                "status": "SUCCESS",
            }
        except Exception as e:
            print(f"  ❌ {label} failed @ {ctx:,} tokens on TPU: {e}")
            results[ctx] = {"context_len": ctx, "status": f"FAIL: {e}"}
        finally:
            del inp
            gc.collect()
            sync_tpu()

    del model
    gc.collect()
    sync_tpu()
    return results


def main():
    device = xm.xla_device()
    print("=" * 95)
    print(f"⚡ GOOGLE TPU XLA BENCHMARK: BareTorch-2.5B vs. Qwen3.5-2B ({device})")
    print("=" * 95)

    bt_res = benchmark_model_on_tpu(
        BARETORCH_ID, "BareTorch-2.5B (TPU XLA)", device, CONTEXT_LENGTHS, GEN_TOKENS, TRIALS
    )
    qw_res = benchmark_model_on_tpu(
        QWEN_ID, "Qwen3.5-2B (TPU XLA)", device, CONTEXT_LENGTHS, GEN_TOKENS, TRIALS
    )

    print("\n" + "=" * 105)
    print("📊 GOOGLE COLAB TPU XLA BENCHMARK REPORT (BF16)")
    print("=" * 105)
    print("| Context | Model (Backend) | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | XLA Compile (s) | Peak HBM |")
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    csv_rows = []
    for ctx in CONTEXT_LENGTHS:
        for label, res_dict in [("BareTorch-2.5B (TPU XLA)", bt_res), ("Qwen3.5-2B (TPU XLA)", qw_res)]:
            if ctx in res_dict:
                r = res_dict[ctx]
                if r["status"] == "SUCCESS":
                    bold = "**" if "BareTorch" in label else ""
                    print(
                        f"| {bold}{ctx:,}{bold} | {bold}{label}{bold} | "
                        f"{bold}{r['ttft_ms']:,.1f} ms{bold} | {bold}{r['prefill_tok_s']:,.1f} tok/s{bold} | "
                        f"{bold}{r['decode_tok_s']:.2f} tok/s{bold} | {r['xla_compile_s']:.1f}s | {r['peak_hbm_gb']:.2f} GB |"
                    )
                    csv_rows.append({"model": label, **r})
                else:
                    print(f"| {ctx:,} | {label} | 💥 {r['status']} | — | — | — | — |")
        print("| --- | --- | --- | --- | --- | --- | --- |")

    with open("benchmark_tpu_colab.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "context_len", "ttft_ms", "prefill_tok_s", "decode_tok_s", "peak_hbm_gb", "xla_compile_s", "status"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()