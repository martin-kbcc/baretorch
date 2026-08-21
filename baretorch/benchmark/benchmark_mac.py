# baretorch/export/benchmark_mac.py
import time
import gc
import argparse
import traceback
import psutil
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

try:
    from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False


def apply_quantization(model: nn.Module, quant_type: str, device: torch.device) -> nn.Module:
    """Applies torchao native weight-only quantization or precision casting."""
    if quant_type in ["int4", "int8"]:
        if not HAS_TORCHAO:
            print("  ⚠️ 'torchao' is not installed (`pip install torchao`). Falling back to FP16...")
            model = model.to(dtype=torch.float16)
        else:
            print(f"  ⚡ Applying torchao native {quant_type.upper()} weight-only quantization...")
            if quant_type == "int4":
                quantize_(model, int4_weight_only(group_size=32))
            elif quant_type == "int8":
                quantize_(model, int8_weight_only())
    elif quant_type == "fp16":
        model = model.to(dtype=torch.float16)
    elif quant_type == "fp32":
        model = model.to(dtype=torch.float32)

    return model.to(device).eval()


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


def benchmark_inference_standardized(
    model: nn.Module,
    prompt_len: int,
    gen_len: int,
    device: torch.device,
    vocab_size: int
) -> dict:
    """
    Standardized benchmark function matching CUDA profiler.py:
    1. Un-timed warmup pass
    2. Prefill phase (TTFT ms for prompt_len)
    3. Decode phase (steady-state tok/s over gen_len=32)
    """
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    curr_token = torch.randint(0, vocab_size, (1, 1), device=device)

    # 1. Warmup Pass
    with torch.no_grad():
        w_out = model(prompt, use_cache=True)
        w_kv = getattr(w_out, "past_key_values", None)
        for _ in range(3):
            w_out = model(curr_token, past_key_values=w_kv, use_cache=True)
            w_kv = getattr(w_out, "past_key_values", None)

    if device.type == "mps":
        torch.mps.synchronize()

    # 2. Prefill Phase (TTFT)
    ttft_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(prompt, use_cache=True)
    if device.type == "mps":
        torch.mps.synchronize()
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    past_key_values = getattr(outputs, "past_key_values", None)
    curr_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    # 3. Decode Phase (gen_len tokens)
    gen_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len):
            outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = getattr(outputs, "past_key_values", None)
            if hasattr(outputs, "logits"):
                curr_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    if device.type == "mps":
        torch.mps.synchronize()
    decode_sec = max(time.perf_counter() - gen_start, 1e-5)

    peak_vram_mb = get_vram_mb()
    tokens_per_sec = gen_len / decode_sec

    return {
        "prompt_len": prompt_len,
        "ttft_ms": round(ttft_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="BareTorch Standardized Apple Silicon MPS Context Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, choices=["mps", "cpu"], default="mps")
    parser.add_argument("--quant_type", type=str, choices=["int4", "int8", "fp16", "fp32"], default="int4")
    parser.add_argument("--prompt_lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--gen_len", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu")

    print("==================================================================================================")
    print(f"🍎 BARETORCH STANDARDIZED CONTEXT SCALING BENCHMARK (MPS | Device: {device} | Quant: {args.quant_type.upper()})")
    print("==================================================================================================")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
    except Exception:
        vocab_size = 50257

    hf_results = {}
    bt_results = {}

    # 1. Benchmark Hugging Face Baseline
    clear_memory()
    print(f"\n📦 Loading Hugging Face Baseline: '{args.hf_model_id}'...")
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6
        hf_vocab_size = getattr(hf_model.config, "vocab_size", vocab_size)

        hf_model = apply_quantization(hf_model, quant_type=args.quant_type, device=device)

        for ctx in args.prompt_lens:
            print(f"  ├─ Benchmarking Baseline ({args.quant_type.upper()}) @ Context: {ctx:<5} tokens...", end="", flush=True)
            try:
                hf_results[ctx] = benchmark_inference_standardized(hf_model, ctx, args.gen_len, device, hf_vocab_size)
                print(f" ✅ (TTFT: {hf_results[ctx]['ttft_ms']} ms | Decode: {hf_results[ctx]['tokens_per_sec']} tok/s | VRAM: {hf_results[ctx]['peak_vram_mb']} MB)")
            except Exception as e_step:
                print(f" ❌ Failed ({type(e_step).__name__})")
                traceback.print_exc()
                hf_results[ctx] = {"ttft_ms": "OOM/Fail", "tokens_per_sec": "OOM/Fail", "peak_vram_mb": "N/A"}

        del hf_model
        clear_memory()
    except Exception as e:
        print(f"⚠️ Failed to initialize Hugging Face baseline ({e})")
        hf_params_m = 1235.81

    # 2. Benchmark BareTorch Matched Hybrid
    clear_memory()
    print(f"\n⚙️ Instantiating BareTorch Matched Blueprint (~{hf_params_m:.2f}M params)...")
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
    bt_model = BareTorchForCausalLM(bt_config).to(dtype=torch.float32)
    bt_model = apply_quantization(bt_model, quant_type=args.quant_type, device=device)

    for ctx in args.prompt_lens:
        print(f"  ├─ Benchmarking BareTorch ({args.quant_type.upper()}) @ Context: {ctx:<5} tokens...", end="", flush=True)
        try:
            bt_results[ctx] = benchmark_inference_standardized(bt_model, ctx, args.gen_len, device, vocab_size)
            print(f" ✅ (TTFT: {bt_results[ctx]['ttft_ms']} ms | Decode: {bt_results[ctx]['tokens_per_sec']} tok/s | VRAM: {bt_results[ctx]['peak_vram_mb']} MB)")
        except Exception as e_step:
            print(f" ❌ Failed ({type(e_step).__name__})")
            traceback.print_exc()
            bt_results[ctx] = {"ttft_ms": "Fail", "tokens_per_sec": "Fail", "peak_vram_mb": "N/A"}

    del bt_model
    clear_memory()

    # 3. Comparative Report
    print("\n" + "=" * 130)
    print(f"📊 STANDARDIZED BARETORCH ({args.quant_type.upper()}) vs. LLAMA 3.2 1B CONTEXT REPORT (APPLE SILICON MPS)")
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