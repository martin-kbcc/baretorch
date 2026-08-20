# baretorch/export/benchmark_mac.py
import time
import gc
import argparse
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM


def clear_memory():
    """Flushes Python garbage collection and MPS VRAM allocation."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_vram_mb() -> float:
    """
    Returns actual Apple Silicon GPU VRAM allocated on MPS in MB, 
    falling back to host CPU RSS if MPS is unavailable.
    """
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


def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_tokens: int,
    device: torch.device
) -> dict:
    """Runs a single generation pass for a specific token count with accurate VRAM profiling."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # Warmup pass to allocate MPS GPU kernels
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    if device.type == "mps":
        torch.mps.synchronize()

    # Time-To-First-Token (TTFT) / Prefill Latency
    ttft_start = time.perf_counter()
    with torch.no_grad():
        _ = model(inputs["input_ids"])
    if device.type == "mps":
        torch.mps.synchronize()
    ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

    # Total Decode Generation Pass
    gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=gen_tokens, 
            do_sample=False, 
            use_cache=True
        )
    if device.type == "mps":
        torch.mps.synchronize()
    gen_total_sec = time.perf_counter() - gen_start

    # Synchronize to capture peak VRAM after total context expansion
    peak_vram_mb = get_vram_mb()

    actual_gen_tokens = output_ids.shape[1] - input_length
    decode_sec = max(gen_total_sec - (ttft_ms / 1000.0), 1e-5)
    tokens_per_sec = actual_gen_tokens / decode_sec

    return {
        "gen_tokens": actual_gen_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "total_time_sec": round(gen_total_sec, 3),
        "throughput_tok_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="BareTorch Context Length VRAM & Latency Benchmarker")
    parser.add_argument("--hf_model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", type=str, default="The future of edge artificial intelligence relies on efficient local architecture because")
    parser.add_argument("--device", type=str, choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()

    token_scaling_steps = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    device = torch.device(args.device if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu")

    print("==================================================================================================")
    print(f"🍎 BARETORCH CONTEXT SCALING BENCHMARK (APPLE SILICON MPS | Device: {device})")
    print("==================================================================================================")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

    hf_results = {}
    bt_results = {}

    # --------------------------------------------------------------------------
    # 2. Benchmark Hugging Face Baseline Model
    # --------------------------------------------------------------------------
    clear_memory()
    print(f"\n📦 Loading Hugging Face Baseline: '{args.hf_model_id}'...")
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6

        for steps in token_scaling_steps:
            print(f"  ├─ Benchmarking Baseline generating {steps:<4} tokens...", end="", flush=True)
            try:
                hf_results[steps] = benchmark_generation(hf_model, tokenizer, args.prompt, steps, device)
                print(f" ✅ ({hf_results[steps]['throughput_tok_sec']} tok/s | VRAM: {hf_results[steps]['peak_vram_mb']} MB)")
            except Exception as e_step:
                print(f" ❌ Failed ({type(e_step).__name__})")
                hf_results[steps] = {"throughput_tok_sec": "OOM/Fail", "peak_vram_mb": "N/A", "ttft_ms": "N/A"}

        del hf_model
        clear_memory()
    except Exception as e:
        print(f"⚠️ Failed to initialize Hugging Face baseline ({e})")
        hf_params_m = 1235.81

    # --------------------------------------------------------------------------
    # 3. Benchmark BareTorch Matched Model
    # --------------------------------------------------------------------------
    clear_memory()
    print(f"\n⚙️ Instantiating BareTorch Matched Blueprint (~{hf_params_m:.2f}M params)...")
    bt_config = BareTorchConfig(
        vocab_size=tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257,
        d_model=1888,
        num_heads=16,
        num_layers=14,
        chunk_size=32,
        rank=8,
        max_seq_len=16384,
        layer_types=["cs_lrad", "cs_lrad", "cs_lrad", "transformer"] * 3 + ["cs_lrad", "cs_lrad"]
    )
    bt_model = BareTorchForCausalLM(bt_config).to(dtype=torch.float32).to(device).eval()
    bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

    for steps in token_scaling_steps:
        print(f"  ├─ Benchmarking BareTorch generating {steps:<4} tokens...", end="", flush=True)
        try:
            bt_results[steps] = benchmark_generation(bt_model, tokenizer, args.prompt, steps, device)
            print(f" ✅ ({bt_results[steps]['throughput_tok_sec']} tok/s | VRAM: {bt_results[steps]['peak_vram_mb']} MB)")
        except Exception as e_step:
            print(f" ❌ Failed ({type(e_step).__name__})")
            bt_results[steps] = {"throughput_tok_sec": "Fail", "peak_vram_mb": "N/A", "ttft_ms": "N/A"}

    del bt_model
    clear_memory()

    # --------------------------------------------------------------------------
    # 4. Comparative Scaling Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("📊 BARETORCH vs. LLAMA 3.2 1B CONTEXT SCALING REPORT (APPLE SILICON MPS)")
    print("=" * 115)
    print(f"{'Gen Length':<12} | {'BareTorch (tok/s)':<20} | {'Llama 3.2 (tok/s)':<20} | {'Throughput Δ':<16} | {'BareTorch VRAM':<15} | {'Llama VRAM':<15}")
    print("-" * 115)

    for steps in token_scaling_steps:
        bt = bt_results.get(steps, {})
        hf = hf_results.get(steps, {})

        bt_spd = bt.get("throughput_tok_sec", "N/A")
        hf_spd = hf.get("throughput_tok_sec", "N/A")

        bt_vram = bt.get("peak_vram_mb", "N/A")
        hf_vram = hf.get("peak_vram_mb", "N/A")

        if isinstance(bt_spd, (int, float)) and isinstance(hf_spd, (int, float)) and hf_spd > 0:
            diff_pct = ((bt_spd - hf_spd) / hf_spd) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"

        bt_spd_str = f"{bt_spd} tok/s" if isinstance(bt_spd, (int, float)) else str(bt_spd)
        hf_spd_str = f"{hf_spd} tok/s" if isinstance(hf_spd, (int, float)) else str(hf_spd)
        bt_vram_str = f"{bt_vram} MB" if isinstance(bt_vram, (int, float)) else str(bt_vram)
        hf_vram_str = f"{hf_vram} MB" if isinstance(hf_vram, (int, float)) else str(hf_vram)

        print(f"{steps:<12} | {bt_spd_str:<20} | {hf_spd_str:<20} | {diff_str:<16} | {bt_vram_str:<15} | {hf_vram_str:<15}")

    print("=" * 115)


if __name__ == "__main__":
    main()