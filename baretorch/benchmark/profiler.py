# baretorch/benchmark/profiler.py
import gc
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List


def clear_gpu_memory(device: str = "cuda"):
    """
    Flushes Python garbage collection and CUDA cache, 
    resetting VRAM allocation tracking between benchmark runs.
    """
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def extract_kv_cache_config(model: nn.Module) -> Dict[str, Any]:
    """
    Safely extracts model layer structure and attention parameters 
    across BareTorch hybrids and arbitrary Hugging Face architectures (MHA/GQA/MQA).
    """
    # Handle torch.compile WrappedModules
    raw_model = getattr(model, "_orig_mod", model)
    cfg = getattr(raw_model, "config", None)
    if cfg is None:
        return {"is_baretorch": False, "num_layers": 0, "num_heads": 0, "num_kv_heads": 0, "head_dim": 0}

    cfg_dict = {}
    if hasattr(cfg, "to_dict"):
        try:
            cfg_dict = cfg.to_dict()
        except Exception:
            pass

    # 1. BareTorch Hybrid Architecture (Explicitly check for "cs_lrad" in layer_types)
    layer_types = cfg_dict.get("layer_types", getattr(cfg, "layer_types", None))
    if layer_types and isinstance(layer_types, list) and len(layer_types) > 0 and "cs_lrad" in layer_types:
        num_heads = cfg_dict.get("num_heads", getattr(cfg, "num_heads", 16))
        num_kv_heads = cfg_dict.get("num_kv_heads", getattr(cfg, "num_kv_heads", 4))
        d_model = cfg_dict.get("d_model", getattr(cfg, "d_model", 1536))
        head_dim = d_model // num_heads if num_heads > 0 else 96
        rank = cfg_dict.get("rank", getattr(cfg, "rank", 8))
        lrad_count = layer_types.count("cs_lrad")
        trans_count = layer_types.count("transformer")
        return {
            "is_baretorch": True,
            "lrad_count": lrad_count,
            "trans_count": trans_count,
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "rank": int(rank)
        }

    # 2. Standard Hugging Face CausalLM (Qwen, DeepSeek, Gemma, Llama, SmolLM, etc.)
    num_layers = (
        cfg_dict.get("num_hidden_layers") or 
        cfg_dict.get("n_layer") or 
        cfg_dict.get("num_layers") or 
        getattr(cfg, "num_hidden_layers", None) or 
        getattr(cfg, "n_layer", None) or 32
    )
    
    num_heads = (
        cfg_dict.get("num_attention_heads") or 
        cfg_dict.get("n_head") or 
        cfg_dict.get("num_heads") or 
        getattr(cfg, "num_attention_heads", None) or 32
    )
    
    num_kv_heads = (
        cfg_dict.get("num_key_value_heads") or 
        cfg_dict.get("num_kv_heads") or 
        getattr(cfg, "num_key_value_heads", None) or 
        num_heads
    )
    
    hidden_size = (
        cfg_dict.get("hidden_size") or 
        cfg_dict.get("n_embd") or 
        cfg_dict.get("d_model") or 2048
    )
    
    head_dim = (
        cfg_dict.get("head_dim") or 
        getattr(cfg, "head_dim", None) or 
        (hidden_size // num_heads if num_heads > 0 else 64)
    )

    return {
        "is_baretorch": False,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim)
    }


class LatencyProfiler:
    """
    Measures Prefill (TTFT) and Autoregressive Decoding Latency for both 
    BareTorch hybrid models and standard Hugging Face CausalLM models.
    Catches CUDA OOM exceptions gracefully without halting the benchmark runner.
    """
    @staticmethod
    def profile_inference(
        model: nn.Module,
        prompt_len: int = 2048,
        gen_len: int = 32,
        device: str = "cuda",
        vocab_size: int = 50257
    ) -> Dict[str, Any]:
        model.eval()
        
        raw_model = getattr(model, "_orig_mod", model)
        cfg = getattr(raw_model, "config", None)
        if cfg is not None:
            cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", vocab_size))

        try:
            prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
            curr_token = torch.randint(0, vocab_size, (1, 1), device=device)

            # ------------------------------------------------------------------
            # 0. Multi-Step JIT Warmup Pass (Forces Triton/Inductor kernel compilation)
            # ------------------------------------------------------------------
            with torch.no_grad():
                w_out = model(prompt, use_cache=True)
                w_kv = getattr(w_out, "past_key_values", None)
                for _ in range(3):
                    w_out = model(curr_token, past_key_values=w_kv, use_cache=True)
                    w_kv = getattr(w_out, "past_key_values", None)

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # ------------------------------------------------------------------
            # 1. Prefill Phase (Time To First Token / TTFT)
            # ------------------------------------------------------------------
            start_prefill = time.perf_counter()
            with torch.no_grad():
                outputs = model(prompt, use_cache=True)
            if device == "cuda":
                torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - start_prefill) * 1000.0

            past_key_values = getattr(outputs, "past_key_values", None)

            # ------------------------------------------------------------------
            # 2. Decode Phase (Autoregressive Token Generation)
            # ------------------------------------------------------------------
            if device == "cuda":
                torch.cuda.synchronize()

            start_decode = time.perf_counter()
            with torch.no_grad():
                for _ in range(gen_len):
                    outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = getattr(outputs, "past_key_values", None)
                    if hasattr(outputs, "logits"):
                        curr_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            if device == "cuda":
                torch.cuda.synchronize()
            decode_time = time.perf_counter() - start_decode
            tokens_per_sec = gen_len / decode_time if decode_time > 0 else 0.0

            return {
                "ttft_ms": round(ttft_ms, 2),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "total_gen_time_s": round(decode_time, 4),
                "status": "success"
            }

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_str:
                clear_gpu_memory(device)
                return {
                    "ttft_ms": "OOM",
                    "tokens_per_sec": "OOM",
                    "total_gen_time_s": "OOM",
                    "status": "OOM"
                }
            raise e


class MemoryProfiler:
    """
    Profiles Peak CUDA VRAM allocation and ExecuTorch AOT static Tensor Arena memory.
    """
    @staticmethod
    def get_peak_vram_mb(device: str = "cuda") -> float:
        if device == "cuda" and torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            return round(peak_bytes / (1024.0 ** 2), 2)
        return 0.0

    @staticmethod
    def profile_executorch_arena(
        model: nn.Module,
        seq_len: int = 128,
        vocab_size: int = 50257,
        backend: str = "xnnpack"
    ) -> Dict[str, Any]:
        raw_model = getattr(model, "_orig_mod", model)
        orig_device = next(raw_model.parameters()).device
        try:
            from executorch.exir import capture, CaptureConfig, EdgeCompileConfig
            
            model_eval = raw_model.eval().cpu()
            example_inputs = (torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long),)
            
            with torch.no_grad():
                exported_program = torch.export.export(model_eval, example_inputs)
                
            edge_program = capture(
                exported_program,
                example_inputs,
                config=CaptureConfig(edge_compile_config=EdgeCompileConfig(_check_ir_validity=False))
            )
            
            planned_program = edge_program.to_edge()
            arena_size_bytes = getattr(planned_program, "non_const_buffer_size", 0)
            arena_size_mb = round(arena_size_bytes / (1024.0 ** 2), 2)
            
            return {"arena_ram_mb": arena_size_mb, "status": "success"}
        except Exception as e:
            return {"arena_ram_mb": "N/A", "status": f"skipped ({type(e).__name__})"}
        finally:
            raw_model.to(orig_device)


class RooflineEstimator:
    """
    Analytical Roofline Model for memory-bandwidth bound LLM autoregressive decoding.
    """
    DEVICE_BANDWIDTH_GBPS = {
        "raspberry_pi_5": 17.0,
        "qualcomm_8_gen3": 60.0,
        "iphone_16_pro": 150.0,
        "apple_m3_max": 300.0,
    }

    @staticmethod
    def calculate_active_cache_bytes(model: nn.Module, seq_len: int, precision_bytes: float = 2.0) -> int:
        info = extract_kv_cache_config(model)
        
        if info.get("is_baretorch", False):
            lrad_count = info["lrad_count"]
            trans_count = info["trans_count"]
            num_heads = info["num_heads"]
            num_kv_heads = info["num_kv_heads"]
            head_dim = info["head_dim"]
            r = info["rank"]

            lrad_state_bytes = lrad_count * (num_heads * r * head_dim) * precision_bytes
            trans_kv_bytes = trans_count * (2 * num_kv_heads * head_dim) * seq_len * precision_bytes
            return int(lrad_state_bytes + trans_kv_bytes)
        else:
            num_layers = info["num_layers"]
            num_kv_heads = info["num_kv_heads"]
            head_dim = info["head_dim"]

            if num_layers == 0 or num_kv_heads == 0 or head_dim == 0:
                return 0

            hf_kv_bytes = num_layers * (2 * num_kv_heads * head_dim) * seq_len * precision_bytes
            return int(hf_kv_bytes)

    @classmethod
    def project_throughput(
        cls,
        param_count_m: float,
        active_cache_bytes: int,
        precision_bytes: float = 2.0
    ) -> Dict[str, float]:
        weight_bytes = (param_count_m * 1e6) * precision_bytes
        total_transfer_bytes = weight_bytes + active_cache_bytes
        total_transfer_gb = total_transfer_bytes / (1024.0 ** 3)

        projections = {}
        for device_name, bandwidth_gbps in cls.DEVICE_BANDWIDTH_GBPS.items():
            fps = bandwidth_gbps / total_transfer_gb
            projections[device_name] = round(fps, 2)
        return projections