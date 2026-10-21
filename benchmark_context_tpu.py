# /content/baretorch/benchmark_context_tpu.py
import os

# Enable TorchTPU's DeferAndFuse mode and increase Dynamo cache limit for multi-context benchmarks
os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"

import argparse
import csv
import gc
import math
import time
import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = True

BARETORCH_ID = "model-rampage/BareTorch-2.5B-Base"
QWEN_ID = "Qwen/Qwen3.5-2B"

DEFAULT_QWEN_TPU_RESULTS = {
    512: {"context_len": 512, "ttft_ms": 188.9, "prefill_tok_s": 2710.3, "decode_tok_s": 20.56, "peak_hbm_gb": 0.0, "warmup_compile_s": 61.7, "status": "SUCCESS"},
    1024: {"context_len": 1024, "ttft_ms": 298.7, "prefill_tok_s": 3428.2, "decode_tok_s": 22.20, "peak_hbm_gb": 0.0, "warmup_compile_s": 92.2, "status": "SUCCESS"},
    2048: {"context_len": 2048, "ttft_ms": 513.9, "prefill_tok_s": 3984.9, "decode_tok_s": 21.07, "peak_hbm_gb": 0.0, "warmup_compile_s": 105.6, "status": "SUCCESS"},
    4096: {"context_len": 4096, "ttft_ms": 897.6, "prefill_tok_s": 4563.4, "decode_tok_s": 21.27, "peak_hbm_gb": 0.0, "warmup_compile_s": 118.7, "status": "SUCCESS"},
    8192: {"context_len": 8192, "ttft_ms": 2272.2, "prefill_tok_s": 3605.2, "decode_tok_s": 20.31, "peak_hbm_gb": 0.0, "warmup_compile_s": 160.3, "status": "SUCCESS"},
    16384: {"context_len": 16384, "ttft_ms": 5733.5, "prefill_tok_s": 2857.6, "decode_tok_s": 20.63, "peak_hbm_gb": 0.0, "warmup_compile_s": 286.5, "status": "SUCCESS"},
    32768: {"context_len": 32768, "ttft_ms": 17510.6, "prefill_tok_s": 1871.3, "decode_tok_s": 21.07, "peak_hbm_gb": 0.0, "warmup_compile_s": 380.7, "status": "SUCCESS"},
}
# /content/baretorch/benchmark_context_tpu.py
import os

# Enable TorchTPU's DeferAndFuse mode for automatic op fusion and lower HBM usage
os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"

import argparse
import csv
import gc
import math
import time
import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.suppress_errors = False

BARETORCH_ID = "model-rampage/BareTorch-2.5B-Base"
QWEN_ID = "Qwen/Qwen3.5-2B"

DEFAULT_QWEN_TPU_RESULTS = {
    512: {"context_len": 512, "ttft_ms": 188.9, "prefill_tok_s": 2710.3, "decode_tok_s": 20.56, "peak_hbm_gb": 0.0, "warmup_compile_s": 61.7, "status": "SUCCESS"},
    1024: {"context_len": 1024, "ttft_ms": 298.7, "prefill_tok_s": 3428.2, "decode_tok_s": 22.20, "peak_hbm_gb": 0.0, "warmup_compile_s": 92.2, "status": "SUCCESS"},
    2048: {"context_len": 2048, "ttft_ms": 513.9, "prefill_tok_s": 3984.9, "decode_tok_s": 21.07, "peak_hbm_gb": 0.0, "warmup_compile_s": 105.6, "status": "SUCCESS"},
    4096: {"context_len": 4096, "ttft_ms": 897.6, "prefill_tok_s": 4563.4, "decode_tok_s": 21.27, "peak_hbm_gb": 0.0, "warmup_compile_s": 118.7, "status": "SUCCESS"},
    8192: {"context_len": 8192, "ttft_ms": 2272.2, "prefill_tok_s": 3605.2, "decode_tok_s": 20.31, "peak_hbm_gb": 0.0, "warmup_compile_s": 160.3, "status": "SUCCESS"},
    16384: {"context_len": 16384, "ttft_ms": 5733.5, "prefill_tok_s": 2857.6, "decode_tok_s": 20.63, "peak_hbm_gb": 0.0, "warmup_compile_s": 286.5, "status": "SUCCESS"},
    32768: {"context_len": 32768, "ttft_ms": 17510.6, "prefill_tok_s": 1871.3, "decode_tok_s": 21.07, "peak_hbm_gb": 0.0, "warmup_compile_s": 380.7, "status": "SUCCESS"},
}


def get_tpu_compile_backend() -> str:
    """Detects the best registered TorchDynamo backend for TorchTPU."""
    avail = torch._dynamo.list_backends()
    for candidate in ["tpu", "openxla", "inductor"]:
        if candidate in avail:
            return candidate
    return avail[0] if avail else "inductor"


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


def _unwrap_kv_pair_torch(past_kv):
    if past_kv is None:
        return None, None
    p_item = past_kv
    while (
        isinstance(p_item, (tuple, list))
        and len(p_item) > 0
        and not isinstance(p_item[0], torch.Tensor)
    ):
        p_item = p_item[0]
    if isinstance(p_item, (tuple, list)) and len(p_item) >= 2:
        pk, pv = p_item[0], p_item[1]
        if isinstance(pk, torch.Tensor) and isinstance(pv, torch.Tensor):
            return pk, pv
    return None, None


# --- Top-Level Class Methods (TorchDynamo-Compatible without __get__ closures) ---

def _fused_mlp_forward(self_mlp, x):
    h1, h2 = F.linear(x, self_mlp._w12_weight).chunk(2, dim=-1)
    return self_mlp.w3(F.silu(h1) * h2)


def _fused_gqa_forward(self_attn, x, past_kv=None, position_ids=None, attention_mask=None):
    B, L, D = x.shape
    pk_past, pv_past = _unwrap_kv_pair_torch(past_kv)
    past_len = pk_past.size(-2) if pk_past is not None else 0

    qkv = F.linear(x, self_attn._qkv_weight)
    qd = self_attn._q_dim
    kvd = self_attn._kv_dim
    q_raw, k_raw, v_raw = torch.split(qkv, [qd, kvd, kvd], dim=-1)

    q = q_raw.view(B, L, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    k = k_raw.view(B, L, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)
    v = v_raw.view(B, L, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)

    if getattr(self_attn, "use_qk_norm", False):
        q = self_attn.q_norm(q)
        k = self_attn.k_norm(k)

    cos = self_attn._cos_cache[:, :, past_len : past_len + L, :]
    sin = self_attn._sin_cache[:, :, past_len : past_len + L, :]
    half = self_attn.head_dim // 2
    q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    k_rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    q = (q * cos) + (q_rot * sin)
    k = (k * cos) + (k_rot * sin)

    if pk_past is not None and pv_past is not None:
        k = torch.cat([pk_past, k], dim=-2)
        v = torch.cat([pv_past, v], dim=-2)

    current_kv = (k, v)

    if self_attn.num_queries_per_kv > 1:
        k_attn = k.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
        v_attn = v.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
    else:
        k_attn, v_attn = k, v

    is_causal = (past_len == 0) and (L > 1) and (attention_mask is None)
    out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=None, is_causal=is_causal)
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    return self_attn.W_out(out), current_kv


def _fused_lrad_chunk_forward(self_attn, x, attention_mask=None, use_cache=True):
    B, L, _ = x.shape
    H, C, d_h, r = self_attn.num_heads, self_attn.chunk_size, self_attn.d_head, self_attn.r
    d_in = self_attn._d_inner
    ur_in = self_attn._ur_dim
    scaling = 1.0 / math.sqrt(d_h)

    N = (L + C - 1) // C
    pad_len = (C - (L % C)) % C
    x_padded = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x

    if attention_mask is not None:
        mask_padded = F.pad(attention_mask, (0, pad_len), value=0) if pad_len > 0 else attention_mask
        mask_5d = mask_padded.view(B, 1, N, C, 1).to(torch.float32)
    else:
        seq_idx = torch.arange(N * C, device=x.device)
        mask_5d = (seq_idx < L).to(torch.float32).view(1, 1, N, C, 1)

    proj = F.linear(x_padded, self_attn._qkvurs_w)
    q_raw, k_raw, v_raw, u_raw, r_raw, swish_raw = torch.split(
        proj, [d_in, d_in, d_in, ur_in, ur_in, d_in], dim=-1
    )
    g_raw, bg_raw = F.linear(x_padded, self_attn._gates_w, self_attn._gates_b).chunk(2, dim=-1)

    Q = F.silu(q_raw).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
    K = F.silu(k_raw).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
    V = v_raw.view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
    U = u_raw.view(B, N, C, H, r).permute(0, 3, 1, 2, 4)
    R = r_raw.view(B, N, C, H, r).permute(0, 3, 1, 2, 4)

    gate_fp32 = torch.sigmoid(g_raw.float()).clamp(1e-4, 0.9999).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
    beta_gate_fp32 = torch.sigmoid(bg_raw.float()).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)

    U_fp32 = U.float() * mask_5d
    V_fp32 = V.float() * mask_5d
    gate_fp32 = torch.where(mask_5d.bool(), gate_fp32, 1.0)

    log_gate = torch.log(gate_fp32)
    Lambda = torch.cumsum(log_gate, dim=-2)
    exp_Lambda = torch.exp(Lambda.clamp(min=-30.0, max=0.0))

    idx = self_attn._chunk_arange
    causal_mask = (idx[:, None] >= idx[None, :]).view(1, 1, 1, C, C)
    diff = torch.where(causal_mask, Lambda - Lambda.transpose(-1, -2), -1e4)
    M_links = torch.exp(diff.clamp(min=-1e4, max=0.0))

    Y_local = ((Q.float() @ K.float().transpose(-1, -2)) * scaling * M_links) @ V_fp32

    chunk_decay_log = log_gate.sum(dim=-2).squeeze(-1).clamp(min=-30.0, max=0.0)
    Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)
    log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
    c_idx = torch.arange(N, device=x.device)
    M_chunks = torch.exp(
        torch.where((c_idx[:, None] > c_idx[None, :]).view(1, 1, N, N), log_M_chunks, -1e4).clamp(
            min=-1e4, max=0.0
        )
    )

    decay = torch.exp((Lambda[:, :, :, -1:, :] - Lambda).clamp(min=-30.0, max=0.0))
    U_decayed = (U_fp32 * beta_gate_fp32) * decay

    S_historical = (M_chunks @ (U_decayed.transpose(-1, -2) @ V_fp32).view(B, H, N, r * d_h)).view(
        B, H, N, r, d_h
    )
    Y_global = (R.float() * exp_Lambda) @ S_historical * scaling

    Out = (
        (Y_local + Y_global)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(B, N * C, d_in)
        .to(x.dtype)[:, :L, :]
    )
    output = self_attn.W_out(Out * F.silu(swish_raw[:, :L, :]))

    next_cache = None
    if use_cache:
        S_local_last = U_decayed[:, :, -1].transpose(-1, -2) @ V_fp32[:, :, -1]
        S_final = ((exp_Lambda[:, :, -1, -1:, :] * S_historical[:, :, -1]) + S_local_last).clamp(
            -50.0, 50.0
        ).to(x.dtype)
        S_intra = torch.zeros((B, H, d_h, d_h), device=x.device, dtype=x.dtype)
        S_lowrank = torch.zeros((B, H, r, d_h), device=x.device, dtype=x.dtype)
        gamma = torch.zeros((B, H, 1, 1), device=x.device, dtype=x.dtype)
        next_cache = (S_final, S_intra, S_lowrank, gamma, 0, L)
    return output, next_cache


def _fused_lrad_step_math(x, qkvurs_w, gates_w, gates_b, w_out_w, past_S, S_intra, S_lowrank, gamma, d_in: int, ur_in: int, H: int, d_h: int, r: int, scaling: float):
    """Pure tensor-in / tensor-out function for single-token CS-LRAD decoding (compiles cleanly with zero graph breaks)."""
    B = x.size(0)
    proj = F.linear(x, qkvurs_w)
    q_raw, k_raw, v_raw, u_raw, r_raw, swish_raw = torch.split(
        proj, [d_in, d_in, d_in, ur_in, ur_in, d_in], dim=-1
    )
    gates = torch.sigmoid(F.linear(x, gates_w, gates_b))
    g_val, bg_val = gates.view(B, H, 2, 1, 1).unbind(dim=2)

    Q = F.silu(q_raw).view(B, H, 1, d_h)
    K_T = F.silu(k_raw).view(B, H, d_h, 1)
    V = v_raw.view(B, H, 1, d_h)
    U_T = u_raw.view(B, H, r, 1) * bg_val
    R = r_raw.view(B, H, 1, r)

    S_intra_new = (g_val * S_intra) + (K_T @ V)
    S_lowrank_new = (g_val * S_lowrank) + (U_T @ V)
    gamma_new = g_val * gamma

    Y = ((Q @ S_intra_new) + ((R * gamma_new) @ past_S)) * scaling
    Out = Y.view(B, 1, d_in)
    output = F.linear(Out * F.silu(swish_raw), w_out_w)
    return output, S_intra_new, S_lowrank_new, gamma_new


def _fused_lrad_step_forward(self_attn, x, past_state=None, use_cache=True, attention_mask=None):
    B, L, _ = x.shape
    if L > 1:
        return self_attn.chunk_forward(x, attention_mask=attention_mask, use_cache=use_cache)
    H, C, d_h, r = self_attn.num_heads, self_attn.chunk_size, self_attn.d_head, self_attn.r
    d_in = self_attn._d_inner
    ur_in = self_attn._ur_dim
    scaling = 1.0 / math.sqrt(d_h)

    past_S, S_intra, S_lowrank, gamma, step_in_chunk, curr_seq_len = past_state

    if step_in_chunk == 0:
        S_intra = torch.zeros_like(S_intra)
        S_lowrank = torch.zeros_like(S_lowrank)
        gamma = torch.ones_like(gamma)

    step_fn = getattr(self_attn, "_compiled_step_math", _fused_lrad_step_math)
    output, S_intra, S_lowrank, gamma = step_fn(
        x,
        self_attn._qkvurs_w,
        self_attn._gates_w,
        self_attn._gates_b,
        self_attn.W_out.weight,
        past_S,
        S_intra,
        S_lowrank,
        gamma,
        d_in,
        ur_in,
        H,
        d_h,
        r,
        scaling,
    )

    next_step = step_in_chunk + 1
    if next_step >= C:
        past_S = ((gamma * past_S) + S_lowrank).clamp(-50.0, 50.0)
        next_step = 0

    return output, (past_S, S_intra, S_lowrank, gamma, next_step, curr_seq_len + 1)


def fuse_baretorch_pytorch_for_inference(model, use_compile: bool = False, backend: str = "tpu"):
    """
    Fuses BareTorch's linear projections at the class level and optionally compiles
    the pure tensor-in/tensor-out CS-LRAD and MLP kernels using `torch.compile`.
    """
    fused_mlps, fused_gqa, fused_lrad = 0, 0, 0
    compiled_step_fn = (
        torch.compile(_fused_lrad_step_math, backend=backend, dynamic=False)
        if use_compile
        else _fused_lrad_step_math
    )

    with torch.no_grad():
        for module in model.modules():
            cls_name = module.__class__.__name__

            # 1. Fuse GatedMLP (w1 + w2 -> _w12_weight)
            if cls_name == "GatedMLP" and hasattr(module, "w1") and hasattr(module, "w2"):
                w12_weight = torch.cat([module.w1.weight.data, module.w2.weight.data], dim=0).contiguous()
                module.register_buffer("_w12_weight", w12_weight, persistent=False)
                del module.w1, module.w2
                module.__class__.forward = _fused_mlp_forward
                fused_mlps += 1

            # 2. Fuse CausalSelfAttention (W_q + W_k + W_v -> _qkv_weight) + Pre-computed RoPE cos/sin table
            elif cls_name == "CausalSelfAttention" and hasattr(module, "W_q") and hasattr(module, "W_k"):
                q_dim = module.W_q.weight.shape[0]
                kv_dim = module.W_k.weight.shape[0]
                qkv_weight = torch.cat(
                    [module.W_q.weight.data, module.W_k.weight.data, module.W_v.weight.data], dim=0
                ).contiguous()
                module.register_buffer("_qkv_weight", qkv_weight, persistent=False)
                module._q_dim = q_dim
                module._kv_dim = kv_dim

                max_rope_len = 33024
                pos_idx = torch.arange(max_rope_len, dtype=torch.float32, device=qkv_weight.device)
                freqs = torch.outer(pos_idx, module.rope.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                module.register_buffer("_cos_cache", emb.cos().to(qkv_weight.dtype).view(1, 1, max_rope_len, -1), persistent=False)
                module.register_buffer("_sin_cache", emb.sin().to(qkv_weight.dtype).view(1, 1, max_rope_len, -1), persistent=False)
                del module.W_q, module.W_k, module.W_v
                module.__class__.forward = _fused_gqa_forward
                fused_gqa += 1

            # 3. Fuse LowRankAssociativeDeltaEngine + O(1) Recurrent Step Forward
            elif cls_name == "LowRankAssociativeDeltaEngine" and hasattr(module, "W_q"):
                d = module.inner_dim
                ur = module.num_heads * module.r
                qkvurs_w = torch.cat(
                    [
                        module.W_q.weight.data,
                        module.W_k.weight.data,
                        module.W_v.weight.data,
                        module.W_u.weight.data,
                        module.W_r.weight.data,
                        module.W_swish_gate.weight.data,
                    ],
                    dim=0,
                ).contiguous()
                gates_w = torch.cat([module.W_gate.weight.data, module.W_beta_gate.weight.data], dim=0).contiguous()
                gates_b = torch.cat([module.W_gate.bias.data, module.W_beta_gate.bias.data], dim=0).contiguous()

                module.register_buffer("_qkvurs_w", qkvurs_w, persistent=False)
                module.register_buffer("_gates_w", gates_w, persistent=False)
                module.register_buffer("_gates_b", gates_b, persistent=False)
                module.register_buffer("_chunk_arange", torch.arange(module.chunk_size, dtype=torch.long, device=qkvurs_w.device), persistent=False)
                module._d_inner = d
                module._ur_dim = ur
                module._compiled_step_math = compiled_step_fn
                del module.W_q, module.W_k, module.W_v, module.W_u, module.W_r, module.W_swish_gate
                del module.W_gate, module.W_beta_gate

                module.__class__.chunk_forward = _fused_lrad_chunk_forward
                module.__class__.step_forward = _fused_lrad_step_forward
                fused_lrad += 1

    print(
        f"  ⚡ Fused BareTorch projections & enabled O(1) Recurrent Step Forward: "
        f"{fused_lrad} CS-LRAD blocks, {fused_gqa} GQA blocks, and {fused_mlps} MLPs! "
        f"(compile={use_compile}, backend='{backend}')"
    )


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

    backend = get_tpu_compile_backend()
    if "baretorch" in model_id.lower():
        fuse_baretorch_pytorch_for_inference(model, use_compile=use_compile, backend=backend)
    elif use_compile:
        print(f"  🔥 Compiling {label} with torch.compile(backend='{backend}')...")
        model = torch.compile(model, backend=backend)

    sync_tpu()

    results = {}
    for ctx in context_lengths:
        print(f"  --> Benchmarking {label} @ {ctx:,} tokens...")
        inp = torch.randint(100, 10000, (1, ctx), dtype=torch.long, device=device)
        sync_tpu()

        try:
            # 1. Warmup: Compiles prefill + decode step graphs
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
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Compile pure tensor math with torch.compile (default: False)",
    )
    parser.add_argument(
        "--no_compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--skip_qwen",
        action="store_true",
        default=False,
        help="Skip re-running Qwen3.5-2B and use verified Qwen TPU baseline numbers in the summary table",
    )
    parser.add_argument("--csv_path", type=str, default="benchmark_tpu_colab.csv")
    args = parser.parse_args()

    device = torch.device("tpu")
    print("=" * 95)
    print(f"⚡ GOOGLE TORCHTPU BENCHMARK: BareTorch-2.5B (O(1) Recurrent) vs. Qwen3.5-2B")
    print(f"Contexts     : {args.context_lengths}")
    print(f"Decode Steps : {args.gen_tokens} tokens")
    print(f"Trials       : {args.trials} per context length")
    print(f"Compile      : {args.compile}")
    print(f"Skip Qwen    : {args.skip_qwen}")
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

    if args.skip_qwen:
        print("\n⏩ Skipping Qwen3.5-2B execution (using verified Qwen3.5-2B TorchTPU baseline results).")
        qw_res = {ctx: DEFAULT_QWEN_TPU_RESULTS[ctx] for ctx in args.context_lengths if ctx in DEFAULT_QWEN_TPU_RESULTS}
    else:
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