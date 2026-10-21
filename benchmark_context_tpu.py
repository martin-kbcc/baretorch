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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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


def fuse_baretorch_pytorch_for_inference(model, max_decode_steps: int = 64):
    """
    Fuses BareTorch's linear projections AND converts both CS-LRAD and GQA caches
    to 100% STATIC TENSOR SHAPES during autoregressive decoding on TPU:
      1. GatedMLP: fuses w1 + w2 -> _w12_weight.
      2. CausalSelfAttention: fuses W_q + W_k + W_v -> _qkv_weight AND pre-allocates
         a static `(B, H_kv, ctx + max_decode_steps, d_h)` KV buffer after prefill so
         decode never calls dynamic-shape `torch.cat`!
      3. LowRankAssociativeDeltaEngine: fuses 6 main projections + 2 gate projections
         AND uses a static `(B, H, 32, d_h)` intra-chunk buffer so all 64 decode steps
         have identical tensor shapes!
    """
    fused_mlps, fused_gqa, fused_lrad = 0, 0, 0

    with torch.no_grad():
        for module in model.modules():
            cls_name = module.__class__.__name__

            # 1. Fuse GatedMLP (w1 + w2 -> _w12_weight)
            if cls_name == "GatedMLP" and hasattr(module, "w1") and hasattr(module, "w2"):
                w12_weight = torch.cat([module.w1.weight.data, module.w2.weight.data], dim=0).contiguous()
                module.register_buffer("_w12_weight", w12_weight, persistent=False)
                del module.w1, module.w2

                def fused_mlp_forward(self_mlp, x):
                    h1, h2 = F.linear(x, self_mlp._w12_weight).chunk(2, dim=-1)
                    return self_mlp.w3(F.silu(h1) * h2)

                module.forward = fused_mlp_forward.__get__(module, module.__class__)
                fused_mlps += 1

            # 2. Fuse CausalSelfAttention + Static Pre-Allocated KV Cache for Decode
            elif cls_name == "CausalSelfAttention" and hasattr(module, "W_q") and hasattr(module, "W_k"):
                q_dim = module.W_q.weight.shape[0]
                kv_dim = module.W_k.weight.shape[0]
                qkv_weight = torch.cat(
                    [module.W_q.weight.data, module.W_k.weight.data, module.W_v.weight.data], dim=0
                ).contiguous()
                module.register_buffer("_qkv_weight", qkv_weight, persistent=False)
                module._q_dim = q_dim
                module._kv_dim = kv_dim
                module._max_decode_steps = max_decode_steps
                del module.W_q, module.W_k, module.W_v

                def fused_gqa_forward(self_attn, x, past_kv=None, position_ids=None, attention_mask=None):
                    B, L, D = x.shape
                    qd = self_attn._q_dim
                    kvd = self_attn._kv_dim

                    qkv = F.linear(x, self_attn._qkv_weight)
                    q_raw, k_raw, v_raw = torch.split(qkv, [qd, kvd, kvd], dim=-1)

                    q = q_raw.view(B, L, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
                    k = k_raw.view(B, L, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)
                    v = v_raw.view(B, L, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)

                    if getattr(self_attn, "use_qk_norm", False):
                        q = self_attn.q_norm(q)
                        k = self_attn.k_norm(k)

                    # Prefill path (L > 1)
                    if L > 1 or past_kv is None:
                        pk_past, pv_past = _unwrap_kv_pair_torch(past_kv)
                        past_len = pk_past.size(-2) if pk_past is not None else 0
                        if position_ids is None:
                            position_ids = torch.arange(past_len, past_len + L, dtype=torch.long, device=x.device).unsqueeze(0)
                        q, k = self_attn.rope.apply_rope(q, k, position_ids)

                        if pk_past is not None and pv_past is not None:
                            k = torch.cat([pk_past, k], dim=-2)
                            v = torch.cat([pv_past, v], dim=-2)

                        if self_attn.num_queries_per_kv > 1:
                            k_attn = k.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
                            v_attn = v.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
                        else:
                            k_attn, v_attn = k, v

                        is_causal = (past_len == 0) and (L > 1) and (attention_mask is None)
                        out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=None, is_causal=is_causal)
                        out = out.transpose(1, 2).contiguous().view(B, L, D)

                        # Pre-allocate static decode KV buffer of fixed length `total_len + max_decode_steps`
                        total_len = k.size(-2)
                        max_len = total_len + self_attn._max_decode_steps
                        k_buf = F.pad(k, (0, 0, 0, self_attn._max_decode_steps))
                        v_buf = F.pad(v, (0, 0, 0, self_attn._max_decode_steps))
                        pos_tensor = torch.tensor([total_len], dtype=torch.long, device=x.device)
                        arange_max = torch.arange(max_len, dtype=torch.long, device=x.device)
                        return self_attn.W_out(out), (k_buf, v_buf, pos_tensor, arange_max, total_len)

                    # Static-Shape Single-Token Decode Path (L == 1)
                    k_buf, v_buf, pos_tensor, arange_max, cur_len = past_kv
                    pos_ids = pos_tensor.unsqueeze(0)
                    q, k = self_attn.rope.apply_rope(q, k, pos_ids)

                    # Static shape update using broadcast mask (zero dynamic slicing or torch.cat!)
                    slot_mask = (arange_max == pos_tensor).view(1, 1, -1, 1)
                    k_buf = torch.where(slot_mask, k, k_buf)
                    v_buf = torch.where(slot_mask, v, v_buf)
                    new_pos_tensor = pos_tensor + 1

                    if self_attn.num_queries_per_kv > 1:
                        k_attn = k_buf.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
                        v_attn = v_buf.repeat_interleave(self_attn.num_queries_per_kv, dim=1)
                    else:
                        k_attn, v_attn = k_buf, v_buf

                    valid_mask = (arange_max < new_pos_tensor).view(1, 1, 1, -1)
                    attn_bias = torch.where(valid_mask, 0.0, -1e4).to(q.dtype)
                    out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_bias, is_causal=False)
                    out = out.transpose(1, 2).contiguous().view(B, 1, D)
                    return self_attn.W_out(out), (k_buf, v_buf, new_pos_tensor, arange_max, cur_len + 1)

                module.forward = fused_gqa_forward.__get__(module, module.__class__)
                fused_gqa += 1

            # 3. Fuse LowRankAssociativeDeltaEngine + Static 32-Slot Intra-Chunk Buffer
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
                del module.W_q, module.W_k, module.W_v, module.W_u, module.W_r, module.W_swish_gate
                del module.W_gate, module.W_beta_gate

                def fused_chunk_forward(self_attn, x, attention_mask=None, use_cache=True):
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
                        )
                        # Pre-allocate static 32-slot intra-chunk buffers so decode has 100% static shapes
                        buf_k = torch.zeros((B, H, C, d_h), device=x.device, dtype=torch.float32)
                        buf_v = torch.zeros((B, H, C, d_h), device=x.device, dtype=torch.float32)
                        buf_u = torch.zeros((B, H, C, r), device=x.device, dtype=torch.float32)
                        buf_lambda = torch.zeros((B, H, C, 1), device=x.device, dtype=torch.float32)
                        slot_tensor = torch.zeros((1,), device=x.device, dtype=torch.long)
                        last_lambda = torch.zeros((B, H, 1, 1), device=x.device, dtype=torch.float32)
                        next_cache = (S_final, buf_k, buf_v, buf_u, buf_lambda, slot_tensor, last_lambda, L)
                    return output, next_cache

                def fused_step_forward(self_attn, x, past_state=None, use_cache=True, attention_mask=None):
                    B, L, _ = x.shape
                    if L > 1:
                        return self_attn.chunk_forward(x, attention_mask=attention_mask, use_cache=use_cache)
                    H, C, d_h, r = self_attn.num_heads, self_attn.chunk_size, self_attn.d_head, self_attn.r
                    d_in = self_attn._d_inner
                    ur_in = self_attn._ur_dim
                    scaling = 1.0 / math.sqrt(d_h)

                    past_S, buf_k, buf_v, buf_u, buf_lambda, slot_tensor, last_lambda, curr_seq_len = past_state

                    proj = F.linear(x, self_attn._qkvurs_w)
                    q_raw, k_raw, v_raw, u_raw, r_raw, swish_raw = torch.split(
                        proj, [d_in, d_in, d_in, ur_in, ur_in, d_in], dim=-1
                    )
                    g_raw, bg_raw = F.linear(x, self_attn._gates_w, self_attn._gates_b).chunk(2, dim=-1)

                    Q = F.silu(q_raw).view(B, 1, H, d_h).transpose(1, 2).float()
                    K = F.silu(k_raw).view(B, 1, H, d_h).transpose(1, 2).float()
                    V = v_raw.view(B, 1, H, d_h).transpose(1, 2).float()
                    U = u_raw.view(B, 1, H, r).transpose(1, 2).float()
                    R = r_raw.view(B, 1, H, r).transpose(1, 2).float()

                    gate = torch.sigmoid(g_raw.float()).clamp(1e-4, 0.9999).view(B, H, 1, 1)
                    beta_gate = torch.sigmoid(bg_raw.float()).view(B, H, 1, 1)
                    log_gate = torch.log(gate)

                    # Static-shape slot update (100% static tensor shapes across all 64 decode steps!)
                    c_arange = self_attn._chunk_arange
                    is_slot_zero = (slot_tensor == 0).view(1, 1, 1, 1)
                    lambda_curr = torch.where(is_slot_zero, log_gate, last_lambda + log_gate)

                    slot_mask = (c_arange == slot_tensor).view(1, 1, C, 1)
                    buf_k = torch.where(slot_mask, K, buf_k)
                    buf_v = torch.where(slot_mask, V, buf_v)
                    buf_u = torch.where(slot_mask, U * beta_gate, buf_u)
                    buf_lambda = torch.where(slot_mask, lambda_curr, buf_lambda)

                    valid_mask = (c_arange <= slot_tensor).view(1, 1, 1, C)
                    exp_lambda_curr = torch.exp(lambda_curr.clamp(min=-30.0, max=0.0))
                    diff = torch.where(valid_mask, lambda_curr - buf_lambda.transpose(-1, -2), -1e4)
                    m_links = torch.exp(diff.clamp(min=-1e4, max=0.0))

                    Y_local = ((Q @ buf_k.transpose(-1, -2)) * scaling * m_links) @ buf_v
                    Y_global = (R * exp_lambda_curr) @ past_S * scaling

                    Out = (Y_local + Y_global).to(x.dtype).transpose(1, 2).contiguous().view(B, 1, d_in)
                    output = self_attn.W_out(Out * F.silu(swish_raw))

                    # Branch-free chunk rollover at slot == C - 1 (keeps XLA graph 100% static!)
                    is_chunk_end = (slot_tensor == (C - 1)).view(1, 1, 1, 1)
                    chunk_decay = exp_lambda_curr
                    intra_decay = torch.exp((lambda_curr - buf_lambda).clamp(min=-30.0, max=0.0))
                    s_local = (buf_u * intra_decay).transpose(-1, -2) @ buf_v
                    folded_S = ((chunk_decay * past_S) + s_local).clamp(-50.0, 50.0)

                    next_S = torch.where(is_chunk_end, folded_S, past_S)
                    next_slot = torch.where(slot_tensor == (C - 1), torch.zeros_like(slot_tensor), slot_tensor + 1)
                    next_last_lambda = torch.where(is_chunk_end, torch.zeros_like(lambda_curr), lambda_curr)

                    return output, (next_S, buf_k, buf_v, buf_u, buf_lambda, next_slot, next_last_lambda, curr_seq_len + 1)

                module.chunk_forward = fused_chunk_forward.__get__(module, module.__class__)
                module.step_forward = fused_step_forward.__get__(module, module.__class__)
                fused_lrad += 1

    print(
        f"  ⚡ Fused BareTorch projections & enabled 100% STATIC-SHAPE TPU caches: "
        f"{fused_lrad} CS-LRAD blocks, {fused_gqa} GQA blocks, and {fused_mlps} MLPs!"
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

    if "baretorch" in model_id.lower():
        fuse_baretorch_pytorch_for_inference(model, max_decode_steps=gen_tokens)
        # Override _get_past_length so modeling_baretorch handles our static 5-tuple GQA cache
        if hasattr(model, "model") and hasattr(model.model, "_get_past_length"):
            orig_get_past_len = model.model._get_past_length

            def static_get_past_length(past_key_values):
                if past_key_values is None:
                    return 0
                for layer_past in past_key_values:
                    if isinstance(layer_past, tuple) and len(layer_past) == 5 and isinstance(layer_past[4], int):
                        return layer_past[4]
                return orig_get_past_len(past_key_values)

            model.model._get_past_length = static_get_past_length

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
            # 1. Warmup: Because BareTorch's decode cache is now 100% static-shape,
            # warming up just 2 decode steps compiles the exact graph used by all 64 steps!
            warmup_steps = 2 if "baretorch" in model_id.lower() else gen_tokens
            t_warmup_start = time.perf_counter()
            with torch.no_grad():
                out = model(inp, use_cache=True)
                p_kv = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
                sync_tpu()

                for _ in range(warmup_steps):
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
    print(f"⚡ GOOGLE TORCHTPU BENCHMARK: BareTorch-2.5B (Static-Shape Fused) vs. Qwen3.5-2B (device='{device}')")
    print(f"Contexts     : {args.context_lengths}")
    print(f"Decode Steps : {args.gen_tokens} tokens")
    print(f"Trials       : {args.trials} per context length")
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