# baretorch/modeling_mlx/transformer.py
import math
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: mx.array) -> mx.array:
        x_fp32 = x.astype(mx.float32)
        variance = mx.mean(mx.square(x_fp32), axis=-1, keepdims=True)
        x_norm = x_fp32 * mx.rsqrt(variance + self.eps)
        return (x_norm * self.weight.astype(mx.float32)).astype(x.dtype)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))

    def _rotate_half(self, x: mx.array) -> mx.array:
        half = self.dim // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def apply_rope(self, q: mx.array, k: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # Defensively slice position_ids to match sequence length of q
        if position_ids is not None and position_ids.shape[-1] != q.shape[-2]:
            if q.shape[-2] == 1:
                position_ids = position_ids[:, -1:]
            else:
                position_ids = position_ids[:, -q.shape[-2]:]

        freqs = mx.expand_dims(position_ids.astype(mx.float32), axis=-1) * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), axis=1).astype(q.dtype)
        sin = mx.expand_dims(mx.sin(emb), axis=1).astype(q.dtype)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Expands Key/Value heads to match Query heads for Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    x = mx.broadcast_to(x[:, :, None, :, :], (B, H, n_rep, L, D))
    return mx.reshape(x, (B, H * n_rep, L, D))


def unwrap_kv_pair(past_kv):
    """Safely extracts matching (key_array, value_array) pair without history contamination."""
    if past_kv is None:
        return None, None
    p_item = past_kv
    while isinstance(p_item, (tuple, list)) and len(p_item) > 0 and not isinstance(p_item[0], mx.array):
        p_item = p_item[0]
    if isinstance(p_item, (tuple, list)) and len(p_item) >= 2:
        pk, pv = p_item[0], p_item[1]
        if isinstance(pk, mx.array) and isinstance(pv, mx.array):
            return pk, pv
    return None, None


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        max_seq_len: int = 4096,
        use_qk_norm: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.use_qk_norm = use_qk_norm

        assert num_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads."
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len, base=rope_theta)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        past_kv: tuple | None = None,
        position_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim

        pk_past, pv_past = unwrap_kv_pair(past_kv)
        past_len = pk_past.shape[-2] if pk_past is not None else 0

        if position_ids is None:
            position_ids = mx.arange(past_len, past_len + L, dtype=mx.int32)[None, :]

        q = mx.transpose(mx.reshape(self.W_q(x), (B, L, H_q, d_h)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(self.W_k(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))
        v = mx.transpose(mx.reshape(self.W_v(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))

        if self.use_qk_norm:
            q = mx.transpose(self.q_norm(mx.transpose(q, (0, 2, 1, 3))), (0, 2, 1, 3))
            k = mx.transpose(self.k_norm(mx.transpose(k, (0, 2, 1, 3))), (0, 2, 1, 3))

        q, k = self.rope.apply_rope(q, k, position_ids)

        if pk_past is not None and pv_past is not None:
            k = mx.concatenate([pk_past, k], axis=-2)
            v = mx.concatenate([pv_past, v], axis=-2)

        # Store cache prior to repeat_kv to preserve compact KV memory footprint
        current_kv = (k, v)

        if H_kv != H_q:
            k = repeat_kv(k, self.num_queries_per_kv)
            v = repeat_kv(v, self.num_queries_per_kv)

        scale = 1.0 / math.sqrt(d_h)
        total_k_len = k.shape[-2]

        # 1. Base Causal Mask
        mask = None
        if L > 1:
            q_idx = mx.arange(past_len, past_len + L)
            k_idx = mx.arange(total_k_len)
            mask = mx.where(q_idx[:, None] >= k_idx[None, :], 0.0, -1e4).astype(q.dtype)

        # 2. Additive Padding Mask Blending
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                pad_mask = mx.where(
                    attention_mask[:, None, None, -total_k_len:] > 0, 0.0, -1e4
                ).astype(q.dtype)
            elif attention_mask.ndim == 4:
                pad_mask = mx.where(
                    attention_mask[:, :, -L:, -total_k_len:] > 0, 0.0, -1e4
                ).astype(q.dtype)
            else:
                pad_mask = None

            if pad_mask is not None:
                mask = mask + pad_mask if mask is not None else pad_mask

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out_flat = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, L, D))
        return self.W_out(out_flat), current_kv


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = 4,
        max_seq_len: int = 4096,
        use_qk_norm: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            use_qk_norm=use_qk_norm,
            rope_theta=rope_theta,
        )
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5))

    def __call__(
        self,
        x: mx.array,
        past_kv: tuple | None = None,
        position_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ):
        attn_out, current_kv = self.attn(
            self.ln1(x),
            past_kv=past_kv,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        x_out = x + attn_out
        x_out = x_out + self.mlp(self.ln2(x_out))
        return x_out, current_kv