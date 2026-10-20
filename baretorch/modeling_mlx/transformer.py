# /home/martinkb/Desktop/BareTorch_F/baretorch/modeling_mlx/transformer.py
from functools import partial
import math
import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def fused_swiglu(x1: mx.array, x2: mx.array) -> mx.array:
    return nn.silu(x1) * x2


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.w12 = None

    def fuse_weights(self):
        """Fuses w1 and w2 into a single w12 Linear layer prior to quantization."""
        if self.w12 is None and hasattr(self, "w1") and hasattr(self, "w2"):
            w12_weight = mx.concatenate([self.w1.weight, self.w2.weight], axis=0)
            self.w12 = nn.Linear(self.d_model, 2 * self.d_ff, bias=False)
            self.w12.weight = w12_weight
            mx.eval(self.w12.weight)
            del self.w1
            del self.w2

    def __call__(self, x: mx.array) -> mx.array:
        if self.w12 is not None:
            h12 = self.w12(x)
            h1, h2 = mx.split(h12, 2, axis=-1)
            return self.w3(fused_swiglu(h1, h2))
        return self.w3(fused_swiglu(self.w1(x), self.w2(x)))


class InPlaceKVCache:
    """
    Pre-allocated in-place KV Cache matching mlx_lm's KVCache design.
    Eliminates O(N) full-cache allocations and copies during autoregressive decoding.
    """

    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        B, H_kv, L, D = k.shape
        prev = self.offset

        if self.keys is None or (prev + L) > self.keys.shape[2]:
            n_steps = (self.step + L - 1) // self.step
            new_alloc_len = n_steps * self.step
            new_k = mx.zeros((B, H_kv, new_alloc_len, D), dtype=k.dtype)
            new_v = mx.zeros((B, H_kv, new_alloc_len, D), dtype=v.dtype)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys = new_k
                self.values = new_v

        self.offset += L
        self.keys[..., prev : self.offset, :] = k
        self.values[..., prev : self.offset, :] = v
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            return ()
        return (self.keys[..., : self.offset, :], self.values[..., : self.offset, :])

    def __iter__(self):
        if self.keys is not None:
            yield self.keys[..., : self.offset, :]
            yield self.values[..., : self.offset, :]


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )

    def apply_rope(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
        position_ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        # Fast fused Metal kernel path when positions are contiguous
        if position_ids is None:
            q_embed = mx.fast.rope(
                q, self.dim, traditional=False, base=self.base, scale=1.0, offset=offset
            )
            k_embed = mx.fast.rope(
                k, self.dim, traditional=False, base=self.base, scale=1.0, offset=offset
            )
            return q_embed, k_embed

        if position_ids.shape[-1] != q.shape[-2]:
            if q.shape[-2] == 1:
                position_ids = position_ids[:, -1:]
            else:
                position_ids = position_ids[:, -q.shape[-2] :]

        freqs = (
            mx.expand_dims(position_ids.astype(mx.float32), axis=-1) * self.inv_freq
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), axis=1).astype(q.dtype)
        sin = mx.expand_dims(mx.sin(emb), axis=1).astype(q.dtype)

        half = self.dim // 2
        q_rot = mx.concatenate([-q[..., half:], q[..., :half]], axis=-1)
        k_rot = mx.concatenate([-k[..., half:], k[..., :half]], axis=-1)
        return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


def unwrap_kv_pair(past_kv):
    if past_kv is None:
        return None, None
    if isinstance(past_kv, InPlaceKVCache):
        if past_kv.keys is None:
            return None, None
        return past_kv.keys[..., : past_kv.offset, :], past_kv.values[..., : past_kv.offset, :]
    p_item = past_kv
    while (
        isinstance(p_item, (tuple, list))
        and len(p_item) > 0
        and not isinstance(p_item[0], mx.array)
    ):
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
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert num_heads % num_kv_heads == 0, (
            "Query heads must be divisible by KV heads."
        )
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_qkv = None

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_seq_len, base=rope_theta
        )
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def fuse_weights(self):
        """Fuses W_q, W_k, W_v into a single W_qkv Linear projection prior to quantization."""
        if self.W_qkv is None and hasattr(self, "W_q"):
            qkv_w = mx.concatenate([self.W_q.weight, self.W_k.weight, self.W_v.weight], axis=0)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            self.W_qkv = nn.Linear(self.d_model, q_dim + 2 * kv_dim, bias=False)
            self.W_qkv.weight = qkv_w
            mx.eval(self.W_qkv.weight)
            del self.W_q
            del self.W_k
            del self.W_v

    def __call__(
        self,
        x: mx.array,
        past_kv: InPlaceKVCache | tuple | None = None,
        position_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim

        if isinstance(past_kv, InPlaceKVCache):
            past_len = past_kv.offset
            kv_cache = past_kv
        elif past_kv is None:
            past_len = 0
            kv_cache = InPlaceKVCache()
        else:
            pk_past, pv_past = unwrap_kv_pair(past_kv)
            past_len = pk_past.shape[-2] if pk_past is not None else 0
            kv_cache = InPlaceKVCache()
            if pk_past is not None and pv_past is not None:
                kv_cache.update_and_fetch(pk_past, pv_past)

        if self.W_qkv is not None:
            qkv = self.W_qkv(x)
            q_dim = H_q * d_h
            kv_dim = H_kv * d_h
            q_raw, k_raw, v_raw = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)
            q = mx.transpose(mx.reshape(q_raw, (B, L, H_q, d_h)), (0, 2, 1, 3))
            k = mx.transpose(mx.reshape(k_raw, (B, L, H_kv, d_h)), (0, 2, 1, 3))
            v = mx.transpose(mx.reshape(v_raw, (B, L, H_kv, d_h)), (0, 2, 1, 3))
        else:
            q = mx.transpose(mx.reshape(self.W_q(x), (B, L, H_q, d_h)), (0, 2, 1, 3))
            k = mx.transpose(mx.reshape(self.W_k(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))
            v = mx.transpose(mx.reshape(self.W_v(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Use native mx.fast.rope when no custom non-contiguous position_ids are required
        use_fast_rope = (attention_mask is None) and (position_ids is None)
        q, k = self.rope.apply_rope(
            q,
            k,
            offset=past_len,
            position_ids=None if use_fast_rope else position_ids,
        )

        k, v = kv_cache.update_and_fetch(k, v)
        total_k_len = k.shape[-2]

        mask = None
        if L > 1:
            if attention_mask is None:
                mask = "causal"
            else:
                q_idx = mx.arange(past_len, past_len + L)
                k_idx = mx.arange(total_k_len)
                mask = mx.where(q_idx[:, None] >= k_idx[None, :], 0.0, -1e4).astype(
                    q.dtype
                )

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
                mask = mask + pad_mask if isinstance(mask, mx.array) else pad_mask

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out_flat = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, L, D))
        return self.W_out(out_flat), kv_cache


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

    def fuse_weights(self):
        self.attn.fuse_weights()
        self.mlp.fuse_weights()

    def __call__(
        self,
        x: mx.array,
        past_kv: InPlaceKVCache | tuple | None = None,
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