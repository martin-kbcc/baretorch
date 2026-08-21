# baretorch/deploy/apple_mlx/modeling_mlx.py
import math
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        return (x * mx.rsqrt(variance + self.eps)).astype(x.dtype) * self.weight.astype(x.dtype)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        self.inv_freq = inv_freq

    def apply_rope(self, q: mx.array, k: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        freqs = mx.expand_dims(position_ids.astype(mx.float32), axis=-1) * self.inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.cos(emb), axis=1).astype(q.dtype)
        sin = mx.expand_dims(mx.sin(emb), axis=1).astype(q.dtype)

        def _rotate_half(x):
            half = self.dim // 2
            return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 16, num_kv_heads: int = 4, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, past_kv: tuple | None = None, position_ids: mx.array | None = None):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim

        if position_ids is None:
            past_len = past_kv[0].shape[-2] if past_kv is not None else 0
            position_ids = mx.arange(past_len, past_len + L)[None, :]

        q = mx.transpose(mx.reshape(self.W_q(x), (B, L, H_q, d_h)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(self.W_k(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))
        v = mx.transpose(mx.reshape(self.W_v(x), (B, L, H_kv, d_h)), (0, 2, 1, 3))

        q, k = self.rope.apply_rope(q, k, position_ids)

        if past_kv is not None:
            pk, pv = past_kv
            k = mx.concatenate([pk, k], axis=-2)
            v = mx.concatenate([pv, v], axis=-2)
        current_kv = (k, v)

        if H_kv != H_q:
            k = mx.repeat(k, self.num_queries_per_kv, axis=1)
            v = mx.repeat(v, self.num_queries_per_kv, axis=1)

        scale = 1.0 / math.sqrt(d_h)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale

        if past_kv is None and L > 1:
            indices = mx.arange(L)
            mask = indices[:, None] < indices[None, :]
            mask = mx.reshape(mask, (1, 1, L, L))
            scores = mx.where(mask, -1e9, scores)

        attn_weights = mx.softmax(scores, axis=-1)
        out = attn_weights @ v
        out_flat = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, L, D))
        return self.W_out(out_flat), current_kv


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = 4, max_seq_len: int = 4096):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, max_seq_len=max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5))

    def __call__(self, x: mx.array, past_kv: tuple | None = None, position_ids: mx.array | None = None):
        attn_out, current_kv = self.attn(self.ln1(x), past_kv=past_kv, position_ids=position_ids)
        x_out = x + attn_out
        x_out = x_out + self.mlp(self.ln2(x_out))
        return x_out, current_kv


class LowRankAssociativeDeltaEngine(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 16, chunk_size: int = 32, rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.r = rank
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head

        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_u = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_r = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_gate = nn.Linear(d_model, num_heads, bias=True)
        self.W_beta_gate = nn.Linear(d_model, num_heads, bias=True)
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward_sequence(self, x: mx.array):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        N = L // C

        Q = mx.transpose(mx.reshape(nn.silu(self.W_q(x)), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))
        K = mx.transpose(mx.reshape(nn.silu(self.W_k(x)), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))
        V = mx.transpose(mx.reshape(self.W_v(x), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))

        U = mx.transpose(mx.reshape(self.W_u(x), (B, N, C, H, r)), (0, 3, 1, 2, 4))
        R = mx.transpose(mx.reshape(self.W_r(x), (B, N, C, H, r)), (0, 3, 1, 2, 4))

        gate = mx.expand_dims(mx.transpose(mx.reshape(mx.clip(mx.sigmoid(self.W_gate(x)), 1e-3, 0.999), (B, N, C, H)), (0, 3, 1, 2)), axis=-1)
        beta_gate = mx.expand_dims(mx.transpose(mx.reshape(mx.sigmoid(self.W_beta_gate(x)), (B, N, C, H)), (0, 3, 1, 2)), axis=-1)

        log_gate = mx.log(gate)
        Lambda = mx.cumsum(log_gate, axis=-2)
        exp_Lambda = mx.exp(Lambda)

        indices = mx.arange(C)
        causal_mask = indices[:, None] >= indices[None, :]
        causal_mask = mx.reshape(causal_mask, (1, 1, 1, C, C))

        diff = Lambda - mx.transpose(Lambda, (0, 1, 2, 4, 3))
        M_links = mx.exp(mx.where(causal_mask, diff, -1e9))

        scaling = 1.0 / math.sqrt(d_h)
        Y_local = (Q @ mx.transpose(K, (0, 1, 2, 4, 3))) * scaling * M_links @ V

        chunk_decay_log = mx.squeeze(mx.sum(log_gate, axis=-2), axis=-1)
        Lambda_chunks = mx.cumsum(chunk_decay_log, axis=2)
        log_M_chunks = (mx.expand_dims(Lambda_chunks, axis=-1) - mx.expand_dims(Lambda_chunks, axis=-2)) - mx.expand_dims(chunk_decay_log, axis=-1)

        c_indices = mx.arange(N)
        causal_mask_chunks = c_indices[:, None] > c_indices[None, :]
        causal_mask_chunks = mx.reshape(causal_mask_chunks, (1, 1, N, N))
        M_chunks = mx.exp(mx.where(causal_mask_chunks, log_M_chunks, -1e9))

        U_decayed = (U * beta_gate) * (exp_Lambda[:, :, :, -1:, :] / mx.maximum(exp_Lambda, 1e-6))
        S_historical_flat = M_chunks @ mx.reshape(mx.transpose(U_decayed, (0, 1, 2, 4, 3)) @ V, (B, H, N, r * d_h))
        S_historical = mx.reshape(S_historical_flat, (B, H, N, r, d_h))

        Y_global = (R * exp_Lambda) @ S_historical * scaling
        Out = mx.reshape(mx.transpose(Y_local + Y_global, (0, 2, 3, 1, 4)), (B, L, self.inner_dim))

        chunk_decay_last = exp_Lambda[:, :, -1, -1:, :]
        S_historical_last = S_historical[:, :, -1]
        S_local_last = mx.transpose(U_decayed[:, :, -1], (0, 1, 3, 2)) @ V[:, :, -1]
        S_final = (chunk_decay_last * S_historical_last) + S_local_last

        return self.W_out(Out * nn.silu(self.W_swish_gate(x))), S_final

    def step_inference(self, x: mx.array, past_S: mx.array | None = None):
        B, L, D = x.shape
        H, d_h, r = self.num_heads, self.d_head, self.r
        scaling = 1.0 / math.sqrt(d_h)

        Q = mx.transpose(mx.reshape(nn.silu(self.W_q(x)), (B, L, H, d_h)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(nn.silu(self.W_k(x)), (B, L, H, d_h)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(self.W_v(x), (B, L, H, d_h)), (0, 2, 1, 3))
        U = mx.transpose(mx.reshape(self.W_u(x), (B, L, H, r)), (0, 2, 1, 3))
        R = mx.transpose(mx.reshape(self.W_r(x), (B, L, H, r)), (0, 2, 1, 3))

        gate = mx.expand_dims(mx.transpose(mx.reshape(mx.clip(mx.sigmoid(self.W_gate(x)), 1e-3, 0.999), (B, L, H)), (0, 2, 1)), axis=-1)
        beta_gate = mx.expand_dims(mx.transpose(mx.reshape(mx.sigmoid(self.W_beta_gate(x)), (B, L, H)), (0, 2, 1)), axis=-1)

        if past_S is None:
            past_S = mx.zeros((B, H, r, d_h), dtype=x.dtype)

        S_decayed = gate * past_S
        Y_local = Q @ (mx.transpose(K, (0, 1, 3, 2)) @ V) * scaling
        Y_global = R @ S_decayed * scaling
        Out = Y_local + Y_global

        S_local = mx.transpose(U * beta_gate, (0, 1, 3, 2)) @ V
        next_S = S_decayed + S_local

        out_flat = mx.reshape(mx.transpose(Out, (0, 2, 1, 3)), (B, L, self.inner_dim))
        return self.W_out(out_flat * nn.silu(self.W_swish_gate(x))), next_S


class LRADDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 32, rank: int = 8):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = LowRankAssociativeDeltaEngine(d_model, num_heads, chunk_size=chunk_size, rank=rank)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5))

    def __call__(self, x: mx.array, past_state: mx.array | None = None):
        h_attn = self.ln1(x)
        B, L, _ = x.shape
        is_step_inference = (past_state is not None) or (L == 1)

        if is_step_inference:
            attn_out, next_state = self.attn.step_inference(h_attn, past_S=past_state)
        else:
            attn_out, next_state = self.attn.forward_sequence(h_attn)

        x_out = x + attn_out
        x_out = x_out + self.mlp(self.ln2(x_out))
        return x_out, next_state


class BareTorchModelMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        layers = []
        for layer_type in config.layer_types:
            if layer_type == "transformer":
                block = TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len
                )
            elif layer_type == "cs_lrad":
                block = LRADDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    chunk_size=config.chunk_size,
                    rank=config.rank
                )
            else:
                raise ValueError(f"Unsupported MLX layer type '{layer_type}'")
            layers.append(block)

        self.layers = layers
        self.final_norm = RMSNorm(config.d_model)

    def __call__(self, input_ids: mx.array, past_key_values: list | None = None):
        B, L = input_ids.shape
        h = self.token_embedding(input_ids)
        next_cache = []

        is_step_inference = (past_key_values is not None) or (L == 1)
        pad_len = 0
        if not is_step_inference:
            chunk_size = self.config.chunk_size
            pad_len = (chunk_size - (L % chunk_size)) % chunk_size
            if pad_len > 0:
                h = mx.pad(h, [(0, 0), (0, pad_len), (0, 0)])

        for i, layer in enumerate(self.layers):
            past_state = past_key_values[i] if past_key_values is not None else None
            if isinstance(layer, TransformerDecoderBlock):
                h, next_state = layer(h, past_kv=past_state)
            else:
                h, next_state = layer(h, past_state=past_state)
            next_cache.append(next_state)

        h = self.final_norm(h)
        if not is_step_inference and pad_len > 0:
            h = h[:, :L, :]

        return h, next_cache


class BareTorchForCausalLMMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BareTorchModelMLX(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, past_key_values: list | None = None):
        hidden_states, next_cache = self.model(input_ids, past_key_values=past_key_values)
        logits = self.lm_head(hidden_states)
        return logits, next_cache