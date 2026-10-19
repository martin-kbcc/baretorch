# baretorch/modeling_mlx/cs_lrad.py
import math
import mlx.core as mx
import mlx.nn as nn
from baretorch.modeling_mlx.transformer import GatedMLP, RMSNorm


class LowRankAssociativeDeltaEngine(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 16, chunk_size: int = 32, rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.r = rank
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head

        # Unfused parameter projections matching PyTorch weights 1-to-1
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_u = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_r = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_gate = nn.Linear(d_model, self.num_heads, bias=True)
        self.W_beta_gate = nn.Linear(d_model, self.num_heads, bias=True)
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def chunk_forward(self, x: mx.array, attention_mask: mx.array | None = None, use_cache: bool = True):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        scaling = 1.0 / math.sqrt(d_h)

        N = (L + C - 1) // C
        pad_len = (C - (L % C)) % C

        if pad_len > 0:
            x_padded = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])
        else:
            x_padded = x

        if attention_mask is not None:
            if pad_len > 0:
                mask_padded = mx.pad(attention_mask, [(0, 0), (0, pad_len)])
            else:
                mask_padded = attention_mask
            mask_5d = mx.expand_dims(
                mx.expand_dims(mx.reshape(mask_padded, (B, N, C)), axis=1), axis=-1
            ).astype(mx.float32)
        else:
            seq_idx = mx.arange(N * C)
            mask_padded = mx.reshape((seq_idx < L).astype(mx.float32), (1, N, C))
            mask_5d = mx.expand_dims(mx.expand_dims(mask_padded, axis=1), axis=-1)

        Q = mx.transpose(mx.reshape(nn.silu(self.W_q(x_padded)), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))
        K = mx.transpose(mx.reshape(nn.silu(self.W_k(x_padded)), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))
        V = mx.transpose(mx.reshape(self.W_v(x_padded), (B, N, C, H, d_h)), (0, 3, 1, 2, 4))
        U = mx.transpose(mx.reshape(self.W_u(x_padded), (B, N, C, H, r)), (0, 3, 1, 2, 4))
        R = mx.transpose(mx.reshape(self.W_r(x_padded), (B, N, C, H, r)), (0, 3, 1, 2, 4))

        gate_fp32 = mx.expand_dims(
            mx.transpose(
                mx.reshape(
                    mx.clip(mx.sigmoid(self.W_gate(x_padded).astype(mx.float32)), 1e-4, 0.9999),
                    (B, N, C, H),
                ),
                (0, 3, 1, 2),
            ),
            axis=-1,
        )
        beta_gate_fp32 = mx.expand_dims(
            mx.transpose(
                mx.reshape(mx.sigmoid(self.W_beta_gate(x_padded).astype(mx.float32)), (B, N, C, H)),
                (0, 3, 1, 2),
            ),
            axis=-1,
        )

        U_fp32 = U.astype(mx.float32) * mask_5d
        V_fp32 = V.astype(mx.float32) * mask_5d
        gate_fp32 = mx.where(mask_5d.astype(mx.bool_), gate_fp32, 1.0)

        log_gate = mx.log(gate_fp32)
        Lambda = mx.cumsum(log_gate, axis=-2)
        exp_Lambda = mx.exp(mx.clip(Lambda, -30.0, 0.0))

        indices = mx.arange(C)
        causal_mask = mx.reshape(indices[:, None] >= indices[None, :], (1, 1, 1, C, C))
        diff = Lambda - mx.transpose(Lambda, (0, 1, 2, 4, 3))
        diff_masked = mx.where(causal_mask, diff, -1e4)
        M_links = mx.exp(mx.clip(diff_masked, -1e4, 0.0))

        Y_local = (
            (Q.astype(mx.float32) @ mx.transpose(K.astype(mx.float32), (0, 1, 2, 4, 3)))
            * scaling
            * M_links
            @ V_fp32
        )

        chunk_decay_log = mx.clip(mx.squeeze(mx.sum(log_gate, axis=-2), axis=-1), -30.0, 0.0)
        Lambda_chunks = mx.cumsum(chunk_decay_log, axis=2)
        log_M_chunks = (
            mx.expand_dims(Lambda_chunks, axis=-1) - mx.expand_dims(Lambda_chunks, axis=-2)
        ) - mx.expand_dims(chunk_decay_log, axis=-1)

        c_indices = mx.arange(N)
        causal_mask_chunks = mx.reshape(c_indices[:, None] > c_indices[None, :], (1, 1, N, N))
        M_chunks = mx.exp(mx.clip(mx.where(causal_mask_chunks, log_M_chunks, -1e4), -1e4, 0.0))

        decay = mx.exp(mx.clip(Lambda[:, :, :, -1:, :] - Lambda, -30.0, 0.0))
        U_decayed = (U_fp32 * beta_gate_fp32) * decay

        S_historical_flat = M_chunks @ mx.reshape(
            mx.transpose(U_decayed, (0, 1, 2, 4, 3)) @ V_fp32, (B, H, N, r * d_h)
        )
        S_historical = mx.reshape(S_historical_flat, (B, H, N, r, d_h))
        Y_global = (R.astype(mx.float32) * exp_Lambda) @ S_historical * scaling

        Out = mx.reshape(
            mx.transpose(Y_local + Y_global, (0, 2, 3, 1, 4)), (B, N * C, self.inner_dim)
        ).astype(x.dtype)
        Out = Out[:, :L, :]

        output = self.W_out(Out * nn.silu(self.W_swish_gate(x)))

        next_cache = None
        if use_cache:
            rem = L % C
            if rem == 0:
                chunk_decay_last = exp_Lambda[:, :, -1, -1:, :]
                S_historical_last = S_historical[:, :, -1]
                S_local_last = mx.transpose(U_decayed[:, :, -1], (0, 1, 3, 2)) @ V_fp32[:, :, -1]
                S_final = mx.clip((chunk_decay_last * S_historical_last) + S_local_last, -50.0, 50.0)
                next_cache = (S_final, None, None, None, None, L)
            else:
                past_S = S_historical[:, :, -1] if N > 1 else mx.zeros((B, H, r, d_h), dtype=mx.float32)
                buf_k = K[:, :, -1, :rem, :].astype(mx.float32)
                buf_v = V_fp32[:, :, -1, :rem, :]
                buf_u = U_fp32[:, :, -1, :rem, :] * beta_gate_fp32[:, :, -1, :rem, :]
                buf_lambda = Lambda[:, :, -1, :rem, :]
                next_cache = (past_S, buf_k, buf_v, buf_u, buf_lambda, L)

        return output, next_cache

    def step_forward(
        self,
        x: mx.array,
        past_state: tuple | mx.array | None = None,
        use_cache: bool = True,
        attention_mask: mx.array | None = None,
    ):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        scaling = 1.0 / math.sqrt(d_h)

        past_S, buf_k, buf_v, buf_u, buf_lambda = None, None, None, None, None
        curr_seq_len = 0

        if past_state is not None:
            if isinstance(past_state, (tuple, list)):
                if len(past_state) >= 1 and past_state[0] is not None:
                    past_S = past_state[0]
                if len(past_state) >= 5:
                    buf_k, buf_v, buf_u, buf_lambda = past_state[1], past_state[2], past_state[3], past_state[4]
                if len(past_state) >= 6 and isinstance(past_state[5], int):
                    curr_seq_len = past_state[5]
            elif isinstance(past_state, mx.array):
                past_S = past_state

        if past_S is None:
            past_S = mx.zeros((B, H, r, d_h), dtype=mx.float32)
        else:
            past_S = past_S.astype(mx.float32)

        if L > 1:
            return self.chunk_forward(x, attention_mask=attention_mask, use_cache=use_cache)

        Q = mx.transpose(mx.reshape(nn.silu(self.W_q(x)), (B, L, H, d_h)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(nn.silu(self.W_k(x)), (B, L, H, d_h)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(self.W_v(x)), (B, L, H, d_h)), (0, 2, 1, 3))
        U = mx.transpose(mx.reshape(self.W_u(x)), (B, L, H, r)), (0, 2, 1, 3))
        R = mx.transpose(mx.reshape(self.W_r(x)), (B, L, H, r)), (0, 2, 1, 3))

        gate = mx.expand_dims(
            mx.transpose(
                mx.reshape(
                    mx.clip(mx.sigmoid(self.W_gate(x).astype(mx.float32)), 1e-4, 0.9999), (B, L, H)
                ),
                (0, 2, 1),
            ),
            axis=-1,
        )
        beta_gate = mx.expand_dims(
            mx.transpose(
                mx.reshape(mx.sigmoid(self.W_beta_gate(x).astype(mx.float32)), (B, L, H)),
                (0, 2, 1),
            ),
            axis=-1,
        )
        log_gate = mx.log(gate)

        if buf_lambda is not None:
            last_lambda = buf_lambda[:, :, -1:, :]
            lambda_curr = last_lambda + log_gate
            all_lambda = mx.concatenate([buf_lambda, lambda_curr], axis=2)
            all_k = mx.concatenate([buf_k, K.astype(mx.float32)], axis=2)
            all_v = mx.concatenate([buf_v, V.astype(mx.float32)], axis=2)
            all_u = mx.concatenate([buf_u, (U.astype(mx.float32) * beta_gate)], axis=2)
        else:
            lambda_curr = log_gate
            all_lambda = lambda_curr
            all_k = K.astype(mx.float32)
            all_v = V.astype(mx.float32)
            all_u = U.astype(mx.float32) * beta_gate

        exp_lambda_curr = mx.exp(mx.clip(lambda_curr, -30.0, 0.0))
        diff = lambda_curr - mx.transpose(all_lambda, (0, 1, 3, 2))
        m_links = mx.exp(mx.clip(diff, -1e4, 0.0))
        attn_weights = (Q.astype(mx.float32) @ mx.transpose(all_k, (0, 1, 3, 2))) * scaling * m_links
        Y_local = attn_weights @ all_v

        Y_global = (R.astype(mx.float32) * exp_lambda_curr) @ past_S * scaling

        Out = mx.reshape(
            mx.transpose(Y_local + Y_global, (0, 2, 1, 3)), (B, 1, self.inner_dim)
        ).astype(x.dtype)
        output = self.W_out(Out * nn.silu(self.W_swish_gate(x)))

        curr_c = all_k.shape[2]
        if curr_c >= C:
            chunk_decay = mx.exp(mx.clip(all_lambda[:, :, -1:, :], -30.0, 0.0))
            intra_decay = mx.exp(mx.clip(all_lambda[:, :, -1:, :] - all_lambda, -30.0, 0.0))
            u_decayed = all_u * intra_decay
            s_local = mx.transpose(u_decayed, (0, 1, 3, 2)) @ all_v
            past_S = mx.clip((chunk_decay * past_S) + s_local, -50.0, 50.0)
            buf_k, buf_v, buf_u, buf_lambda = None, None, None, None
        else:
            buf_k, buf_v, buf_u, buf_lambda = all_k, all_v, all_u, all_lambda

        next_cache = (past_S, buf_k, buf_v, buf_u, buf_lambda, curr_seq_len + 1)
        return output, next_cache

    def __call__(
        self,
        x: mx.array,
        past_state: tuple | mx.array | None = None,
        use_cache: bool = True,
        attention_mask: mx.array | None = None,
    ):
        if past_state is not None or (x.shape[1] == 1 and use_cache):
            return self.step_forward(
                x, past_state=past_state, use_cache=use_cache, attention_mask=attention_mask
            )
        return self.chunk_forward(x, attention_mask=attention_mask, use_cache=use_cache)


class LRADDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 32, rank: int = 8):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = LowRankAssociativeDeltaEngine(d_model, num_heads, chunk_size=chunk_size, rank=rank)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5))

    def __call__(
        self,
        x: mx.array,
        past_state: tuple | mx.array | None = None,
        use_cache: bool = True,
        attention_mask: mx.array | None = None,
    ):
        h_attn = self.ln1(x)
        attn_out, next_state = self.attn(
            h_attn, past_state=past_state, use_cache=use_cache, attention_mask=attention_mask
        )
        x_out = x + attn_out
        x_out = x_out + self.mlp(self.ln2(x_out))
        return x_out, next_state