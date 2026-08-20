import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

# ==========================================
# 1. Pure GEMM-Compliant Core Utilities
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ==========================================
# 2. Pure PyTorch SOTA Attention & RoPE Modules
# ==========================================

class RotaryEmbedding(nn.Module):
    """
    Export-friendly, dynamic Rotary Position Embeddings (RoPE).
    Pre-computes static buffers for graph export, but dynamically extends cache 
    if prompt + generated tokens exceed initial buffer bounds during runtime generation.
    """
    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute static lookup table up to default max_position_embeddings
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device="cpu", dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        
        # Handle meta tensor initialization from Hugging Face from_pretrained
        if getattr(self.inv_freq, "is_meta", False) or self.inv_freq.device.type == "meta":
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            )
        else:
            inv_freq = self.inv_freq.to(device)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, q, k, position_ids):
        # Bypass .item() evaluation during torch.export / ExecuTorch tracing
        if not torch.compiler.is_compiling():
            try:
                max_pos = int(position_ids.max().item()) if position_ids.numel() > 0 else 0
                if (
                    self.cos_cached is None 
                    or max_pos >= self.cos_cached.size(0) 
                    or self.cos_cached.device != q.device
                    or self.cos_cached.dtype != q.dtype
                ):
                    new_len = max(max_pos + 1024, self.max_position_embeddings)
                    self._set_cos_sin_cache(seq_len=new_len, device=q.device, dtype=q.dtype)
            except Exception:
                pass

        position_ids = position_ids.to(q.device)
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, L, d_h]
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [B, 1, L, d_h]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    SOTA Grouped-Query Attention (GQA) with fused Rotary Position Embeddings (RoPE).
    Utilizes PyTorch's native hardware-accelerated Scaled Dot-Product Attention (SDPA).
    """
    def __init__(self, d_model, num_heads=16, num_kv_heads=4, dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        assert num_heads % num_kv_heads == 0, "Query heads must be perfectly divisible by KV heads."
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len)
        
        self.dropout_p = dropout
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, position_ids=None):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim
        
        if position_ids is None:
            past_len = past_kv[0].size(-2) if past_kv is not None else 0
            position_ids = torch.arange(past_len, past_len + L, dtype=torch.long, device=x.device).unsqueeze(0)
            
        q = self.W_q(x).view(B, L, H_q, d_h).transpose(1, 2)
        k = self.W_k(x).view(B, L, H_kv, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, L, H_kv, d_h).transpose(1, 2)
        
        q, k = self.rope.apply_rope(q, k, position_ids)
        
        if past_kv is not None:
            pk, pv = past_kv
            k, v = torch.cat([pk, k], dim=-2), torch.cat([pv, v], dim=-2)
        current_kv = (k, v)
        
        if H_kv != H_q:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1).contiguous()
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1).contiguous()
            
        is_causal_mask = (past_kv is None)
        
        # Dynamically evaluate dropout based on training mode
        dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal_mask
        )
        
        out_flat = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.W_out(out_flat)), current_kv


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=4, dropout=0.1, max_seq_len=4096, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, dropout, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x, past_kv=None, position_ids=None):
        def _block_forward(x_in, p_kv, pos_ids):
            attn_out, current_kv = self.attn(self.ln1(x_in), past_kv=p_kv, position_ids=pos_ids)
            x_out = x_in + attn_out
            x_out = x_out + self.mlp(self.ln2(x_out))
            return x_out, current_kv
        
        if self.use_grad_checkpointing and self.training:
            return checkpoint.checkpoint(_block_forward, x, past_kv, position_ids, use_reentrant=False)
        else:
            return _block_forward(x, past_kv, position_ids)


# ==========================================
# 3. Hugging Face Serialization Configuration & Wrappers
# ==========================================

class TransformerConfig(PretrainedConfig):
    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_kv_heads=4,
        num_layers=8,
        dropout=0.1,
        max_seq_len=4096,
        use_grad_checkpointing=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing


class TransformerPreTrainedModel(PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_grad_checkpointing=config.use_grad_checkpointing
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        h = self.drop(inputs_embeds)
        next_decoder_cache = [] if use_cache else None
        
        if position_ids is None:
            past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)
                
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, current_kv = layer(h, past_kv=past_kv, position_ids=position_ids)
                
            if use_cache:
                next_decoder_cache.append(current_kv)

        h = self.final_norm(h)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (h,)

        if not return_dict:
            return tuple(v for v in [h, next_decoder_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class TransformerForCausalLM(TransformerPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past += (None,)
            else:
                k, v = layer_past
                reordered_past += ((k.index_select(0, beam_idx), v.index_select(0, beam_idx)),)
        return reordered_past