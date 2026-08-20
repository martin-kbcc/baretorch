import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


def safe_coreml_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes cumulative sum safely for CoreML export by temporarily flattening 
    tensors with rank > 4 down to 3D. Prevents coremltools MIL 5D operator failure 
    without altering parameters or numerical outputs.
    """
    orig_shape = x.shape
    if dim < 0:
        dim = len(orig_shape) + dim

    if x.dim() > 4:
        pre_dim_size = 1
        for s in orig_shape[:dim]:
            pre_dim_size *= s

        post_dim_size = 1
        for s in orig_shape[dim + 1:]:
            post_dim_size *= s

        target_dim_size = orig_shape[dim]

        x_3d = x.reshape(pre_dim_size, target_dim_size, post_dim_size)
        out_3d = torch.cumsum(x_3d, dim=1)
        return out_3d.reshape(orig_shape)
    else:
        return torch.cumsum(x, dim=dim)


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
# 2. Pure PyTorch BareTorch Sequence Mixer
# ==========================================

class LowRankAssociativeDeltaEngine(nn.Module):
    def __init__(self, d_model=256, num_heads=16, chunk_size=32, rank=8):
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

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        N = L // C  
        
        Q = F.silu(self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K = F.silu(self.W_k(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        V = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        U = self.W_u(x).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)   
        R = self.W_r(x).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)   
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        beta_gate = torch.sigmoid(self.W_beta_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
        
        log_gate = torch.log(gate)
        Lambda = safe_coreml_cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)  
        
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        M_links = torch.exp((Lambda - Lambda.transpose(-1, -2)).masked_fill(causal_mask == 0, float('-inf')))
        
        scaling = 1.0 / math.sqrt(d_h)
        Y_local = torch.matmul(torch.matmul(Q, K.transpose(-1, -2)) * scaling * M_links, V)  
        
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = safe_coreml_cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, float('-inf')))
        
        U_decayed = (U * beta_gate) * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        S_historical = torch.matmul(M_chunks, torch.matmul(U_decayed.transpose(-1, -2), V).view(B, H, N, r * d_h)).view(B, H, N, r, d_h)
        
        Y_global = torch.matmul(R * exp_Lambda, S_historical) * scaling  
        
        Out = (Y_local + Y_global).permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        
        # --- Hugging Face Cache Optimization: S_final (State at step L) ---
        chunk_decay_last = exp_Lambda[:, :, -1, -1:, :] 
        S_historical_last = S_historical[:, :, -1]      
        S_local_last = torch.matmul(U_decayed[:, :, -1].transpose(-1, -2), V[:, :, -1]) 
        S_final = (chunk_decay_last * S_historical_last) + S_local_last
        
        return self.W_out(Out * F.silu(self.W_swish_gate(x))), S_final

    def step_inference(self, x, past_S=None):
        B, L, D = x.shape
        H, d_h, r = self.num_heads, self.d_head, self.r
        scaling = 1.0 / math.sqrt(d_h)
        
        Q = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K = F.silu(self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        U, R = self.W_u(x).view(B, L, H, r).permute(0, 2, 1, 3), self.W_r(x).view(B, L, H, r).permute(0, 2, 1, 3)
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        beta_gate = torch.sigmoid(self.W_beta_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1)
        
        past_S = past_S if past_S is not None else torch.zeros(B, H, r, d_h, device=x.device, dtype=x.dtype)
        
        # 1. Decay past historical state for the current step
        S_decayed = gate * past_S
        
        # 2. Local step attention + global historical attention (read BEFORE local state write)
        Y_local = torch.matmul(Q, torch.matmul(K.transpose(-1, -2), V)) * scaling
        Y_global = torch.matmul(R, S_decayed) * scaling
        Out = Y_local + Y_global
        
        # 3. Update state with current token's associative delta for future steps
        S_local = torch.matmul((U * beta_gate).transpose(-1, -2), V)
        next_S = S_decayed + S_local
        
        return self.W_out(Out.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim) * F.silu(self.W_swish_gate(x))), next_S


class LRADDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, rank=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = LowRankAssociativeDeltaEngine(d_model, num_heads, chunk_size=chunk_size, rank=rank)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x, past_state=None, use_cache=False):
        def _block_forward(x_in, p_state):
            h_attn = self.ln1(x_in)
            B, L, _ = x_in.shape
            
            is_step_inference = (p_state is not None) or (L == 1)
            
            if is_step_inference:
                attn_out, next_state = self.attn.step_inference(h_attn, past_S=p_state)
            else:
                attn_out, next_state = self.attn(h_attn)
                
            x_out = x_in + attn_out
            x_out = x_out + self.mlp(self.ln2(x_out))
            return x_out, next_state
        
        if self.use_grad_checkpointing and self.training:
            return checkpoint.checkpoint(_block_forward, x, past_state, use_reentrant=False)
        else:
            return _block_forward(x, past_state)


class CSLRADConfig(PretrainedConfig):
    model_type = "cs_lrad"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_layers=8,
        chunk_size=32,
        rank=8,
        dropout=0.1,
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
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.rank = rank
        self.dropout = dropout
        self.use_grad_checkpointing = use_grad_checkpointing


class CSLRADPreTrainedModel(PreTrainedModel):
    config_class = CSLRADConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CSLRADModel(CSLRADPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            LRADDecoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                chunk_size=config.chunk_size,
                rank=config.rank,
                dropout=config.dropout,
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

        h = inputs_embeds
        next_decoder_cache = [] if use_cache else None
        
        is_step_inference = (past_key_values is not None) or (seq_length == 1)
        
        pad_len = 0
        if not is_step_inference:
            chunk_size = self.config.chunk_size
            pad_len = (chunk_size - (seq_length % chunk_size)) % chunk_size
            if pad_len > 0:
                h = F.pad(h, (0, 0, 0, pad_len), value=0)
        
        all_hidden_states = () if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)
                
            past_state = past_key_values[i] if past_key_values is not None else None
            h, next_state = layer(h, past_state=past_state, use_cache=use_cache)
                
            if use_cache:
                next_decoder_cache.append(next_state)

        h = self.final_norm(h)
        
        if not is_step_inference and pad_len > 0:
            h = h[:, :seq_length, :]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (h,)

        if not return_dict:
            return tuple(v for v in [h, next_decoder_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class CSLRADForCausalLM(CSLRADPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = CSLRADModel(config)
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
                reordered_past += (layer_past.index_select(0, beam_idx),)
        return reordered_past