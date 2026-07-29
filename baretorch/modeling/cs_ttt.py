import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


# ==========================================
# 1. Config & Core Stabilizer Blocks
# ==========================================

class CSTTTConfig(PretrainedConfig):
    model_type = "cs_ttt"
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


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.down_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate, x = self.gate_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * x))


# ==========================================
# 2. Stable Test-Time Training Engine (NLMS)
# ==========================================

class ChunkwiseTestTimeTrainingEngine(nn.Module):
    """
    Chunk-Segmented Test-Time Training (CS-TTT) Engine.
    Employs Normalized Least Mean Squares (NLMS) updates inside its sequential chunk
    forward loop to enforce unconditional mathematical convergence under any input scale.
    """
    def __init__(self, d_model, num_heads, chunk_size=32, rank=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = d_model // num_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Projection normalization layers
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.v_norm = RMSNorm(self.head_dim)
        
        # Learnable inner-loop learning rate
        self.ttt_lr = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, past_state=None, use_cache=False):
        B, L, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        C = self.chunk_size
        
        # 1. Project to Head Space
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)
        
        # 2. Regularize projection variances
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        
        is_step_inference = (past_state is not None) or (L == 1)
        
        # ----------------- RECURRENT DECODE PATH -----------------
        if is_step_inference:
            S = past_state if past_state is not None else torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
            
            # 1. Prediction using state S BEFORE current step update (matching prefill order)
            out = torch.matmul(q, S)
            out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            out = self.out_proj(out)
            
            # 2. Compute step gradient update (scaled by chunk_size C for prefill alignment)
            preds = torch.matmul(k, S)
            error = preds - v
            grad = torch.matmul(k.transpose(-1, -2), error) / C
            
            # 3. NLMS learning rate divisor
            k_norm_sq = torch.sum(k * k, dim=(-1, -2), keepdim=True)
            scaled_lr = torch.sigmoid(self.ttt_lr) * 1.0
            effective_lr = scaled_lr / (k_norm_sq + 1e-5)
            
            # 4. State update for next step
            next_S = S - effective_lr * grad
            
            return out, next_S
            
        # ----------------- PARALLEL PREFILL PATH -----------------
        else:
            pad_len = (C - (L % C)) % C
            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                
            total_len = L + pad_len
            num_chunks = total_len // C
            
            q_chunks = q.view(B, H, num_chunks, C, D)
            k_chunks = k.view(B, H, num_chunks, C, D)
            v_chunks = v.view(B, H, num_chunks, C, D)
            
            S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
            outputs_list = []
            scaled_lr = torch.sigmoid(self.ttt_lr) * 1.0
            
            for i in range(num_chunks):
                K_i = k_chunks[:, :, i]
                V_i = v_chunks[:, :, i]
                Q_i = q_chunks[:, :, i]
                
                out_i = torch.matmul(Q_i, S)
                outputs_list.append(out_i)
                
                preds = torch.matmul(K_i, S)
                error = preds - V_i
                grad = torch.matmul(K_i.transpose(-1, -2), error) / C
                
                k_norm_sq = torch.sum(K_i * K_i, dim=(-1, -2), keepdim=True)
                effective_lr = scaled_lr / (k_norm_sq + 1e-5)
                
                S = S - effective_lr * grad
                
            out = torch.stack(outputs_list, dim=2)
            out = out.view(B, H, total_len, D)
            
            if pad_len > 0:
                out = out[:, :, :L, :]
                
            out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            out = self.out_proj(out)
            return out, S


class TTTDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, rank=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.norm1 = RMSNorm(d_model)
        self.ttt = ChunkwiseTestTimeTrainingEngine(d_model, num_heads, chunk_size, rank, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, dropout)

    def forward(self, x, past_state=None, use_cache=False):
        residual = x
        normed_x = self.norm1(x)
        if self.use_grad_checkpointing and self.training:
            ttt_out, next_state = torch.utils.checkpoint.checkpoint(self.ttt, normed_x, past_state, use_cache)
        else:
            ttt_out, next_state = self.ttt(normed_x, past_state, use_cache)
        x = residual + ttt_out

        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x, next_state


class CSTTTPreTrainedModel(PreTrainedModel):
    config_class = CSTTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CSTTTModel(CSTTTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TTTDecoderBlock(
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

        h = self.drop(inputs_embeds)
        next_decoder_cache = [] if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)
                
            past_state = past_key_values[i] if past_key_values is not None else None
            h, next_state = layer(h, past_state=past_state, use_cache=use_cache)
                
            if use_cache:
                next_decoder_cache.append(next_state)

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


class CSTTTForCausalLM(CSTTTPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = CSTTTModel(config)
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