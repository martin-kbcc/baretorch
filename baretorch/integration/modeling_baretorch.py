import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

# Absolute imports of our modular layer-specific configs and architectures
from baretorch.integration.configuration_baretorch import (
    BareTorchConfig,
    CSLRADConfig,
    TransformerConfig,
    CSTTTConfig,
    CSLRADTransformerConfig,
    CSTTTTransformerConfig,
    CSLRADCSTTTTransformerConfig,
)
from baretorch.modeling.transformer import TransformerDecoderBlock, RMSNorm
from baretorch.modeling.cs_lrad import LRADDecoderBlock, CSLRADForCausalLM
from baretorch.modeling.cs_ttt import TTTDecoderBlock, CSTTTForCausalLM
from baretorch.modeling.transformer import TransformerForCausalLM
from baretorch.modeling.cs_lrad_transformer import CSLRADTransformerForCausalLM
from baretorch.modeling.cs_ttt_transformer import CSTTTTransformerForCausalLM
from baretorch.modeling.cs_lrad_cs_ttt_transformer import CSLRADCSTTTTransformerForCausalLM


# ==========================================
# 1. Master Dynamic Unified Model Model
# ==========================================

class BareTorchPreTrainedModel(PreTrainedModel):
    config_class = BareTorchConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _set_gradient_checkpointing(self, module, value=True, **kwargs):
        """
        Hugging Face registration hook to enable/disable gradient checkpointing 
        dynamically across BareTorch's custom layer architectures.
        """
        if isinstance(module, (BareTorchModel, BareTorchForCausalLM)):
            module.gradient_checkpointing = value
            if hasattr(module, "config"):
                module.config.use_grad_checkpointing = value
        if hasattr(module, "use_grad_checkpointing"):
            module.use_grad_checkpointing = value
            module.gradient_checkpointing = value

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        """
        Bypasses Hugging Face's default DynamicCache initialization to allow
        BareTorch's heterogeneous sequence mixers (Transformer, CS-LRAD, CS-TTT)
        to handle state caching using native tuple/tensor state lists.
        """
        if "past_key_values" not in model_kwargs:
            model_kwargs["past_key_values"] = None


class BareTorchModel(BareTorchPreTrainedModel):
    """
    The Dynamic Unified Sequence Engine of BareTorch.
    Assembles heterogeneous sequence mixers dynamically in runtime based on 
    the config's layer_types list, remaining 100% kernel-free and GEMM-compliant.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Assemble heterogeneous blocks on the fly based on configuration
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_type = config.layer_types[i]
            if layer_type == "transformer":
                block = TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    dropout=config.dropout,
                    max_seq_len=config.max_seq_len,
                    use_grad_checkpointing=config.use_grad_checkpointing
                )
            elif layer_type == "cs_lrad":
                block = LRADDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    chunk_size=config.chunk_size,
                    rank=config.rank,
                    dropout=config.dropout,
                    use_grad_checkpointing=config.use_grad_checkpointing
                )
            elif layer_type == "cs_ttt":
                block = TTTDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    chunk_size=config.chunk_size,
                    rank=config.rank,
                    dropout=config.dropout,
                    use_grad_checkpointing=config.use_grad_checkpointing
                )
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' at index {i}")
            self.layers.append(block)
            
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
        
        is_step_inference = (past_key_values is not None) or (seq_length == 1)
        
        # Determine dynamic block padding for sub-quadratic prefill loops
        pad_len = 0
        if not is_step_inference:
            chunk_size = self.config.chunk_size
            pad_len = (chunk_size - (seq_length % chunk_size)) % chunk_size
            if pad_len > 0:
                h = F.pad(h, (0, 0, 0, pad_len), value=0)
        
        effective_seq_len = seq_length + pad_len
        
        # Dynamically evaluate positions for Multi-Query/Grouped-Query Attention Blocks
        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                for cache in past_key_values:
                    if cache is not None and isinstance(cache, tuple): # Transformer KV cache
                        past_length = cache[0].size(-2)
                        break
            position_ids = torch.arange(
                past_length, past_length + effective_seq_len, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)
                
            past_state = past_key_values[i] if past_key_values is not None else None
            
            # Heterogeneous routing logic based on block architecture
            if isinstance(layer, TransformerDecoderBlock):
                h, next_state = layer(h, past_kv=past_state, position_ids=position_ids)
                # Trim KV cache vectors produced by chunk-padding during prefill
                if not is_step_inference and pad_len > 0 and next_state is not None:
                    k, v = next_state
                    next_state = (k[:, :, :seq_length, :], v[:, :, :seq_length, :])
            elif isinstance(layer, LRADDecoderBlock):
                h, next_state = layer(h, past_state=past_state, use_cache=use_cache)
            elif isinstance(layer, TTTDecoderBlock):
                h, next_state = layer(h, past_state=past_state, use_cache=use_cache)
                
            if use_cache:
                next_decoder_cache.append(next_state)

        h = self.final_norm(h)
        
        # Squeeze padding dimensions out of the final layer representations
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


class BareTorchForCausalLM(BareTorchPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = BareTorchModel(config)
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
        for i, layer_past in enumerate(past_key_values):
            if layer_past is None:
                reordered_past += (None,)
            elif isinstance(layer_past, tuple):  # Is a Transformer cache (k, v)
                k, v = layer_past
                reordered_past += ((k.index_select(0, beam_idx), v.index_select(0, beam_idx)),)
            else:  # Is a sub-quadratic state tensor (CS-LRAD or CS-TTT state matrix)
                reordered_past += (layer_past.index_select(0, beam_idx),)
        return reordered_past


# ==========================================
# 2. Global Hugging Face Model Registration
# ==========================================

AutoModel.register(BareTorchConfig, BareTorchModel)
AutoModelForCausalLM.register(BareTorchConfig, BareTorchForCausalLM)
AutoModelForCausalLM.register(CSLRADConfig, CSLRADForCausalLM)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)
AutoModelForCausalLM.register(CSTTTConfig, CSTTTForCausalLM)
AutoModelForCausalLM.register(CSLRADTransformerConfig, CSLRADTransformerForCausalLM)
AutoModelForCausalLM.register(CSTTTTransformerConfig, CSTTTTransformerForCausalLM)
AutoModelForCausalLM.register(CSLRADCSTTTTransformerConfig, CSLRADCSTTTTransformerForCausalLM)