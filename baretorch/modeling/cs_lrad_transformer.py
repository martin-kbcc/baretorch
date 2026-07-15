import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

# Direct imports of your native mathematical blocks to maintain zero code duplication
from baretorch.modeling.transformer import TransformerDecoderBlock, RMSNorm
from baretorch.modeling.cs_lrad import LRADDecoderBlock


# ==========================================
# 1. Hugging Face Hybrid Configuration
# ==========================================

class CSLRADTransformerConfig(PretrainedConfig):
    model_type = "cs_lrad_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_kv_heads=4,
        num_layers=8,
        chunk_size=32,
        rank=8,
        dropout=0.1,
        max_seq_len=4096,
        use_grad_checkpointing=False,
        layer_types=None,  # Optional list of strings specifying layer order, e.g. ["transformer", "cs_lrad", ...]
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
        self.chunk_size = chunk_size
        self.rank = rank
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing
        
        # Default to an alternating layout if not explicitly defined
        if layer_types is None:
            self.layer_types = []
            for i in range(num_layers):
                if i % 2 == 0:
                    self.layer_types.append("transformer")
                else:
                    self.layer_types.append("cs_lrad")
        else:
            self.layer_types = layer_types


# ==========================================
# 2. Hybrid Model Implementations
# ==========================================

class CSLRADTransformerPreTrainedModel(PreTrainedModel):
    config_class = CSLRADTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CSLRADTransformerModel(CSLRADTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Modular assembly based on specified layer types
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
        
        # Heterogeneous sequence-mixing modes detection
        is_step_inference = (past_key_values is not None) or (seq_length == 1)
        
        # Apply padding dynamically for the CS-LRAD parallel prefill paths
        pad_len = 0
        if not is_step_inference:
            chunk_size = self.config.chunk_size
            pad_len = (chunk_size - (seq_length % chunk_size)) % chunk_size
            if pad_len > 0:
                h = F.pad(h, (0, 0, 0, pad_len), value=0)
        
        effective_seq_len = seq_length + pad_len
        
        # Calculate Position IDs dynamically for Transformer RoPE tracking
        if position_ids is None:
            past_length = 0
            if past_key_values is not None:
                for cache in past_key_values:
                    if cache is not None and isinstance(cache, tuple): # Is a Transformer cache
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
            
            # Heterogeneous input & cache forwarding
            if isinstance(layer, TransformerDecoderBlock):
                h, next_state = layer(h, past_kv=past_state, position_ids=position_ids)
            elif isinstance(layer, LRADDecoderBlock):
                h, next_state = layer(h, past_state=past_state, use_cache=use_cache)
                
            if use_cache:
                next_decoder_cache.append(next_state)

        h = self.final_norm(h)
        
        # Slice original length out of the padded sequence
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


class CSLRADTransformerForCausalLM(CSLRADTransformerPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.token_embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = CSLRADTransformerModel(config)
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
            else:  # Is a CS-LRAD state tensor (S_state)
                reordered_past += (layer_past.index_select(0, beam_idx),)
        return reordered_past