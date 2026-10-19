# baretorch/integration_mlx/modeling_baretorch.py
import glob
import json
import logging
import os
import mlx.core as mx
import mlx.nn as nn

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.modeling_mlx.cs_lrad import LRADDecoderBlock
from baretorch.modeling_mlx.transformer import RMSNorm, TransformerDecoderBlock

logger = logging.getLogger(__name__)


class CausalLMOutput:
    """Provides dual access: tuple unpacking (logits, cache) and attribute access (.logits, .past_key_values)."""

    def __init__(self, logits: mx.array, past_key_values: list | None = None):
        self.logits = logits
        self.past_key_values = past_key_values

    def __iter__(self):
        yield self.logits
        yield self.past_key_values

    def __getitem__(self, index):
        return (self.logits, self.past_key_values)[index]


class BareTorchModelMLX(nn.Module):
    def __init__(self, config: BareTorchConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        layers = []
        for i in range(config.num_layers):
            layer_type = config.layer_types[i]
            if layer_type == "transformer":
                block = TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    max_seq_len=config.max_seq_len,
                    use_qk_norm=getattr(config, "use_qk_norm", False),
                    rope_theta=getattr(config, "rope_theta", 10000.0),
                )
            elif layer_type == "cs_lrad":
                block = LRADDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    chunk_size=config.chunk_size,
                    rank=config.rank,
                )
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' at index {i}")
            layers.append(block)

        self.layers = layers
        self.final_norm = RMSNorm(config.d_model)

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = True,
    ):
        # Defensive slicing matching PyTorch forward when past_key_values is present
        if past_key_values is not None and input_ids.shape[1] > 1:
            input_ids = input_ids[:, -1:]

        h = self.token_embedding(input_ids)
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_state = past_key_values[i] if past_key_values is not None else None

            if isinstance(layer, TransformerDecoderBlock):
                h, next_state = layer(
                    h,
                    past_kv=past_state,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            else:
                h, next_state = layer(
                    h,
                    past_state=past_state,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )

            if use_cache:
                next_cache.append(next_state)

        h = self.final_norm(h)
        return h, next_cache


class BareTorchForCausalLMMLX(nn.Module):
    def __init__(self, config: BareTorchConfig):
        super().__init__()
        self.config = config
        self.model = BareTorchModelMLX(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = True,
        num_logits_to_keep: int = 0,
    ) -> tuple[mx.array, list | None]:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )

        if num_logits_to_keep > 0 and hidden_states.shape[1] > num_logits_to_keep:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]

        logits = self.lm_head(hidden_states)
        logits = mx.clip(logits, -50.0, 50.0)

        # Return pure tuple so MLX compile and eval can track all arrays
        return logits, next_cache




    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, dtype=mx.float16):
        """Loads BareTorch PyTorch checkpoint weights directly into MLX architecture.
        
        Handles single-file safetensors, sharded safetensors, and pytorch_model.bin.
        """
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")

        with open(config_path, "r") as f:
            cfg_dict = json.load(f)

        config = BareTorchConfig(**cfg_dict)
        model = cls(config)

        # 1. Gather all weight files (Safetensors or PyTorch Bin)
        raw_weights = {}
        sf_index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        sf_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        if sf_files:
            for sf_f in sf_files:
                logger.info(f"Loading MLX weights from safetensors: {os.path.basename(sf_f)}")
                raw_weights.update(mx.load(sf_f))
        else:
            bin_index_path = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
            bin_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.bin")))
            if bin_files:
                import torch
                for bin_f in bin_files:
                    logger.info(f"Loading MLX weights from pytorch bin: {os.path.basename(bin_f)}")
                    pt_state = torch.load(bin_f, map_location="cpu", weights_only=False)
                    for k, v in pt_state.items():
                        raw_weights[k] = mx.array(v.numpy())
            else:
                raise FileNotFoundError(f"No .safetensors or .bin weights found in {checkpoint_dir}")

        # 2. Normalize prefixes (FSDP, DDP, torch.compile)
        cleaned_weights = {}
        for k, v in raw_weights.items():
            clean_k = k
            for pfx in ["_fsdp_wrapped_module.", "module.", "_orig_mod."]:
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx):]

            # Ensure keys map cleanly to self.model if saved as base model
            if not clean_k.startswith("model.") and clean_k != "lm_head.weight":
                clean_k = f"model.{clean_k}"

            cleaned_weights[clean_k] = v.astype(dtype)

        # 3. Handle Tied Word Embeddings
        if "lm_head.weight" not in cleaned_weights:
            for cand in ["model.token_embedding.weight", "token_embedding.weight"]:
                if cand in cleaned_weights:
                    cleaned_weights["lm_head.weight"] = cleaned_weights[cand]
                    break

        # 4. Load weights into the MLX model structure
        model.load_weights(list(cleaned_weights.items()), strict=False)
        return model