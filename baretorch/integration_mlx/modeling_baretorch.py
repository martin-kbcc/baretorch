# /home/martinkb/Desktop/BareTorch_F/baretorch/integration_mlx/modeling_baretorch.py
import gc
import glob
import json
import logging
import os
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.modeling_mlx.cs_lrad import LRADDecoderBlock
from baretorch.modeling_mlx.transformer import (
    InPlaceKVCache,
    RMSNorm,
    TransformerDecoderBlock,
    unwrap_kv_pair,
)

logger = logging.getLogger(__name__)


def extract_baretorch_cache_arrays(cache_list: list | None) -> list:
    """Extracts underlying mx.array buffers from both CS-LRAD tuples and InPlaceKVCache."""
    if not cache_list:
        return []
    arrays = []
    for item in cache_list:
        if isinstance(item, InPlaceKVCache):
            arrays.extend(item.state)
        elif isinstance(item, (tuple, list)):
            for x in item:
                if isinstance(x, mx.array):
                    arrays.append(x)
        elif isinstance(item, mx.array):
            arrays.append(item)
    return arrays


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
        self.prefill_chunk_size = getattr(config, "prefill_chunk_size", 4096)

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

    def fuse_weights(self):
        """Fuses linear projections across all 24 layers to minimize Metal kernel launch overhead."""
        for layer in self.layers:
            if hasattr(layer, "fuse_weights"):
                layer.fuse_weights()

    def _get_past_length(self, past_key_values: list | None) -> int:
        if past_key_values is None:
            return 0
        for idx, layer_past in enumerate(past_key_values):
            if isinstance(self.layers[idx], TransformerDecoderBlock):
                if isinstance(layer_past, InPlaceKVCache):
                    return layer_past.offset
                pk, _ = unwrap_kv_pair(layer_past)
                if pk is not None:
                    return pk.shape[-2]
        return 0

    def _forward_chunk(
        self,
        input_ids: mx.array,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = True,
    ) -> tuple[mx.array, list | None]:
        seq_length = input_ids.shape[1]
        h = self.token_embedding(input_ids)
        next_cache = [] if use_cache else None

        # Only allocate explicit position_ids when an attention_mask is present;
        # otherwise mx.fast.rope uses the integer offset directly with zero tensor allocation.
        if position_ids is None and attention_mask is not None:
            past_length = self._get_past_length(past_key_values)
            if past_length == 0:
                position_ids = mx.maximum(
                    mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1, 0
                )
            else:
                position_ids = mx.arange(
                    past_length, past_length + seq_length, dtype=mx.int32
                )[None, :]

        for i, layer in enumerate(self.layers):
            past_state = (
                past_key_values[i] if past_key_values is not None else None
            )

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

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = True,
        num_logits_to_keep: int = 0,
        prefill_chunk_size: int | None = None,
    ) -> tuple[mx.array, list | None]:
        seq_length = input_ids.shape[1]
        chunk_limit = (
            prefill_chunk_size
            if prefill_chunk_size is not None
            else self.prefill_chunk_size
        )

        # Automatic Chunked Prefill for long contexts (caps temporary activation memory)
        if use_cache and chunk_limit and seq_length > chunk_limit:
            past_length = self._get_past_length(past_key_values)
            full_position_ids = position_ids
            if full_position_ids is None and attention_mask is not None:
                if past_length == 0:
                    full_position_ids = mx.maximum(
                        mx.cumsum(attention_mask.astype(mx.int32), axis=-1) - 1, 0
                    )
                else:
                    full_position_ids = mx.arange(
                        past_length, past_length + seq_length, dtype=mx.int32
                    )[None, :]

            curr_cache = past_key_values
            keep_only_last_chunk = (
                num_logits_to_keep > 0 and num_logits_to_keep <= chunk_limit
            )
            collected_hidden = []
            last_h = None

            for start in range(0, seq_length, chunk_limit):
                end = min(start + chunk_limit, seq_length)
                chunk_ids = input_ids[:, start:end]
                chunk_pos = (
                    full_position_ids[:, start:end]
                    if full_position_ids is not None
                    else None
                )
                chunk_mask = (
                    attention_mask[:, start:end]
                    if attention_mask is not None
                    else None
                )

                last_h, curr_cache = self._forward_chunk(
                    input_ids=chunk_ids,
                    past_key_values=curr_cache,
                    attention_mask=chunk_mask,
                    position_ids=chunk_pos,
                    use_cache=True,
                )

                if not keep_only_last_chunk:
                    if num_logits_to_keep > 0 and end <= seq_length - num_logits_to_keep:
                        pass
                    else:
                        collected_hidden.append(last_h)

                # Evaluate once per 4,096-token chunk (matching mlx_lm) without stalling between layers
                if end < seq_length:
                    mx.eval(extract_baretorch_cache_arrays(curr_cache))

            if keep_only_last_chunk:
                return last_h, curr_cache
            return mx.concatenate(collected_hidden, axis=1), curr_cache

        return self._forward_chunk(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )


class BareTorchForCausalLMMLX(nn.Module):
    def __init__(self, config: BareTorchConfig):
        super().__init__()
        self.config = config
        self.model = BareTorchModelMLX(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def fuse_weights(self):
        self.model.fuse_weights()

    def __call__(
        self,
        input_ids: mx.array,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = True,
        num_logits_to_keep: int = 0,
        prefill_chunk_size: int | None = None,
    ) -> tuple[mx.array, list | None]:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            prefill_chunk_size=prefill_chunk_size,
        )

        if num_logits_to_keep > 0 and hidden_states.shape[1] > num_logits_to_keep:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]

        logits = self.lm_head(hidden_states)
        logits = mx.clip(logits, -50.0, 50.0)

        return logits, next_cache

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, dtype=mx.float16, fuse: bool = True):
        """
        Loads BareTorch PyTorch checkpoint weights into MLX architecture using CPU-stream
        batched conversion (preventing Metal GPU watchdog timeouts on 10GB FP32 files)
        and fuses projections for fast inference.
        """
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")

        with open(config_path, "r") as f:
            cfg_dict = json.load(f)

        config = BareTorchConfig(**cfg_dict)
        model = cls(config)

        raw_weights = {}
        sf_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        if sf_files:
            for sf_f in sf_files:
                logger.info(
                    f"Loading MLX weights from safetensors: {os.path.basename(sf_f)}"
                )
                raw_weights.update(mx.load(sf_f, stream=mx.cpu))
        else:
            bin_files = sorted(
                glob.glob(os.path.join(checkpoint_dir, "pytorch_model*.bin"))
            )
            if bin_files:
                import torch

                for bin_f in bin_files:
                    logger.info(
                        f"Loading MLX weights from pytorch bin: {os.path.basename(bin_f)}"
                    )
                    pt_state = torch.load(bin_f, map_location="cpu", weights_only=False)
                    for k, v in pt_state.items():
                        raw_weights[k] = mx.array(v.numpy())
            else:
                raise FileNotFoundError(
                    f"No .safetensors or pytorch_model*.bin weights found in {checkpoint_dir}"
                )

        cleaned_items = []
        batch_eval = []
        for k, v in raw_weights.items():
            clean_k = k
            for pfx in ["_fsdp_wrapped_module.", "module.", "_orig_mod."]:
                if clean_k.startswith(pfx):
                    clean_k = clean_k[len(pfx) :]

            if not clean_k.startswith("model.") and clean_k != "lm_head.weight":
                clean_k = f"model.{clean_k}"

            v_cast = mx.astype(v, dtype, stream=mx.cpu) if v.dtype != dtype else v
            cleaned_items.append((clean_k, v_cast))
            batch_eval.append(v_cast)

            if len(batch_eval) >= 32:
                mx.eval(batch_eval)
                batch_eval.clear()

        if batch_eval:
            mx.eval(batch_eval)
            batch_eval.clear()

        del raw_weights
        gc.collect()

        cleaned_keys = {k for k, _ in cleaned_items}
        if "lm_head.weight" not in cleaned_keys:
            for k, v in cleaned_items:
                if k in ("model.token_embedding.weight", "token_embedding.weight"):
                    cleaned_items.append(("lm_head.weight", v))
                    cleaned_keys.add("lm_head.weight")
                    break

        expected_keys = {
            k
            for k, _ in tree_flatten(model.parameters())
            if not k.endswith(".inv_freq")
        }
        missing_keys = expected_keys - cleaned_keys
        if missing_keys:
            raise RuntimeError(
                f"❌ Missing {len(missing_keys)} weight keys when loading MLX model: {sorted(list(missing_keys))[:5]}"
            )

        model.update(tree_unflatten(cleaned_items))
        del cleaned_items
        gc.collect()

        if fuse:
            model.fuse_weights()

        for layer in model.model.layers:
            mx.eval(layer.parameters())
            mx.synchronize()
        mx.eval(model.parameters())
        mx.synchronize()
        return model