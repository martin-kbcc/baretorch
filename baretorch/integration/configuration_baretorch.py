# /home/martinkb/Desktop/BareTorch_F/baretorch/integration/configuration_baretorch.py
from transformers import AutoConfig, PretrainedConfig

try:
    from baretorch.modeling.cs_lrad import CSLRADConfig
    from baretorch.modeling.transformer import TransformerConfig
except ImportError:
    from .cs_lrad import CSLRADConfig
    from .transformer import TransformerConfig


class BareTorchConfig(PretrainedConfig):
    """
    Unified Master Configuration for the BareTorch Framework.
    """

    model_type = "baretorch"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_heads=16,
        num_kv_heads=4,
        num_layers=12,
        chunk_size=32,
        rank=8,
        dropout=0.1,
        max_seq_len=4096,
        use_grad_checkpointing=False,
        use_qk_norm=False,
        rope_theta=10000.0,
        layer_types=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        auto_map=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
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
        self.use_qk_norm = use_qk_norm
        self.rope_theta = rope_theta
        self.use_cache = kwargs.get("use_cache", True)

        self.auto_map = (
            auto_map
            if auto_map is not None
            else {
                "AutoConfig": "configuration_baretorch.BareTorchConfig",
                "AutoModel": "modeling_baretorch.BareTorchModel",
                "AutoModelForCausalLM": "modeling_baretorch.BareTorchForCausalLM",
            }
        )

        if layer_types is None:
            self.layer_types = []
            for i in range(num_layers):
                if (i + 1) % 4 == 0:
                    self.layer_types.append("transformer")
                else:
                    self.layer_types.append("cs_lrad")
        else:
            self.layer_types = layer_types


try:
    AutoConfig.register("baretorch", BareTorchConfig, exist_ok=True)
    AutoConfig.register("cs_lrad", CSLRADConfig, exist_ok=True)
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
except TypeError:
    # Fallback for older transformers versions without exist_ok
    try:
        AutoConfig.register("baretorch", BareTorchConfig)
    except ValueError:
        pass