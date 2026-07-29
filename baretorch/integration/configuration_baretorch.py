from transformers import PretrainedConfig, AutoConfig

# Direct absolute imports of our modular layer-specific configurations to maintain registry coverage
from baretorch.modeling.cs_lrad import CSLRADConfig
from baretorch.modeling.transformer import TransformerConfig
from baretorch.modeling.cs_ttt import CSTTTConfig
from baretorch.modeling.cs_lrad_transformer import CSLRADTransformerConfig
from baretorch.modeling.cs_ttt_transformer import CSTTTTransformerConfig
from baretorch.modeling.cs_lrad_cs_ttt_transformer import CSLRADCSTTTTransformerConfig


# ==========================================
# 1. Master Unified BareTorch Configuration
# ==========================================

class BareTorchConfig(PretrainedConfig):
    """
    Unified Master Configuration for the BareTorch Framework.
    Allows developers to dynamically instantiate any hybrid or pure sequence mixing topology
    using a single consolidated config schema.
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
        layer_types=None,  # Dynamic sequence mapping list, e.g. ["transformer", "cs_lrad", "cs_ttt"]
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
        self.use_cache = kwargs.get("use_cache", True)

        # If no explicit layer order is specified, default to an alternating rotation
        if layer_types is None:
            self.layer_types = []
            for i in range(num_layers):
                mod = i % 3
                if mod == 0:
                    self.layer_types.append("transformer")
                elif mod == 1:
                    self.layer_types.append("cs_lrad")
                else:
                    self.layer_types.append("cs_ttt")
        else:
            self.layer_types = layer_types


# ==========================================
# 2. Hugging Face Global AutoConfig Registration
# ==========================================

# Register the master configuration
AutoConfig.register("baretorch", BareTorchConfig)

# Register specific architecture configurations for granular serialization paths
AutoConfig.register("cs_lrad", CSLRADConfig)
AutoConfig.register("transformer", TransformerConfig)
AutoConfig.register("cs_ttt", CSTTTConfig)
AutoConfig.register("cs_lrad_transformer", CSLRADTransformerConfig)
AutoConfig.register("cs_ttt_transformer", CSTTTTransformerConfig)
AutoConfig.register("cs_lrad_cs_ttt_transformer", CSLRADCSTTTTransformerConfig)