# ==============================================================================
#                               BareTorch Ecosystem
#  Kernel-Free, Pure GEMM-Compliant Next-Gen Sequence Mixing Architectures
# ==============================================================================

__version__ = "0.1.0"

# --- 1. Expose Configurations ---
from baretorch.integration.configuration_baretorch import (
    BareTorchConfig,
    CSLRADConfig,
    TransformerConfig,
    CSTTTConfig,
    CSLRADTransformerConfig,
    CSTTTTransformerConfig,
    CSLRADCSTTTTransformerConfig,
)

# --- 2. Expose Core Native Modeling Blocks (Pure PyTorch)[cite: 1] ---
from baretorch.modeling.transformer import (
    RotaryEmbedding,
    CausalSelfAttention,
    TransformerDecoderBlock,
    TransformerModel,
    TransformerForCausalLM,
)

from baretorch.modeling.cs_lrad import (
    RMSNorm,
    GatedMLP,
    LowRankAssociativeDeltaEngine,
    LRADDecoderBlock,
    CSLRADModel,
    CSLRADForCausalLM,
)

from baretorch.modeling.cs_ttt import (
    ChunkwiseTestTimeTrainingEngine,
    TTTDecoderBlock,
    CSTTTModel,
    CSTTTForCausalLM,
)

# --- 3. Expose Hybrid Architectures ---
from baretorch.modeling.cs_lrad_transformer import (
    CSLRADTransformerModel,
    CSLRADTransformerForCausalLM,
)

from baretorch.modeling.cs_ttt_transformer import (
    CSTTTTransformerModel,
    CSTTTTransformerForCausalLM,
)

from baretorch.modeling.cs_lrad_cs_ttt_transformer import (
    CSLRADCSTTTTransformerModel,
    CSLRADCSTTTTransformerForCausalLM,
)

# --- 4. Expose Master Unified Models ---
from baretorch.integration.modeling_baretorch import (
    BareTorchModel,
    BareTorchForCausalLM,
)

# Defined exports for wildcard imports
__all__ = [
    "BareTorchConfig",
    "CSLRADConfig",
    "TransformerConfig",
    "CSTTTConfig",
    "CSLRADTransformerConfig",
    "CSTTTTransformerConfig",
    "CSLRADCSTTTTransformerConfig",
    "RotaryEmbedding",
    "CausalSelfAttention",
    "TransformerDecoderBlock",
    "TransformerModel",
    "TransformerForCausalLM",
    "RMSNorm",
    "GatedMLP",
    "LowRankAssociativeDeltaEngine",
    "LRADDecoderBlock",
    "CSLRADModel",
    "CSLRADForCausalLM",
    "ChunkwiseTestTimeTrainingEngine",
    "TTTDecoderBlock",
    "CSTTTModel",
    "CSTTTForCausalLM",
    "CSLRADTransformerModel",
    "CSLRADTransformerForCausalLM",
    "CSTTTTransformerModel",
    "CSTTTTransformerForCausalLM",
    "CSLRADCSTTTTransformerModel",
    "CSLRADCSTTTTransformerForCausalLM",
    "BareTorchModel",
    "BareTorchForCausalLM",
]