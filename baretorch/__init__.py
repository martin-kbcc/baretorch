# ==============================================================================
#                                BareTorch Ecosystem
#   Kernel-Free, Pure GEMM-Compliant Next-Gen Sequence Mixing Architectures
# ==============================================================================

__version__ = "0.1.0"

# --- 1. Expose Configurations ---
from baretorch.integration.configuration_baretorch import (
    BareTorchConfig,
    CSLRADConfig,
    TransformerConfig,
)

# --- 2. Expose Core Native Modeling Blocks (Pure PyTorch) ---
from baretorch.modeling.transformer import (
    CausalSelfAttention,
    RotaryEmbedding,
    TransformerDecoderBlock,
    TransformerForCausalLM,
    TransformerModel,
)

from baretorch.modeling.cs_lrad import (
    CSLRADForCausalLM,
    CSLRADModel,
    GatedMLP,
    LowRankAssociativeDeltaEngine,
    LRADDecoderBlock,
    RMSNorm,
)

# --- 3. Expose Master Unified Models ---
from baretorch.integration.modeling_baretorch import (
    BareTorchForCausalLM,
    BareTorchModel,
)

# Defined exports for wildcard imports
__all__ = [
    # Configurations
    "BareTorchConfig",
    "CSLRADConfig",
    "TransformerConfig",
    # Transformer Blocks & Model
    "RotaryEmbedding",
    "CausalSelfAttention",
    "TransformerDecoderBlock",
    "TransformerModel",
    "TransformerForCausalLM",
    # CS-LRAD Blocks & Model
    "RMSNorm",
    "GatedMLP",
    "LowRankAssociativeDeltaEngine",
    "LRADDecoderBlock",
    "CSLRADModel",
    "CSLRADForCausalLM",
    # Master Unified BareTorch Architecture
    "BareTorchModel",
    "BareTorchForCausalLM",
]