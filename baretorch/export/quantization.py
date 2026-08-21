# baretorch/export/quantization.py
import torch
import torch.nn as nn

try:
    from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False


def apply_quantization(
    model: nn.Module,
    quant_type: str = "int4"
) -> nn.Module:
    """
    Applies ahead-of-time (AOT) weight quantization via torchao prior to EXIR lowering.
    Supports 'int4', 'int8', and 'fp32'.
    """
    quant_str = quant_type.lower().strip()

    if quant_str == "fp32":
        return model.eval().cpu()

    elif quant_str in ["int4", "int8"]:
        if not HAS_TORCHAO:
            print("  ⚠️ 'torchao' is not installed (`pip install torchao`). Falling back to unquantized FP32...")
            return model.eval().cpu()

        print(f"  ⚡ Applying torchao native {quant_str.upper()} weight-only quantization...")
        try:
            model = model.eval().cpu()
            if quant_str == "int4":
                quantize_(model, int4_weight_only(group_size=32))
            elif quant_str == "int8":
                quantize_(model, int8_weight_only())
            print(f"  ✅ {quant_str.upper()} quantization applied successfully via torchao.")
            return model
        except Exception as e:
            print(f"  ⚠️ torchao quantization failed ({e}). Proceeding with unquantized FP32 graph...")
            return model.eval().cpu()
    else:
        raise ValueError(f"Unsupported quantization type '{quant_type}'. Choose 'fp32', 'int8', or 'int4'.")