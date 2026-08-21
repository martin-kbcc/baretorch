# baretorch/export/quantization.py
import torch
import torch.nn as nn

try:
    import torchao
    from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False


def apply_quantization(
    model: nn.Module,
    quant_type: str = "int4"
) -> nn.Module:
    """
    Applies ahead-of-time (AOT) weight quantization via torchao on Apple Silicon (MPS) or CPU.
    Specifies version=1 to bypass CUDA MSLK/Triton kernel checks.
    """
    quant_str = quant_type.lower().strip()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if quant_str == "fp32":
        return model.eval().to(device)

    elif quant_str in ["int4", "int8"]:
        if not HAS_TORCHAO:
            print("  ⚠️ 'torchao' is not installed (`pip install torchao`). Falling back to FP32...")
            return model.eval().to(device)

        print(f"  ⚡ Applying torchao native {quant_str.upper()} weight-only quantization on {device}...")
        try:
            model = model.eval().to(device)
            if quant_str == "int4":
                # version=1 uses pure PyTorch layouts that do not check for CUDA MSLK
                config = Int4WeightOnlyConfig(group_size=32, version=1)
                quantize_(model, config)
            elif quant_str == "int8":
                config = Int8WeightOnlyConfig()
                quantize_(model, config)

            print(f"  ✅ {quant_str.upper()} quantization applied successfully via torchao.")
            return model
        except Exception as e:
            print(f"  ⚠️ torchao quantization failed ({type(e).__name__}: {e}). Proceeding with unquantized FP32 graph...")
            return model.eval().to(device)
    else:
        raise ValueError(f"Unsupported quantization type '{quant_type}'. Choose 'fp32', 'int8', or 'int4'.")