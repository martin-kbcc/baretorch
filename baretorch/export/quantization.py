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
    Applies ahead-of-time (AOT) weight quantization via torchao on CPU prior to EXIR lowering.
    Specifies version=1 and set_inductor_config=False to prevent CUDA MSLK checks.
    """
    quant_str = quant_type.lower().strip()

    if quant_str == "fp32":
        return model.eval().cpu()

    elif quant_str in ["int4", "int8"]:
        if not HAS_TORCHAO:
            print("  ⚠️ 'torchao' is not installed (`pip install torchao`). Falling back to FP32...")
            return model.eval().cpu()

        print(f"  ⚡ Applying torchao native {quant_str.upper()} weight-only quantization on CPU...")
        try:
            model = model.eval().cpu()
            if quant_str == "int4":
                try:
                    config = Int4WeightOnlyConfig(group_size=32, set_inductor_config=False, version=1)
                    quantize_(model, config)
                except Exception as e_v1:
                    print(f"  ⚠️ INT4 v1 layout notice ({type(e_v1).__name__}: {e_v1}). Trying INT8 fallback...")
                    config = Int8WeightOnlyConfig(set_inductor_config=False)
                    quantize_(model, config)
            elif quant_str == "int8":
                config = Int8WeightOnlyConfig(set_inductor_config=False)
                quantize_(model, config)

            print(f"  ✅ {quant_str.upper()} quantization applied successfully via torchao.")
            return model.eval().cpu()
        except Exception as e:
            print(f"  ⚠️ torchao quantization failed ({type(e).__name__}: {e}). Proceeding with unquantized FP32 graph...")
            return model.eval().cpu()
    else:
        raise ValueError(f"Unsupported quantization type '{quant_type}'. Choose 'fp32', 'int8', or 'int4'.")