# baretorch/export/quantization.py
import torch
import torch.nn as nn
from typing import Tuple


def apply_quantization(
    wrapper_model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    quant_type: str = "fp32"
) -> nn.Module:
    """
    Applies ahead-of-time (AOT) weight and activation quantization prior to EXIR lowering.
    Supports 'fp32' (eager baseline) and 'int8' (W8A8 dynamic/static).
    """
    if quant_type.lower() == "fp32":
        return wrapper_model.eval().cpu()

    elif quant_type.lower() == "int8":
        print("  ⚙️ Applying INT8 (W8A8) dynamic quantization pass...")

        try:
            from torchao.quantization import quantize_

            # Multi-version fallback for torchao API evolution
            quant_plan = None
            try:
                from torchao.quantization import Int8DynamicActivationInt8WeightConfig
                quant_plan = Int8DynamicActivationInt8WeightConfig()
            except ImportError:
                try:
                    from torchao.quantization import int8_dynamic_activation_int8_weight
                    quant_plan = int8_dynamic_activation_int8_weight()
                except ImportError:
                    from torchao.quantization.quant_api import int8_dynamic_activation_int8_weight
                    quant_plan = int8_dynamic_activation_int8_weight()

            quantized_model = wrapper_model.eval().cpu()
            quantize_(quantized_model, quant_plan)
            print("  ✅ INT8 quantization applied successfully via torchao.")
            return quantized_model

        except Exception as e_ao:
            print(f"  ⚠️ torchao quantization failed ({e_ao}). Proceeding with unquantized FP32 graph...")
            return wrapper_model.eval().cpu()

    else:
        raise ValueError(f"Unsupported quantization type '{quant_type}'. Choose 'fp32' or 'int8'.")