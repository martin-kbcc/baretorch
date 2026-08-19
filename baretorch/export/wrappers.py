# baretorch/export/wrappers.py
import torch
import torch.nn as nn


class ModelExportWrapper(nn.Module):
    """
    Wraps CausalLM models (BareTorch or Hugging Face) to return ONLY logits 
    as a pure torch.Tensor. Strips dataclasses, tuples, and DynamicCache objects
    to ensure clean FX graph tracing via torch.export.export().
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, use_cache=False, return_dict=False)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out
        return logits