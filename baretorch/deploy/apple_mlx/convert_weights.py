# baretorch/deploy/apple_mlx/convert_weights.py
import os
import argparse
import torch
import mlx.core as mx
from transformers import AutoModelForCausalLM
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM


def convert_pytorch_to_mlx_safetensors(
    pt_model: torch.nn.Module,
    output_path: str,
    dtype: str = "float16"
):
    """
    Converts PyTorch state_dict parameters directly into MLX safetensors format.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    target_dtype = getattr(mx, dtype, mx.float16)
    pt_state = pt_model.state_dict()

    mlx_weights = {}
    for name, tensor in pt_state.items():
        np_arr = tensor.detach().cpu().to(torch.float32).numpy()
        mlx_weights[name] = mx.array(np_arr).astype(target_dtype)

    mx.save_safetensors(output_path, mlx_weights)
    size_mb = round(os.path.getsize(output_path) / (1024.0 ** 2), 2)
    print(f"✅ Successfully exported MLX weights to '{output_path}' ({size_mb} MB).")


def main():
    parser = argparse.ArgumentParser(description="BareTorch PyTorch to MLX Weight Converter")
    parser.add_argument("--hf_model_id", type=str, default=None, help="Optional HF checkpoint path")
    parser.add_argument("--output_path", type=str, default="./baretorch_mlx.safetensors")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    if args.hf_model_id:
        print(f"📦 Loading pretrained PyTorch model from '{args.hf_model_id}'...")
        pt_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, trust_remote_code=True)
    else:
        print("⚙️ Instantiating default BareTorch model for weight structure export...")
        from baretorch.integration.configuration_baretorch import BareTorchConfig
        config = BareTorchConfig(
            vocab_size=50257,
            d_model=1888,
            num_heads=16,
            num_layers=14,
            chunk_size=32,
            rank=8,
            layer_types=["cs_lrad", "cs_lrad", "cs_lrad", "transformer"] * 3 + ["cs_lrad", "cs_lrad"]
        )
        pt_model = BareTorchForCausalLM(config)

    convert_pytorch_to_mlx_safetensors(pt_model, args.output_path, dtype=args.dtype)


if __name__ == "__main__":
    main()