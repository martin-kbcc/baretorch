# baretorch/deploy/apple_mlx/convert_weights.py
import os
import argparse
import torch
import mlx.core as mx
from transformers import AutoModelForCausalLM
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM


def convert_pytorch_to_fused_mlx_safetensors(
    pt_model: torch.nn.Module,
    output_path: str,
    dtype: str = "float16"
):
    """
    Fuses separate PyTorch projection weights (W_q, W_k, W_v, W_u, W_r, W_gate, W_beta, W_swish)
    into a single MLX 'W_in' weight matrix per layer.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    target_dtype = getattr(mx, dtype, mx.float16)
    pt_state = pt_model.state_dict()

    mlx_weights = {}
    fused_layers = set()

    # Identify cs_lrad layer indices
    for key in pt_state.keys():
        if "attn.W_q.weight" in key:
            layer_prefix = key.rsplit(".attn.W_q.weight", 1)[0]
            fused_layers.add(layer_prefix)

    # 1. Process and Fuse CS-LRAD Layers
    for prefix in fused_layers:
        w_q = pt_state.pop(f"{prefix}.attn.W_q.weight")
        w_k = pt_state.pop(f"{prefix}.attn.W_k.weight")
        w_v = pt_state.pop(f"{prefix}.attn.W_v.weight")
        w_u = pt_state.pop(f"{prefix}.attn.W_u.weight")
        w_r = pt_state.pop(f"{prefix}.attn.W_r.weight")
        
        w_gate = pt_state.pop(f"{prefix}.attn.W_gate.weight")
        b_gate = pt_state.pop(f"{prefix}.attn.W_gate.bias")
        
        w_beta = pt_state.pop(f"{prefix}.attn.W_beta_gate.weight")
        b_beta = pt_state.pop(f"{prefix}.attn.W_beta_gate.bias")
        
        w_swish = pt_state.pop(f"{prefix}.attn.W_swish_gate.weight")

        # Fuse weights along output dim (dim 0 in PyTorch linear weights)
        fused_weight = torch.cat([w_q, w_k, w_v, w_u, w_r, w_gate, w_beta, w_swish], dim=0)

        # Construct fused bias vector (zero-padded for layers without bias)
        b_q = torch.zeros(w_q.size(0), device=w_q.device, dtype=w_q.dtype)
        b_k = torch.zeros(w_k.size(0), device=w_k.device, dtype=w_k.dtype)
        b_v = torch.zeros(w_v.size(0), device=w_v.device, dtype=w_v.dtype)
        b_u = torch.zeros(w_u.size(0), device=w_u.device, dtype=w_u.dtype)
        b_r = torch.zeros(w_r.size(0), device=w_r.device, dtype=w_r.dtype)
        b_swish = torch.zeros(w_swish.size(0), device=w_swish.device, dtype=w_swish.dtype)

        fused_bias = torch.cat([b_q, b_k, b_v, b_u, b_r, b_gate, b_beta, b_swish], dim=0)

        mlx_weights[f"{prefix}.attn.W_in.weight"] = mx.array(fused_weight.cpu().to(torch.float32).numpy()).astype(target_dtype)
        mlx_weights[f"{prefix}.attn.W_in.bias"] = mx.array(fused_bias.cpu().to(torch.float32).numpy()).astype(target_dtype)

    # 2. Copy remaining weights (Transformer blocks, Embeddings, Norms, LM Head)
    for name, tensor in pt_state.items():
        np_arr = tensor.detach().cpu().to(torch.float32).numpy()
        mlx_weights[name] = mx.array(np_arr).astype(target_dtype)

    mx.save_safetensors(output_path, mlx_weights)
    size_mb = round(os.path.getsize(output_path) / (1024.0 ** 2), 2)
    print(f"✅ Successfully exported Fused MLX weights to '{output_path}' ({size_mb} MB).")


def main():
    parser = argparse.ArgumentParser(description="BareTorch PyTorch to Fused MLX Weight Converter")
    parser.add_argument("--hf_model_id", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./baretorch_mlx.safetensors")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    if args.hf_model_id:
        print(f"📦 Loading pretrained PyTorch model from '{args.hf_model_id}'...")
        pt_model = AutoModelForCausalLM.from_pretrained(args.hf_model_id, trust_remote_code=True)
    else:
        print("⚙️ Instantiating default BareTorch model for fused weight structure export...")
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

    convert_pytorch_to_fused_mlx_safetensors(pt_model, args.output_path, dtype=args.dtype)


if __name__ == "__main__":
    main()