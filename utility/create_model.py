import sys
import torch
import torch.nn as nn

def format_params(num_params: int) -> str:
    """Formats raw parameter numbers into human-readable Millions/Billions format."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B ({num_params:,})"
    return f"{num_params / 1e6:.2f}M ({num_params:,})"

# Attempt import from local BareTorch repository modules
try:
    from model import BareTorchForCausalLM, BareTorchConfig
except ImportError:
    try:
        from baretorch import BareTorchForCausalLM, BareTorchConfig
    except ImportError:
        BareTorchConfig = None
        BareTorchForCausalLM = None


def main():
    # Architecture hyperparameters matching launch_lrad_hybrid.sh
    vocab_size = 49152       # HuggingFaceTB/SmolLM2-360M tokenizer
    d_model = 1152
    num_layers = 24
    layer_sequence = "cs_lrad,cs_lrad,cs_lrad,transformer"
    chunk_size = 32
    rank = 8
    seq_len = 2048
    
    print("=" * 70)
    print("🛠️  BareTorch 500M Hybrid Architecture Parameter Counter")
    print("=" * 70)
    print(f" • Tokenizer Vocab Size : {vocab_size}")
    print(f" • Hidden Dim (d_model)  : {d_model}")
    print(f" • Total Layers         : {num_layers} ({layer_sequence})")
    print(f" • Rank / Chunk Size    : Rank={rank}, Chunk={chunk_size}")
    print(f" • Context Window       : {seq_len}")
    print("=" * 70)

    if BareTorchForCausalLM is not None and BareTorchConfig is not None:
        config = BareTorchConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            layer_sequence=layer_sequence,
            chunk_size=chunk_size,
            rank=rank,
            max_position_embeddings=seq_len,
        )
        
        # Instantiate model on CPU meta device or standard initialization
        with torch.device("meta"):
            model = BareTorchForCausalLM(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Isolate embedding and head parameters
        embed_params = sum(
            p.numel() for name, p in model.named_parameters() 
            if any(k in name.lower() for k in ["embed", "lm_head", "wte"])
        )
        non_embed_params = total_params - embed_params

        bf16_size_mb = (total_params * 2) / (1024 ** 2)
        fp32_size_mb = (total_params * 4) / (1024 ** 2)

        print("\n✅ Model Parameter Summary:")
        print(f" ├─ Total Parameters       : {format_params(total_params)}")
        print(f" ├─ Trainable Parameters   : {format_params(trainable_params)}")
        print(f" ├─ Non-Embedding Params   : {format_params(non_embed_params)}")
        print(f" ├─ Model Size (bfloat16)  : {bf16_size_mb:.2f} MB")
        print(f" └─ Model Size (float32)   : {fp32_size_mb:.2f} MB")
        print("=" * 70)
    else:
        print("\n⚠️  BareTorch module imports not found.")
        print("   Please execute this script from your project root directory where model.py resides.")


if __name__ == "__main__":
    main()