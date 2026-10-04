import os
import torch
from transformers import AutoTokenizer

from baretorch import BareTorchConfig, BareTorchForCausalLM


def inspect_parameters():
    tokenizer_name = "Qwen/Qwen3.5-9B"
    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    vocab_size = len(tokenizer)

    d_model = 768
    num_heads = 16
    num_layers = 12
    raw_sequence = "cs_lrad,cs_lrad,cs_lrad,transformer".split(",")
    layer_types = [
        raw_sequence[i % len(raw_sequence)] for i in range(num_layers)
    ]

    config_args = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "use_qk_norm": True,
        "tie_word_embeddings": True,
    }

    config = BareTorchConfig(**config_args)
    model = BareTorchForCausalLM(config)

    # Tie embedding weights if needed
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    elif hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
        model.lm_head.weight = model.embed_tokens.weight

    # 1. Total unique parameters (avoiding double-counting tied weights)
    unique_params_dict = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params_dict.values())

    # 2. Embedding parameters (Vocab Size * d_model)
    embed_params_count = vocab_size * d_model

    # 3. Backbone parameters (Hidden layers, CS-LRAD, Transformer, Norms, etc.)
    backbone_params_count = total_params - embed_params_count

    embed_pct = (embed_params_count / total_params) * 100
    backbone_pct = (backbone_params_count / total_params) * 100

    print("\n" + "=" * 70)
    print("📊 BARETORCH 300M PARAMETER DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Vocab Size            : {vocab_size:,}")
    print(f"Hidden Dimension (d)  : {d_model}")
    print(f"Number of Layers      : {num_layers}")
    print(f"Tied Embeddings       : True")
    print("-" * 70)
    print(
        f"Embedding Parameters  : {embed_params_count:,} ({embed_params_count/1e6:.2f}M)  -->  {embed_pct:.2f}%"
    )
    print(
        f"Backbone (CS-LRAD/Attn): {backbone_params_count:,} ({backbone_params_count/1e6:.2f}M)  -->  {backbone_pct:.2f}%"
    )
    print(
        f"Total Unique Params   : {total_params:,} ({total_params/1e6:.2f}M)  -->  100.00%"
    )
    print("=" * 70)


if __name__ == "__main__":
    inspect_parameters()