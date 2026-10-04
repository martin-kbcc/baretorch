import torch
from transformers import AutoTokenizer

from baretorch import BareTorchConfig, BareTorchForCausalLM


def inspect_candidate(tokenizer_name, vocab_size, d_model, num_layers):
    raw_sequence = "cs_lrad,cs_lrad,cs_lrad,transformer".split(",")
    layer_types = [
        raw_sequence[i % len(raw_sequence)] for i in range(num_layers)
    ]

    config_args = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": 16,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "use_qk_norm": True,
        "tie_word_embeddings": True,
    }

    config = BareTorchConfig(**config_args)
    model = BareTorchForCausalLM(config)

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    elif hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
        model.lm_head.weight = model.embed_tokens.weight

    unique_params_dict = {p.data_ptr(): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params_dict.values())
    embed_params_count = vocab_size * d_model
    backbone_params_count = total_params - embed_params_count

    embed_pct = (embed_params_count / total_params) * 100
    backbone_pct = (backbone_params_count / total_params) * 100

    print(
        f"\n  [Config: d_model={d_model}, num_layers={num_layers}, num_heads=16]"
    )
    print(
        f"  ├─ Embedding Parameters  : {embed_params_count:,} ({embed_params_count/1e6:.2f}M)  -->  {embed_pct:.2f}%"
    )
    print(
        f"  ├─ Backbone (CS-LRAD/Attn): {backbone_params_count:,} ({backbone_params_count/1e6:.2f}M)  -->  {backbone_pct:.2f}%"
    )
    print(
        f"  └─ Total Unique Params   : {total_params:,} ({total_params/1e6:.2f}M)"
    )


def main():
    tokenizer_name = "Qwen/Qwen3.5-9B"
    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    vocab_size = len(tokenizer)

    print("\n" + "=" * 70)
    print("📊 BARETORCH 500M PARAMETER SCALE ANALYSIS")
    print("=" * 70)
    print(f"Vocab Size: {vocab_size:,} | Tied Embeddings: True")

    # Baseline 300M (for comparison)
    print("\n--- Current 300M Model Baseline ---")
    inspect_candidate(tokenizer_name, vocab_size, d_model=768, num_layers=12)

    # Candidate 1: 500M Target (20 Layers)
    print("\n--- Option 1: ~450M-500M Target (20 Layers) ---")
    inspect_candidate(tokenizer_name, vocab_size, d_model=1024, num_layers=20)

    # Candidate 2: 550M Target (24 Layers)
    print("\n--- Option 2: ~520M-550M Target (24 Layers) ---")
    inspect_candidate(tokenizer_name, vocab_size, d_model=1024, num_layers=24)

    print("=" * 70)


if __name__ == "__main__":
    main()