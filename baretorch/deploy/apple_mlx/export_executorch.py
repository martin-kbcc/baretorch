# baretorch/deploy/apple_mlx/export_executorch.py
import os
import argparse
import torch
from transformers import AutoTokenizer

from baretorch import BareTorchConfig, BareTorchForCausalLM

try:
    from executorch.exir import to_edge, EdgeCompileConfig
    from executorch.backends.apple.mps.partition import MpsPartitioner
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False


def export_baretorch_to_executorch_pte(
    checkpoint_dir: str,
    output_pte_path: str,
    seq_len: int = 2048,
    dtype: str = "float16"
):
    """Exports trained BareTorch PyTorch weights to an ExecuTorch .pte Metal binary."""
    os.makedirs(os.path.dirname(output_pte_path) or ".", exist_ok=True)
    
    target_dtype = torch.float16 if dtype == "float16" else torch.float32
    print(f"📦 Loading BareTorch weights from '{checkpoint_dir}' in {target_dtype}...")

    model = BareTorchForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=target_dtype,
    ).eval()

    dummy_input_ids = torch.zeros((1, 1), dtype=torch.long)

    print(f"⚡ Exporting PyTorch graph using torch.export...")
    with torch.no_grad():
        exported_program = torch.export.export(model, (dummy_input_ids,))

    if HAS_EXECUTORCH:
        print(f"🍏 Applying EXIR lowering and Apple Metal MPS Partitioner...")
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_program_mps = edge_program.to_backend(MpsPartitioner())
        exec_program = edge_program_mps.to_executorch()

        with open(output_pte_path, "wb") as f:
            exec_program.write_to_file(f)

        size_mb = round(os.path.getsize(output_pte_path) / (1024.0 ** 2), 2)
        print(f"✅ Successfully exported ExecuTorch Metal binary to '{output_pte_path}' ({size_mb} MB).")
    else:
        print("⚠️ ExecuTorch package not found in Python environment.")
        print(f"💾 Saving TorchScript fallback model to '{output_pte_path}.pt'...")
        traced_model = torch.jit.trace(model, dummy_input_ids)
        torch.jit.save(traced_model, f"{output_pte_path}.pt")


def main():
    parser = argparse.ArgumentParser(description="BareTorch ExecuTorch Metal (.pte) Exporter")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_500m_sft",
        help="Path to trained BareTorch model directory."
    )
    parser.add_argument(
        "--output_pte",
        type=str,
        default="./baretorch_500m_metal.pte",
        help="Target path for output ExecuTorch .pte file."
    )
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    export_baretorch_to_executorch_pte(
        checkpoint_dir=args.checkpoint_dir,
        output_pte_path=args.output_pte,
        seq_len=args.seq_len,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()