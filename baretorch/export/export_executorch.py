# baretorch/export/export_executorch.py
import os
import gc
import json
import csv
import argparse
import torch
import torch.nn as nn
from typing import List, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM

from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM
from baretorch.export.wrappers import ModelExportWrapper
from baretorch.export.quantization import apply_quantization
from baretorch.export.partitioners import get_backend_partitioner


def clear_host_memory():
    """Flushes Python garbage collection and clears CUDA/RAM allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_baretorch_params_fast(
    d_model: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    rank: int = 8,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer"
) -> float:
    """Computes exact BareTorch model parameter count in pure Python integer math."""
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]
    full_layer_types = [raw_seq[i % len(raw_seq)] for i in range(num_layers)]

    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    d_ff = int(d_model * 3.5)

    embed_params = 2 * vocab_size * d_model
    final_norm = d_model
    layer_params = 0

    for l_type in full_layer_types:
        norms = 2 * d_model
        mlp = 3 * d_model * d_ff

        if l_type == "cs_lrad":
            attn = (5 * (d_model ** 2)) + (2 * d_model * num_heads * rank) + (2 * (d_model * num_heads + num_heads))
        else:
            head_dim = d_model // num_heads
            attn = (2 * (d_model ** 2)) + (2 * d_model * (num_kv_heads * head_dim))

        layer_params += (norms + mlp + attn)

    total_params = embed_params + final_norm + layer_params
    return total_params / 1e6  # Millions


def find_matching_baretorch_config(
    target_params_m: float,
    target_vocab_size: int = 50257,
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    max_seq_len: int = 32768
) -> tuple[BareTorchConfig, float]:
    """Finds optimal BareTorch configuration aligned with baseline parameter count."""
    raw_seq = [s.strip().lower() for s in layer_sequence.split(",") if s.strip()]

    best_cfg = None
    best_diff = float("inf")
    best_params_m = 0.0

    for nl in range(12, 36, 2):
        for d in range(512, 3072, 32):
            for nh in [8, 12, 16, 20, 24, 32]:
                if d % nh != 0:
                    continue
                head_dim = d // nh
                if head_dim < 64 or head_dim > 128:
                    continue
                if head_dim % 2 != 0:
                    continue

                num_kv_heads = max(1, nh // 4)
                while nh % num_kv_heads != 0:
                    num_kv_heads -= 1

                p_m = count_baretorch_params_fast(
                    d_model=d,
                    num_layers=nl,
                    num_heads=nh,
                    vocab_size=target_vocab_size,
                    rank=8,
                    layer_sequence=layer_sequence
                )

                diff = abs(p_m - target_params_m)
                if diff < best_diff:
                    best_diff = diff
                    best_params_m = p_m
                    full_layer_types = [raw_seq[i % len(raw_seq)] for i in range(nl)]
                    best_cfg = BareTorchConfig(
                        vocab_size=target_vocab_size,
                        d_model=d,
                        num_heads=nh,
                        num_kv_heads=num_kv_heads,
                        num_layers=nl,
                        chunk_size=32,
                        rank=8,
                        dropout=0.0,
                        max_seq_len=max_seq_len,
                        layer_types=full_layer_types
                    )

    div_pct = (best_diff / target_params_m) * 100
    print(f"  ⚡ Vocab-Aware Match completed (|Δ| = {best_diff:.2f}M, {div_pct:.2f}%)")
    return best_cfg, best_params_m


def export_single_model_to_pte(
    model: nn.Module,
    model_name: str,
    vocab_size: int,
    seq_len: int,
    output_pte_path: str,
    quant_type: str = "int4",
    backend_delegate: str = "none"
) -> Dict[str, Any]:
    """
    Exports a model to ExecuTorch (.pte) format using graph capture, torchao quantization,
    backend partitioning, and AOT MemoryPlanningPass.
    """
    os.makedirs(os.path.dirname(output_pte_path) or ".", exist_ok=True)

    # 1. Wrap model for pure Tensor logits output
    wrapper = ModelExportWrapper(model).eval().cpu()
    param_count_m = sum(p.numel() for p in wrapper.parameters()) / 1e6
    example_inputs = (torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long),)

    print(f"\n📦 Exporting '{model_name}' ({param_count_m:.2f}M params) | SeqLen: {seq_len} | Quant: {quant_type.upper()} | Backend: {backend_delegate.upper()}...")

    try:
        from torch.export import export
        from executorch.exir import to_edge, EdgeCompileConfig, ExecutorchBackendConfig
        from executorch.exir.passes import MemoryPlanningPass

        # 2. Apply torchao Native Quantization
        prepared_model = apply_quantization(wrapper, quant_type=quant_type)

        # 3. Capture PyTorch ExportedProgram
        with torch.no_grad():
            exported_prog = export(prepared_model, example_inputs)

        # 4. Lower to ExecuTorch Edge Dialect
        edge_prog = to_edge(
            exported_prog,
            compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        # 5. Resolve & Apply Hardware Partitioner
        partitioners = get_backend_partitioner(backend_delegate)
        if partitioners:
            print("  🧩 Lowering graph through backend delegate partitioner...")
            edge_prog = edge_prog.to_backend(partitioners[0])

        # 6. Lower to ExecuTorch Program with AOT Memory Planning
        et_prog = edge_prog.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass()
            )
        )

        # 7. Serialize to .pte binary file
        buf = getattr(et_prog, "buffer", None)
        raw_bytes = buf if isinstance(buf, (bytes, bytearray)) else (buf() if callable(buf) else et_prog.buffer)
        with open(output_pte_path, "wb") as f:
            f.write(raw_bytes)

        file_size_mb = round(os.path.getsize(output_pte_path) / (1024.0 ** 2), 2)

        # 8. Extract Tensor Arena Memory Allocation from ExecuTorch Plan
        arena_bytes = 0
        try:
            program_flatbuffer = getattr(et_prog, "executorch_program", None)
            if program_flatbuffer and hasattr(program_flatbuffer, "execution_plan"):
                plans = program_flatbuffer.execution_plan
                if len(plans) > 0 and hasattr(plans[0], "non_const_buffer_sizes"):
                    arena_bytes = sum(plans[0].non_const_buffer_sizes)
        except Exception as read_err:
            print(f"    ⚠️ Could not extract non_const_buffer_sizes ({read_err})")

        arena_ram_mb = round(arena_bytes / (1024.0 ** 2), 2)

        print(f"    ✅ ExecuTorch export successful:")
        print(f"        • .pte File Size    : {file_size_mb} MB")
        print(f"        • Tensor Arena RAM  : {arena_ram_mb} MB")

        return {
            "model_name": model_name,
            "param_count_m": round(param_count_m, 2),
            "seq_len": seq_len,
            "quant_type": quant_type,
            "backend_delegate": backend_delegate,
            "pte_file_size_mb": file_size_mb,
            "tensor_arena_ram_mb": arena_ram_mb,
            "status": "success",
            "pte_path": output_pte_path
        }

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"    ⚠️ ExecuTorch export failed for '{model_name}' ({err_msg})")
        return {
            "model_name": model_name,
            "param_count_m": round(param_count_m, 2),
            "seq_len": seq_len,
            "quant_type": quant_type,
            "backend_delegate": backend_delegate,
            "pte_file_size_mb": "N/A",
            "tensor_arena_ram_mb": "N/A",
            "status": f"failed ({type(e).__name__})",
            "pte_path": "N/A"
        }


def print_stage1_export_report(export_results: List[Dict[str, Any]]):
    """Prints summary table comparing ExecuTorch edge lowering metrics."""
    print("\n" + "=" * 135)
    print("📱 EXECUTORCH AOT EDGE LOWERING & HARDWARE DELEGATE REPORT")
    print("=" * 135)

    for idx, pair in enumerate(export_results, 1):
        hf_res = pair["hf_baseline"]
        bt_res = pair["baretorch_matched"]

        print(f"\n🎯 PAIR {idx}: {hf_res['model_name']}")
        print(f"    • Baseline Params          : {hf_res['param_count_m']:.2f}M")
        print(f"    • BareTorch Matched Params: {bt_res['param_count_m']:.2f}M (Δ = {abs(bt_res['param_count_m'] - hf_res['param_count_m']):.2f}M)")
        print("-" * 135)

        f_b, f_h = bt_res["pte_file_size_mb"], hf_res["pte_file_size_mb"]
        f_adv = f"-{((f_h - f_b) / f_h) * 100:.1f}% Disk" if (isinstance(f_h, (int, float)) and isinstance(f_b, (int, float)) and f_h > 0) else "N/A"
        print(f"    .pte Binary Size (MB)    : BareTorch = {str(f_b):<12} | Baseline = {str(f_h):<12} | {f_adv}")

        a_b, a_h = bt_res["tensor_arena_ram_mb"], hf_res["tensor_arena_ram_mb"]
        a_adv = f"-{((a_h - a_b) / a_h) * 100:.1f}% RAM" if (isinstance(a_h, (int, float)) and isinstance(a_b, (int, float)) and a_h > 0) else "N/A"
        print(f"    Tensor Arena RAM (MB)    : BareTorch = {str(a_b):<12} | Baseline = {str(a_h):<12} | {a_adv}")

        print("-" * 135)


def export_to_csv(export_results: List[Dict[str, Any]], output_csv: str):
    """Saves structured comparative ExecuTorch metrics to CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "Baseline_Model_ID", "BareTorch_Params_M", "Baseline_Params_M", "Quant_Type", "Backend_Delegate",
        "BareTorch_PTE_Size_MB", "Baseline_PTE_Size_MB", "Disk_Reduction_Pct",
        "BareTorch_Tensor_Arena_MB", "Baseline_Tensor_Arena_MB", "Arena_RAM_Reduction_Pct"
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for pair in export_results:
            hf = pair["hf_baseline"]
            bt = pair["baretorch_matched"]

            f_b, f_h = bt["pte_file_size_mb"], hf["pte_file_size_mb"]
            f_pct = f"{((f_h - f_b) / f_h) * 100:.2f}%" if (isinstance(f_h, (int, float)) and isinstance(f_b, (int, float)) and f_h > 0) else "N/A"

            a_b, a_h = bt["tensor_arena_ram_mb"], hf["tensor_arena_ram_mb"]
            a_pct = f"{((a_h - a_b) / a_h) * 100:.2f}%" if (isinstance(a_h, (int, float)) and isinstance(a_b, (int, float)) and a_h > 0) else "N/A"

            writer.writerow([
                hf["model_name"], bt["param_count_m"], hf["param_count_m"], bt["quant_type"], bt["backend_delegate"],
                f_b, f_h, f_pct,
                a_b, a_h, a_pct
            ])


def run_stage1_export_suite(
    hf_model_ids: List[str],
    layer_sequence: str = "cs_lrad,cs_lrad,cs_lrad,transformer",
    seq_len: int = 128,
    quant_type: str = "int4",
    backend_delegate: str = "none",
    dummy_weights: bool = False,
    output_dir: str = "./pte_models",
    output_json: str = "./executorch_stage1_results.json",
    output_csv: str = "./executorch_stage1_results.csv"
):
    print("\n======================================================================")
    print(f"🚀 STAGE 1: EXECUTORCH AOT EDGE LOWERING & MEMORY SUITE")
    print(f"    • Baseline Target Models ({len(hf_model_ids)}) : {', '.join(hf_model_ids)}")
    print(f"    • Quantization: {quant_type.upper()} (torchao) | Backend Delegate: {backend_delegate.upper()}")
    print(f"    • Weight Initialization Mode: {'DUMMY (Random Weights)' if dummy_weights else 'PRETRAINED'}")
    print("======================================================================\n")

    export_results = []

    for model_id in hf_model_ids:
        clear_host_memory()
        sanitized_name = model_id.replace("/", "_").replace("-", "_").lower()
        print(f"\n📦 Processing Hugging Face baseline model: '{model_id}'...")

        try:
            if dummy_weights:
                print(f"  ⚡ Instantiating dummy randomly initialized model for '{model_id}'...")
                hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True).to(dtype=torch.float32)
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            actual_model_id = model_id
        except Exception as e:
            print(f"⚠️ Could not load '{model_id}' ({e}). Falling back to dummy random 'gpt2'...")
            actual_model_id = f"gpt2 (dummy fallback for {model_id})"
            hf_config = AutoConfig.from_pretrained("gpt2")
            hf_model = AutoModelForCausalLM.from_config(hf_config).to(dtype=torch.float32)

        cfg = getattr(hf_model, "config", None)
        cfg_dict = cfg.to_dict() if (cfg is not None and hasattr(cfg, "to_dict")) else {}
        target_vocab_size = cfg_dict.get("vocab_size", getattr(cfg, "vocab_size", 50257))
        hf_params_m = sum(p.numel() for p in hf_model.parameters()) / 1e6

        # 1. Export HF Baseline to .pte
        hf_pte_path = os.path.join(output_dir, f"baseline_{sanitized_name}_{quant_type}_{backend_delegate}.pte")
        hf_res = export_single_model_to_pte(
            model=hf_model,
            model_name=actual_model_id,
            vocab_size=target_vocab_size,
            seq_len=seq_len,
            output_pte_path=hf_pte_path,
            quant_type=quant_type,
            backend_delegate=backend_delegate
        )

        del hf_model
        clear_host_memory()

        # 2. Find matching BareTorch blueprint
        print(f"\n⚙️ Finding BareTorch blueprint matching ~{hf_params_m:.2f}M parameters...")
        bt_config, bt_params_m_predicted = find_matching_baretorch_config(
            target_params_m=hf_params_m,
            target_vocab_size=target_vocab_size,
            layer_sequence=layer_sequence,
            max_seq_len=32768
        )

        # 3. Instantiate BareTorch
        bt_model = BareTorchForCausalLM(bt_config).to(dtype=torch.float32)
        actual_bt_params_m = sum(p.numel() for p in bt_model.parameters()) / 1e6

        print(f"  🎯 Target Params: {hf_params_m:.2f}M | Predicted Math: {bt_params_m_predicted:.2f}M | Actual BareTorch Instantiated: {actual_bt_params_m:.2f}M (Δ = {abs(actual_bt_params_m - hf_params_m):.2f}M)")
        print(f"     Config: d_model={bt_config.d_model}, num_layers={bt_config.num_layers}, num_heads={bt_config.num_heads}")

        bt_pte_path = os.path.join(output_dir, f"baretorch_{sanitized_name}_{quant_type}_{backend_delegate}.pte")

        bt_res = export_single_model_to_pte(
            model=bt_model,
            model_name=f"BareTorch Matched ({actual_bt_params_m:.1f}M)",
            vocab_size=target_vocab_size,
            seq_len=seq_len,
            output_pte_path=bt_pte_path,
            quant_type=quant_type,
            backend_delegate=backend_delegate
        )

        del bt_model
        clear_host_memory()

        export_results.append({
            "hf_baseline": hf_res,
            "baretorch_matched": bt_res
        })

    print_stage1_export_report(export_results)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(export_results, f, indent=2)

    export_to_csv(export_results, output_csv)
    print(f"💾 Stage 1 ExecuTorch JSON report saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="BareTorch ExecuTorch Stage 1 Export Suite")
    parser.add_argument(
        "--hf_model_ids",
        nargs="+",
        type=str,
        default=[
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        ],
        help="Space-separated list of Hugging Face model IDs to export"
    )
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--quant_type", type=str, choices=["fp32", "int8", "int4"], default="int4", help="Quantization mode (via torchao)")
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["none", "xnnpack", "coreml", "qnn", "vulkan"], 
        default="none", 
        help="ExecuTorch delegate partitioner target"
    )
    parser.add_argument("--dummy_weights", action="store_true", help="Use randomly initialized weights (fast testing without downloading checkpoints)")
    parser.add_argument("--output_dir", type=str, default="./pte_models")
    parser.add_argument("--output_json", type=str, default="./executorch_stage1_results.json")
    parser.add_argument("--output_csv", type=str, default="./executorch_stage1_results.csv")

    args = parser.parse_args()

    run_stage1_export_suite(
        hf_model_ids=args.hf_model_ids,
        layer_sequence=args.layer_sequence,
        seq_len=args.seq_len,
        quant_type=args.quant_type,
        backend_delegate=args.backend,
        dummy_weights=args.dummy_weights,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()