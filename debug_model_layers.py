# /home/martinkb/Desktop/BareTorch_F/debug_model_layers.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM


def inspect_tensor(name, tensor, pass_type="FORWARD"):
    if tensor is None:
        print(f"[{pass_type}] {name}: None")
        return
    if isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                inspect_tensor(f"{name}[tuple_{i}]", t, pass_type)
        return
    if not isinstance(tensor, torch.Tensor):
        return

    t_float = tensor.float()
    nan_cnt = torch.isnan(t_float).sum().item()
    inf_cnt = torch.isinf(t_float).sum().item()
    
    if nan_cnt > 0 or inf_cnt > 0:
        status = "🔴 INVALID (NaN/Inf)"
    else:
        status = "✅ OK"

    min_val = t_float.min().item() if t_float.numel() > 0 else 0.0
    max_val = t_float.max().item() if t_float.numel() > 0 else 0.0
    mean_val = t_float.mean().item() if t_float.numel() > 0 else 0.0

    print(
        f"[{pass_type:8s}] {status:20s} | {name:40s} | Shape: {str(list(tensor.shape)):20s} | "
        f"Min: {min_val:10.4f} | Max: {max_val:10.4f} | Mean: {mean_val:10.4f} | NaNs: {nan_cnt} | Infs: {inf_cnt}"
    )


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_path = "/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_sft/checkpoint-2759"

    print("=" * 120)
    print("🔍 BARETORCH LAYER-BY-LAYER NUMERICAL STABILITY DIAGNOSTIC")
    print("=" * 120)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    desired_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tokens_to_add = [t for t in desired_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    model = BareTorchForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to(device)
    
    # Verify embedding weights after resize
    print("\n--- 1. Embeddings Pre-Resize Check ---")
    inspect_tensor("token_embedding.weight (pre-resize)", model.get_input_embeddings().weight, "PARAM")

    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        print("\n--- 2. Embeddings Post-Resize Check ---")
        inspect_tensor("token_embedding.weight (post-resize)", model.get_input_embeddings().weight, "PARAM")

    model.train()

    # 2. Attach Forward and Backward Hooks
    print("\n--- 3. Attaching Layer Hooks ---")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            def make_fw_hook(mod_name):
                def fw_hook(m, inp, out):
                    inspect_tensor(f"out -> {mod_name}", out, "FORWARD")
                return fw_hook

            def make_bw_hook(mod_name):
                def bw_hook(m, grad_inp, grad_out):
                    inspect_tensor(f"grad_out -> {mod_name}", grad_out, "BACKWARD")
                return bw_hook

            module.register_forward_hook(make_fw_hook(name))
            module.register_full_backward_hook(make_bw_hook(name))

    # 3. Create Sample Batch
    dummy_text = (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n2+2=4\n</think>\n\\boxed{4}<|im_end|>\n"
    )
    inputs = tokenizer(dummy_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    print(f"\n--- 4. Running Forward Pass (Input Length: {input_ids.shape[1]}) ---")
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    print("\n--- 5. Loss Inspection ---")
    inspect_tensor("Loss", loss, "FORWARD")

    print("\n--- 6. Running Backward Pass ---")
    if loss is not None and not torch.isnan(loss):
        loss.backward()
    else:
        print("⚠️ Loss is NaN or None! Synthesizing dummy scalar loss to trace backward gradients...")
        dummy_loss = outputs.logits.sum() * 0.0
        dummy_loss.backward()

    print("\n--- 7. Parameter Gradient Inspection ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            inspect_tensor(f"grad -> {name}", param.grad, "GRADIENT")

    print("\n" + "=" * 120)
    print("✅ Diagnostic complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()