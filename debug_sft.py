# /home/martinkb/Desktop/BareTorch_F/debug_sft.py
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM
from train_sft import pack_chatml_dataset

# 1. Setup Device & Precision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
print(f"🔍 Running SFT Debugger on {device} using {dtype}...")

# 2. Tokenizer & Model Initialization
tokenizer_name = "HuggingFaceTB/SmolLM2-360M"
checkpoint_path = "./checkpoints_500m_hybrid_baretorch/checkpoint-190735"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    "pad_token": "<|im_end|>",
})

model = BareTorchForCausalLM.from_pretrained(checkpoint_path).to(device=device, dtype=dtype)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.tie_weights()
model.train()

# 3. Apply Targeted Torch Compile
print("⚡ Applying torch.compile to CS-LRAD sub-modules...")
compiled_blocks = 0
for name, module in model.named_modules():
    cls_name = module.__class__.__name__.lower()
    if "lrad" in cls_name or "lrad" in name.lower():
        module.forward = torch.compile(module.forward)
        compiled_blocks += 1
print(f"Compiled {compiled_blocks} CS-LRAD sub-modules.")

# 4. Fetch 50 Samples & Pack Sequence
print("📦 Fetching 50 sample dialogues...")
raw_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train").select(range(50))
packed_ds = pack_chatml_dataset(raw_ds, tokenizer, max_seq_len=2048, local_rank=0)

batch_input_ids = torch.tensor(packed_ds[:2]["input_ids"]).to(device)
batch_labels = torch.tensor(packed_ds[:2]["labels"]).to(device)

print(f"\n📊 Batch Tensor Diagnostics:")
print(f"  ├─ Input IDs Shape : {batch_input_ids.shape} | Range: [{batch_input_ids.min()}, {batch_input_ids.max()}]")
print(f"  └─ Target Labels   : Valid Tokens = {(batch_labels != -100).sum().item()} / {batch_labels.numel()}")

# 5. Step-by-Step Execution Verification
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

for step in range(1, 6):
    optimizer.zero_grad()
    
    # Forward Pass
    outputs = model(input_ids=batch_input_ids, labels=batch_labels)
    loss = outputs.loss
    logits = outputs.logits

    # Check Logits & Loss for NaN / Inf
    has_nan_logits = torch.isnan(logits).any().item()
    has_nan_loss = torch.isnan(loss).any().item()
    
    print(f"\n--- Step {step} Execution ---")
    print(f"  ├─ Logits Stats    : Min={logits.min().item():.2f}, Max={logits.max().item():.2f}, HasNaN={has_nan_logits}")
    print(f"  ├─ Calculated Loss : {loss.item():.4f} (HasNaN={has_nan_loss})")

    # Backward Pass
    loss.backward()
    
    # Check Gradient Norms Across Model Parameters
    total_grad_norm = 0.0
    nan_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_grads += 1
            else:
                total_grad_norm += p.grad.detach().data.norm(2).item() ** 2
        else:
            print(f"⚠️ Warning: Grad is None for parameter: {name}")

    total_grad_norm = total_grad_norm ** 0.5
    print(f"  └─ Gradient Norm   : {total_grad_norm:.4f} (NaN Param Count: {nan_grads})")

    optimizer.step()

print("\n✅ Debug session finished.")