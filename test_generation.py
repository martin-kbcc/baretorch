import torch
from transformers import AutoTokenizer
from baretorch import BareTorchConfig, BareTorchForCausalLM

CHECKPOINT_PATH = "./checkpoint-90000"
TOKENIZER_NAME = "Qwen/Qwen3.5-9B"

print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Crucial for causal LM batched generation: pad on the left so new tokens append to the right
tokenizer.padding_side = "left"

print(f"Loading BareTorch 2.5B from {CHECKPOINT_PATH}...")
model = BareTorchForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

# ==============================================================================
# Test 1: Single Sequence Generation
# ==============================================================================
print("\n" + "=" * 60)
print("TEST 1: Single Sequence Generation (Unpadded)")
print("=" * 60)

single_prompt = "The key architectural advantage of linear recurrence over standard softmax attention is"
single_inputs = tokenizer(single_prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    single_output = model.generate(
        **single_inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n[Prompt]:", single_prompt)
print("[Completion]:", tokenizer.decode(single_output[0], skip_special_tokens=True))

# ==============================================================================
# Test 2: Batched Generation with Asymmetric Left-Padding
# ==============================================================================
print("\n" + "=" * 60)
print("TEST 2: Batched Generation with Asymmetric Left-Padding")
print("=" * 60)

batch_prompts = [
    # Short prompt (~4 tokens) -> Will be left-padded
    "Machine learning is",
    # Longer prompt (~18 tokens)
    "In modern deep learning architectures, hybrid recurrence and attention mechanisms enable models to achieve",
]

batch_inputs = tokenizer(
    batch_prompts,
    padding=True,
    return_tensors="pt",
).to("cuda:0")

print(f"Input batch shape: {batch_inputs['input_ids'].shape}")
print(f"Attention mask tensor:\n{batch_inputs['attention_mask'].cpu()}")

with torch.no_grad():
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n--- Batched Results ---")
for idx, out_tokens in enumerate(batch_outputs):
    decoded_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    print(f"\n[Sample {idx + 1} Original Prompt]: {batch_prompts[idx]}")
    print(f"[Sample {idx + 1} Output]:\n{decoded_text.strip()}")
print("\n" + "=" * 60)