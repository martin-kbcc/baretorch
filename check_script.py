import torch
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM

checkpoint = "./checkpoints_100m_sft/checkpoint-1980"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"], "pad_token": "<|im_end|>"}
tokenizer.add_special_tokens(special_tokens)

model = BareTorchForCausalLM.from_pretrained(checkpoint).to(device).eval()
model._supports_cache_class = False

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a short Python function to check if a string is a palindrome.<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print("--- TESTING WITH CACHE (use_cache=True) ---")
out_cache = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7, repetition_penalty=1.2, use_cache=True)
print(tokenizer.decode(out_cache[0][input_ids.shape[1]:], skip_special_tokens=False))

print("\n--- TESTING WITHOUT CACHE (use_cache=False) ---")
out_nocache = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7, repetition_penalty=1.2, use_cache=False)
print(tokenizer.decode(out_nocache[0][input_ids.shape[1]:], skip_special_tokens=False))