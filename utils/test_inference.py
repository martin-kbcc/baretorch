import torch
from transformers import AutoTokenizer
from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

model_id = "model-rampage/BareTorch-500M-Base"

# Ensure tokenizer matches the vocabulary size the 500M model was trained on
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

model = BareTorchForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
).cuda()

prompt = "The key innovation of pure GEMM sub-quadratic architectures is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))