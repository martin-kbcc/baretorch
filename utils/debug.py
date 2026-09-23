from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Automatically resolves the exact cached file path
file_path = hf_hub_download(
    repo_id="model-rampage/BareTorch-500M-Base", 
    filename="model.safetensors"
)

state_dict = load_file(file_path)

# Print all keys for layer 3 attention
for k in state_dict.keys():
    if "layers.3.attn" in k:
        print(k)