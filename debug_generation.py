# /home/martinkb/Desktop/BareTorch_F/debug_generation.py
import torch
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM

CHECKPOINT_PATH = "./2RTX_300M_runs/checkpoints_300M_sft/checkpoint-3584"
TOKENIZER_NAME = "Qwen/Qwen3.5-9B"

print("🔍 Initializing Debug Suite...")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"📦 Loading model from {CHECKPOINT_PATH}...")
model = BareTorchForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
).cuda().eval()

if getattr(model.config, "tie_word_embeddings", False):
    model.tie_weights()

model.config.use_cache = True

# Hook into forward to print exact layer inputs during generate()
step_counter = [0]
orig_forward = model.model.forward

def debug_forward(*args, **kwargs):
    step = step_counter[0]
    p_kv = kwargs.get("past_key_values", None)
    input_ids = kwargs.get("input_ids", None)
    seq_len = input_ids.shape[1] if input_ids is not None else "None"
    
    print(f"\n--- [FORWARD STEP {step}] input_ids length: {seq_len} ---")
    if p_kv is None:
        print("  past_key_values: None (Prefill Phase)")
    else:
        print(f"  past_key_values length: {len(p_kv)}")
        for idx, layer_cache in enumerate(p_kv):
            layer_type = model.config.layer_types[idx]
            if layer_cache is None:
                print(f"    Layer {idx:<2} ({layer_type:<11}): None")
            elif isinstance(layer_cache, tuple):
                types_str = [f"{type(x).__name__}" for x in layer_cache]
                shapes_str = [f"{list(x.shape)}" if isinstance(x, torch.Tensor) else str(type(x)) for x in layer_cache]
                print(f"    Layer {idx:<2} ({layer_type:<11}): Tuple len={len(layer_cache)} | Shapes: {shapes_str}")
            else:
                print(f"    Layer {idx:<2} ({layer_type:<11}): Bare {type(layer_cache).__name__} | Shape: {list(layer_cache.shape) if isinstance(layer_cache, torch.Tensor) else 'N/A'}")
                
    step_counter[0] += 1
    return orig_forward(*args, **kwargs)

model.model.forward = debug_forward

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n🚀 Triggering model.generate(max_new_tokens=5)...")
try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            use_cache=True,
        )
    print("\n✅ Generation Completed Successfully!")
    print("Generated Text:\n", tokenizer.decode(outputs[0]))
except Exception as e:
    print(f"\n❌ GENERATION CRASHED WITH ERROR:\n{e}")