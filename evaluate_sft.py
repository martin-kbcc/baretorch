import argparse
import time
import torch
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM

BENCHMARK_PROMPTS = [
    {
        "category": "Reasoning & Math",
        "system": "You are a helpful, concise assistant.",
        "user": "If a train travels at 60 mph for 2.5 hours, how far does it travel? Show your work."
    },
    {
        "category": "Code Generation",
        "system": "You are an expert Python developer.",
        "user": "Write a short Python function to check if a string is a palindrome."
    },
    {
        "category": "Creative Writing",
        "system": "You are a creative story generator.",
        "user": "Write a two-sentence mystery story set in an abandoned space station."
    },
    {
        "category": "Multi-turn Format Test",
        "system": "You are a friendly AI companion.",
        "user": "Hello! Who are you and what can you help me with today?"
    },
]

def format_chatml(system_prompt: str, user_prompt: str) -> str:
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def parse_args():
    parser = argparse.ArgumentParser(description="BareTorch SFT Inference & Benchmarking Engine")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_100m_sft/checkpoint-1980", help="Path to SFT checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat mode after benchmark")
    return parser.parse_args()

def run_benchmark(model, tokenizer, device, args):
    print("\n" + "=" * 80)
    print("🚀 RUNNING AUTOMATED BENCHMARK SUITE")
    print("=" * 80)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    total_tokens_generated = 0
    total_latency_seconds = 0.0

    for idx, sample in enumerate(BENCHMARK_PROMPTS, 1):
        prompt_str = format_chatml(sample["system"], sample["user"])
        input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
        input_length = input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.1,
                use_cache=True,  
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time

        gen_tokens = output_ids.shape[1] - input_length
        tokens_per_sec = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=False)
        clean_text = generated_text.replace("<|im_end|>", "").strip()
        hit_eos = "<|im_end|>" in generated_text

        total_tokens_generated += gen_tokens
        total_latency_seconds += elapsed_sec

        print(f"\n--- [Test {idx}/{len(BENCHMARK_PROMPTS)}] Category: {sample['category']} ---")
        print(f"💬 Prompt  : {sample['user']}")
        print(f"🤖 Response:\n{clean_text}")
        print("-" * 50)
        print(f"⏱️  Latency   : {elapsed_sec:.2f}s | Tokens: {gen_tokens} | Speed: {tokens_per_sec:.2f} tok/s")
        print(f"💾 Peak VRAM : {peak_vram_gb:.2f} GB | Clean EOS Stop: {'✅ Yes' if hit_eos else '❌ Hit Max Tokens'}")

    avg_speed = total_tokens_generated / total_latency_seconds if total_latency_seconds > 0 else 0
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"• Total Generated Tokens : {total_tokens_generated}")
    print(f"• Total Inference Time   : {total_latency_seconds:.2f} seconds")
    print(f"• Average Throughput     : {avg_speed:.2f} tokens/sec")
    print("=" * 80 + "\n")

def run_interactive_chat(model, tokenizer, device, args):
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE CHAT MODE (Type 'exit' or 'quit' to stop)")
    print("=" * 80)

    system_prompt = "You are a helpful, knowledgeable, and polite AI assistant built with BareTorch."
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting interactive chat session. Goodbye!")
                break

            prompt_str = format_chatml(system_prompt, user_input)
            input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)

            print("🤖 Assistant: ", end="", flush=True)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=1.1,
                    use_cache=True,  
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )

            gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
            clean_response = gen_text.replace("<|im_end|>", "").strip()
            print(clean_response)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer and SFT model from: '{args.checkpoint_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"], "pad_token": "<|im_end|>"}
    tokenizer.add_special_tokens(special_tokens)

    model = BareTorchForCausalLM.from_pretrained(args.checkpoint_dir)
    model.to(device)
    model.eval()

    run_benchmark(model, tokenizer, device, args)

    if args.interactive:
        run_interactive_chat(model, tokenizer, device, args)

if __name__ == "__main__":
    main()