# /Users/martinkovacevic/Desktop/baretorch/demo_mac_speed.py
import sys
import time
import argparse
import mlx.core as mx
from mlx.utils import tree_map
from transformers import AutoTokenizer

from baretorch.benchmarks.apple.modeling_mlx import BareTorchForCausalLMMLX
from baretorch.benchmarks.apple.benchmark_mlx import (
    find_matching_baretorch_config,
    count_mlx_params_m,
    HAS_MLX_LM,
)

if HAS_MLX_LM:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.models.cache import make_prompt_cache


def main():
    parser = argparse.ArgumentParser(description="BareTorch M1 Terminal Speed Demo")
    parser.add_argument("--mode", type=str, choices=["baretorch", "baseline"], required=True, help="Engine architecture")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-360M", help="Target HuggingFace baseline model ID")
    parser.add_argument("--layer_sequence", type=str, default="cs_lrad,cs_lrad,cs_lrad,transformer", help="Pattern for hybrid layers")
    parser.add_argument("--ctx_len", type=int, default=32768, help="Context prefill length in tokens")
    parser.add_argument("--gen_tokens", type=int, default=60, help="Number of tokens to decode")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay to prevent live GPU contention")
    args = parser.parse_args()

    if args.delay > 0:
        time.sleep(args.delay)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        vocab_size = getattr(tokenizer, "vocab_size", 49152)
    except Exception:
        tokenizer = None
        vocab_size = 49152

    max_seq_len = args.ctx_len + args.gen_tokens + 1024

    if args.mode == "baseline":
        print(f"\n📦 Loading Baseline Model '{args.model_id}' from MLX Hub...")
        if not HAS_MLX_LM:
            raise ImportError("`mlx_lm` is required to run the baseline model.")
        
        model, _ = mlx_lm_load(args.model_id)
        param_count_m = count_mlx_params_m(model)
        engine_title = f"BASELINE ({args.model_id})"
    else:
        target_params_m = 360.0
        print(f"\n⚙️  Matching BareTorch blueprint to ~{target_params_m:.1f}M baseline parameters...")
        bt_config, _ = find_matching_baretorch_config(
            target_params_m=target_params_m,
            target_vocab_size=vocab_size,
            layer_sequence=args.layer_sequence,
            max_seq_len=max_seq_len,
        )

        model = BareTorchForCausalLMMLX(bt_config)
        model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
        mx.eval(model.parameters())
        
        param_count_m = count_mlx_params_m(model)
        engine_title = "BARETORCH CS-LRAD (PURE GEMM)"

    print("\n" + "=" * 65)
    print(f"🚀 ENGINE: {engine_title}")
    print(f"⚙️  PARAM COUNT: {param_count_m:.1f}M Params | CONTEXT: {args.ctx_len} Tokens")
    print("=" * 65 + "\n")

    prompt_ids = mx.random.randint(0, vocab_size, (1, args.ctx_len))
    mx.eval(prompt_ids)

    print(f"⏳ Running Prefill ({args.ctx_len} tokens)...", end="", flush=True)
    t_prefill_start = time.perf_counter()

    if args.mode == "baseline":
        cache = make_prompt_cache(model)
        outputs = model(prompt_ids, cache=cache)
        mx.eval(outputs)
        curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)
        mx.eval(curr_token)
        del outputs
    else:
        # JIT Compile Prefill
        def prefill_step_bt(p):
            return model(p)

        compiled_prefill = mx.compile(prefill_step_bt)
        logits, cache = compiled_prefill(prompt_ids)
        mx.eval(logits, cache)
        curr_token = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(curr_token)

    prefill_time = time.perf_counter() - t_prefill_start
    print(f" Done ({prefill_time:.2f}s)\n")

    print("⚡ Streaming Decoded Tokens Live:\n" + "-" * 65)

    t_dec_start = time.perf_counter()

    if args.mode == "baseline":
        for _ in range(args.gen_tokens):
            outputs = model(curr_token, cache=cache)
            mx.eval(outputs)
            curr_token = mx.argmax(outputs[:, -1:, :], axis=-1)
            
            tok_id = curr_token.item()
            tok_str = tokenizer.decode([tok_id]) if tokenizer else f" {tok_id}"
            sys.stdout.write(tok_str)
            sys.stdout.flush()
    else:
        def decode_step_bt(tok, p_cache):
            step_logits, next_cache = model(tok, past_key_values=p_cache)
            next_tok = mx.argmax(step_logits[:, -1:, :], axis=-1)
            return next_tok, next_cache

        compiled_step = mx.compile(decode_step_bt)
        curr_token, cache = compiled_step(curr_token, cache)
        mx.eval(curr_token, cache)

        for _ in range(args.gen_tokens):
            curr_token, cache = compiled_step(curr_token, cache)
            mx.eval(curr_token, cache)

            tok_id = curr_token.item()
            tok_str = tokenizer.decode([tok_id]) if tokenizer else f" {tok_id}"
            sys.stdout.write(tok_str)
            sys.stdout.flush()

    dec_time = time.perf_counter() - t_dec_start
    tok_sec = args.gen_tokens / dec_time

    print("\n" + "-" * 65)
    print("=" * 65)
    print(f"📊 DECODING SPEED: {tok_sec:.2f} tok/s")
    print(f"⏱️  Total Time: {dec_time:.2f}s ({args.gen_tokens} tokens)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()