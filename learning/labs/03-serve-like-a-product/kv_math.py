"""Lab 03: predict memory use from the model's own config (Module 13).

Reads config.json for the model vLLM downloaded and works out:
  - weight memory at 16-bit
  - KV cache bytes per token
  - how many tokens of cache fit in the budget you gave vLLM
  - how many full-length requests can run at once

Compare the last two with the lines vLLM prints in its log.

    python kv_math.py
    python kv_math.py --gpu-gb 12 --utilization 0.80 --max-len 4096
"""

import argparse
import glob
import json
import os

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def find_config(model):
    folder = "models--" + model.replace("/", "--")
    hits = glob.glob(os.path.join(HUB, folder, "snapshots", "*", "config.json"))
    if not hits:
        raise SystemExit(f"No config for {model} under {HUB}. "
                         "Start vLLM first so it downloads the model.")
    with open(hits[0]) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gpu-gb", type=float, default=12.0)
    p.add_argument("--utilization", type=float, default=0.80)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--overhead-gb", type=float, default=1.0,
                   help="rough allowance for activations and CUDA graphs")
    args = p.parse_args()

    c = find_config(args.model)
    d = c["hidden_size"]
    layers = c["num_hidden_layers"]
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    head_size = c.get("head_dim") or d // heads
    vocab = c["vocab_size"]
    mlp = c["intermediate_size"]

    # Parameter count for a Llama/Qwen-style block (Module 7): attention
    # (Q and output: d*d each; K and V: d * kv_heads*head_size each) plus
    # a three-matrix SwiGLU MLP, plus embeddings (tied or not).
    attn = 2 * d * d + 2 * d * kv_heads * head_size
    per_block = attn + 3 * d * mlp
    tied = c.get("tie_word_embeddings", False)
    params = layers * per_block + vocab * d * (1 if tied else 2)
    weights_gb = params * 2 / 1e9  # 16-bit = 2 bytes per weight

    kv_bytes = 2 * layers * kv_heads * head_size * 2  # K and V, 16-bit
    budget_gb = args.gpu_gb * args.utilization
    cache_gb = budget_gb - weights_gb - args.overhead_gb
    cache_tokens = cache_gb * 1e9 / kv_bytes

    print(f"model                 {args.model}")
    print(f"layers, d             {layers}, {d}")
    print(f"query / KV heads      {heads} / {kv_heads}  (grouped-query, Module 7)")
    print(f"parameters            about {params / 1e9:.2f}B")
    print(f"weights at 16-bit     about {weights_gb:.1f} GB")
    print()
    print(f"KV cache per token    2 * {layers} * {kv_heads} * {head_size} * 2 bytes")
    print(f"                      = {kv_bytes:,} bytes ({kv_bytes / 1024:.0f} KB)")
    print()
    print(f"vLLM's budget         {args.gpu_gb} GB * {args.utilization} = {budget_gb:.1f} GB")
    print(f"left for KV cache     about {cache_gb:.1f} GB "
          f"(after weights and ~{args.overhead_gb} GB overhead)")
    print(f"tokens of cache       about {cache_tokens:,.0f}")
    print(f"full {args.max_len}-token requests at once   "
          f"about {cache_tokens / args.max_len:.0f}")
    print()
    print("Now compare with vLLM's own figures:")
    print("  docker compose logs vllm | grep -i 'kv cache\\|concurrency'")


if __name__ == "__main__":
    main()
