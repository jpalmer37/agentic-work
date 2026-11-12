"""Lab 02: measure a local model served by Ollama.

Shows the two phases of inference from Module 8:
  prefill  - the whole prompt in one parallel pass (fast per token)
  decode   - one token at a time (slow per token)

Then calls the same model through the OpenAI-compatible API, the
standard interface from Module 13.

    python llm_stats.py
    python llm_stats.py --model llama3.2:3b
"""

import argparse
import time

import httpx

OLLAMA = "http://localhost:11434"

SHORT = "In one sentence, what is a dot product?"
# A long prompt: the same paragraph repeated, to make prefill visible.
PARAGRAPH = (
    "Attention lets each token gather information from other tokens. "
    "Each token asks a question with its query, advertises what it has "
    "with its key, and hands over its value when chosen. "
)
LONG = PARAGRAPH * 60 + "\nSummarise the text above in one sentence."


def native_generate(model, prompt, max_tokens=120):
    """Ollama's own API returns detailed timings, in nanoseconds."""
    r = httpx.post(f"{OLLAMA}/api/generate", timeout=600, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    })
    r.raise_for_status()
    return r.json()


def report(label, res):
    s = 1e9  # nanoseconds per second
    load = res.get("load_duration", 0) / s
    p_n = res.get("prompt_eval_count", 0)
    p_t = res.get("prompt_eval_duration", 1) / s
    d_n = res.get("eval_count", 0)
    d_t = res.get("eval_duration", 1) / s
    print(f"--- {label}")
    print(f"load model     {load:6.2f} s")
    print(f"prefill        {p_n:5d} tokens in {p_t:6.2f} s "
          f"= {p_n / p_t:8.0f} tokens/s")
    print(f"decode         {d_n:5d} tokens in {d_t:6.2f} s "
          f"= {d_n / d_t:8.0f} tokens/s")
    print(f"answer         {res['response'].strip()[:150]!r}")
    print()


def openai_style(model):
    """The same model through the standard OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama")  # key ignored
    start = time.perf_counter()
    first = None
    pieces = []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name three uses of embeddings."}],
        max_tokens=120,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter() - start
            pieces.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - start
    print("--- OpenAI-compatible API, streaming")
    print(f"time to first token  {first:.2f} s")
    print(f"total time           {total:.2f} s")
    print("answer:")
    print("".join(pieces).strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2:3b")
    args = p.parse_args()

    try:
        httpx.get(OLLAMA, timeout=3)
    except httpx.HTTPError:
        print("Ollama is not running. Start it: sudo systemctl start ollama")
        return

    # First call may include loading the model into GPU memory.
    report("short prompt (first call: includes loading)",
           native_generate(args.model, SHORT))
    report("short prompt (model already loaded)",
           native_generate(args.model, SHORT))
    report("long prompt", native_generate(args.model, LONG))
    openai_style(args.model)


if __name__ == "__main__":
    main()
