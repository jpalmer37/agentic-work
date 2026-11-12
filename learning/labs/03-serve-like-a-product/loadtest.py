"""Lab 03: load-test an OpenAI-compatible server at rising concurrency.

For each concurrency level it sends a batch of streaming requests at
once and reports what Module 8 says matters:
  time to first token (TTFT), per-user decode speed, total throughput,
  and the slow tail (p95), not just the average.

    python loadtest.py                                  # vLLM directly
    python loadtest.py --url http://localhost:9000 \
        --key lab-key-loadtest                          # via the gateway
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx


def pct(values, q):
    """The q-th percentile (0 to 100) of a list."""
    values = sorted(values)
    i = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[i]


async def one_request(client, args, i):
    body = {
        "model": args.model,
        # A slightly different prompt each time, so nothing is cached.
        "messages": [{"role": "user", "content":
                      f"Request {i}: write a short story about a robot "
                      "who learns to garden. About 150 words."}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first, tokens, chunks = None, 0, 0
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            await r.aread()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        async for line in r.aiter_lines():
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                chunks += 1
                if first is None:
                    first = time.perf_counter() - start
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
    total = time.perf_counter() - start
    tokens = tokens or chunks  # fall back if the server sent no usage
    first = first if first is not None else total
    decode = tokens / max(total - first, 1e-6)
    return dict(ttft=first, total=total, tokens=tokens, decode=decode)


async def run_level(args, concurrency):
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=args.url, headers=headers,
                                 timeout=600, limits=limits) as client:
        n = max(args.min_requests, concurrency * 2)
        sem = asyncio.Semaphore(concurrency)

        async def limited(i):
            async with sem:
                return await one_request(client, args, i)

        start = time.perf_counter()
        results = await asyncio.gather(*(limited(i) for i in range(n)))
        wall = time.perf_counter() - start

    return dict(
        concurrency=concurrency,
        requests=n,
        ttft_p50=pct([r["ttft"] for r in results], 50),
        ttft_p95=pct([r["ttft"] for r in results], 95),
        latency_p95=pct([r["total"] for r in results], 95),
        per_user=statistics.mean(r["decode"] for r in results),
        throughput=sum(r["tokens"] for r in results) / wall,
    )


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", default="", help="API key, if using the gateway")
    p.add_argument("--model", default="lab-model")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--min-requests", type=int, default=8)
    p.add_argument("--save", default="results.json")
    args = p.parse_args()

    rows = []
    print(f"{'users':>5} {'TTFT p50':>9} {'TTFT p95':>9} {'p95 total':>10} "
          f"{'tok/s/user':>11} {'tok/s all':>10}")
    for c in args.levels:
        r = await run_level(args, c)
        rows.append(r)
        print(f"{c:>5} {r['ttft_p50']:>8.2f}s {r['ttft_p95']:>8.2f}s "
              f"{r['latency_p95']:>9.2f}s {r['per_user']:>11.0f} "
              f"{r['throughput']:>10.0f}")

    with open(args.save, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nSaved {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
