# Lab 03: Serve it like a product

## Goal

Run a model the way a company would: on a high-throughput inference
server, behind a gateway that knows who is calling, under load. Then
take it all down.

## What you will see

- vLLM loading a model and reporting how much KV cache it has.
- Your own prediction of that number, from the model's config file.
- A gateway refusing callers without a key, over their rate limit, or
  over their token budget.
- Throughput climbing steeply with concurrency while per-user speed
  barely drops: continuous batching at work.

## Time and hardware

One to two hours. Needs the NVIDIA GPU and Docker from Lab 00, and about
20 GB of disk for the container image and the model.

## Read first

- [Module 8: Inference](../../08-inference-and-generation.md), for
  the KV cache, batching and the latency metrics.
- [Module 12: MLOps](../../12-mlops.md), for monitoring.
- [Module 13: Hosting](../../13-hosting-your-own-models.md), for
  sizing, the gateway and cost.

## The pieces

```text
loadtest.py / curl
      |
gateway.py      port 9000
      |   keys, limits, metering
vLLM            port 8000
      |   batching, KV cache
GPU
```

**vLLM** runs in Docker. It serves `Qwen/Qwen2.5-1.5B-Instruct`, an
open model with 1.5 billion parameters and the Apache 2.0 licence, at
16-bit. Checked November 2026.

**The gateway** is about 120 lines of Python in `gateway.py`. Real
gateways do more, but every idea here is the real one.

## Steps

### 1. Start vLLM

```bash
cd ~/agent-skills/learning/labs
source .venv/bin/activate
cd 03-serve-like-a-product
docker compose up -d
docker compose logs -f
```

The first start downloads the vLLM image (several GB) and the model
(about 3 GB), then loads the model and warms up. Expect 5 to 15
minutes. Later starts take about a minute.

It is ready when the log shows the server listening on port 8000.
Press Ctrl+C to stop watching. The server keeps running.

### 2. Talk to it directly

```bash
curl localhost:8000/v1/models
```

```bash
curl \
  localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "lab-model",
       "messages": [{"role": "user",
         "content": "Say hello."}]}'
```

Look at `usage` in the reply: tokens in, tokens out.

### 3. Predict the memory, then check

```bash
python kv_math.py
```

The script reads the model's `config.json` and works out the weight
size, the KV cache per token, and how many tokens of cache should fit
in the share of memory vLLM was given. The result looks like this:

```text
KV cache per token
  2 * 28 * 2 * 128 * 2 bytes
  = 28,672 bytes (28 KB)
tokens of cache   about 190,000
full 4096-token requests
  at once         about 47
```

Now find vLLM's own figures:

```bash
docker compose logs vllm \
  | grep -i 'kv cache\|concurrency'
```

Your prediction should land in the same range. The gap is the overhead
the script only estimates.

### 4. Start the gateway

```bash
cp keys.example.json keys.json
python -m uvicorn gateway:app \
  --port 9000
```

Leave it running and open a second terminal (activate `.venv` and `cd`
into this folder again).

### 5. Try to get past it

No key:

```bash
curl -i \
  localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "lab-model",
       "messages": [{"role": "user",
         "content": "Hi"}]}'
```

You get `401`. Now as Alice, who may make 10 requests a minute and
use 2,000 tokens a day:

```bash
URL=localhost:9000/v1/chat/completions
AUTH="Bearer lab-key-alice"
CT='Content-Type: application/json'
BODY='{"model": "lab-model",
  "messages": [{"role": "user",
    "content": "Write a poem."}]}'

for i in $(seq 1 12); do
  curl -s -o /dev/null \
    -w '%{http_code} ' \
    -H "Authorization: $AUTH" \
    -H "$CT" \
    -d "$BODY" $URL
done
```

You see several `200`s, then `429`. Whether her budget or her rate
limit ran out first depends on how long the poems were. Check:

```bash
curl localhost:9000/usage
tail -n 3 gateway.log
```

The log records who called, the tokens and the time taken. It never
records what was asked or answered.

### 6. Load-test

Through the gateway, with the load-test key:

```bash
python loadtest.py \
  --url http://localhost:9000 \
  --key lab-key-loadtest
```

This sends requests at 1, 4, 16 and 64 at a time. Write down the
table. The columns are:

| Column | Meaning |
|---|---|
| TTFT p50, p95 | Time to first token |
| p95 total | Slowest 5% of requests |
| tok/s/user | Speed each user sees |
| tok/s all | Total output of the GPU |

### 7. Turn batching down (optional)

Edit `docker-compose.yml` and add two lines at the end of `command`:

```yaml
      - --max-num-seqs
      - "1"
```

This lets vLLM work on only one request at a time. Restart and test:

```bash
docker compose up -d
python loadtest.py \
  --levels 1 4 16
```

Compare the `tok/s all` column with step 6, then remove the two lines
and run `docker compose up -d` again.

### 8. Work out the cost

Take your best `tok/s all` figure, and a price for renting a similar GPU
by the hour. Using [Module 13](../../13-hosting-your-own-models.md):

```text
tokens per hour
  = tok/s all * 3600
cost per million tokens
  = hourly price
    / (tokens per hour / 1e6)
```

Then divide by the share of the day it would really be busy.

## What to notice

**Throughput grows almost in step with users.** Going from 1 to 16
users multiplies total tokens per second many times over, while each
user's speed falls only a little. One request leaves most of the GPU
idle. Continuous batching fills it.

**Then it stops growing.** Somewhere between 16 and 64 users, total
throughput flattens and time to first token climbs. The GPU is full and
requests start to queue. That is your capacity.

**Batching off is dramatic.** With `--max-num-seqs 1`, total
throughput stays at the single-user figure no matter how many users
arrive. They just wait longer.

**The KV cache sets the limit.** Weights took about 3 GB. Most of the
rest went to the cache, and the cache size divided by request length is
how many requests can be served at once. A shorter `--max-model-len`
means more room for concurrent requests.

**The gateway is cheap.** Compare a load test straight to port 8000
with one through port 9000. The extra time is tiny next to the model's.

**Budgets are checked before the request.** A caller just under their
limit can still make one long request that takes them over. Real
gateways reserve the maximum output in advance for this reason.

## Spin down and clean up

**1. Stop the gateway:** Ctrl+C in its terminal.

**2. Stop vLLM and free the GPU:**

```bash
docker compose down
nvidia-smi
```

GPU memory should be back to where it was before the lab.

**3. Delete the image** (several GB):

```bash
docker image rm \
  vllm/vllm-openai:v0.33.0
```

**4. Delete the model.** The container wrote it as root:

```bash
sudo rm -rf ~/.cache/huggingface/\
hub/models--Qwen--Qwen2.5-1.5B-Instruct
```

**5. Remove local files:**

```bash
rm -f keys.json gateway.log \
  results.json
```

## Troubleshooting

**`could not select device driver`.** Docker cannot see the GPU. Re-run
the GPU check in step 8 of Lab 00.

**vLLM exits with an out-of-memory error.** Something else is using the
GPU, often Ollama from Lab 02. Run `ollama ps`, then
`sudo systemctl stop ollama`. Or lower `--gpu-memory-utilization` to
`0.70` or `--max-model-len` to `2048`.

**The log stops for a long time on first start.** It is downloading.
Check disk activity, or wait. Later starts reuse the download.

**`connection refused` on port 9000.** The gateway is not running. Start
it as in step 4.

**`KeyError` or `FileNotFoundError: keys.json`.** Copy
`keys.example.json` to `keys.json`, as in step 4.

---

[← Lab 02: Run a local LLM](../02-run-a-local-llm/README.md) ·
[Labs home](../README.md) ·
[Next: Lab 04, RAG over your docs →](../04-rag-over-your-docs/README.md)
