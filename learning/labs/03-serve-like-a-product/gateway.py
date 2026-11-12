"""Lab 03: a minimal LLM gateway (Module 13).

Sits in front of the inference server and adds what a raw model server
lacks:
  - identity:     every request needs an API key
  - rate limits:  requests per minute, per key
  - budgets:      tokens per day, per key
  - caps:         maximum prompt size and maximum output tokens
  - metering:     a log line per request with tokens and latency,
                  but never the prompt or answer text

Run (from this folder, with ../.venv active):
    cp keys.example.json keys.json
    uvicorn gateway:app --port 9000
"""

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

HERE = Path(__file__).parent
UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:8000")
KEYS = json.loads((HERE / os.environ.get("KEYS_FILE", "keys.json")).read_text())
LOG = HERE / "gateway.log"
MAX_OUTPUT_TOKENS = 512
MAX_PROMPT_CHARS = 16_000

app = FastAPI(title="Lab gateway")
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=300)
tokens_used = defaultdict(int)     # key -> tokens today (resets on restart)
recent = defaultdict(deque)        # key -> timestamps of recent requests


def authorise(request: Request) -> str:
    """Return the caller's key, or refuse the request."""
    key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if key not in KEYS:
        raise HTTPException(401, "missing or unknown API key")
    limits = KEYS[key]

    window = recent[key]
    now = time.time()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limits["requests_per_minute"]:
        raise HTTPException(429, "rate limit: too many requests per minute")
    if tokens_used[key] >= limits["tokens_per_day"]:
        raise HTTPException(429, "budget exhausted: daily token limit reached")
    window.append(now)
    return key


def apply_caps(body: dict) -> dict:
    """Refuse oversized prompts and cap the answer length."""
    prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    if prompt_chars > MAX_PROMPT_CHARS:
        raise HTTPException(413, f"prompt too long ({prompt_chars} characters)")
    asked = body.get("max_tokens") or MAX_OUTPUT_TOKENS
    body["max_tokens"] = min(asked, MAX_OUTPUT_TOKENS)
    return body


def record(key, status, usage, started):
    """Meter the request. Note what is NOT logged: the text itself."""
    usage = usage or {}
    tokens_used[key] += usage.get("total_tokens", 0)
    entry = {
        "time": round(started, 3),
        "caller": KEYS[key]["name"],
        "status": status,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "seconds": round(time.time() - started, 3),
        "used_today": tokens_used[key],
    }
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    key = authorise(request)
    body = apply_caps(await request.json())
    started = time.time()

    if not body.get("stream"):
        r = await client.post("/v1/chat/completions", json=body)
        data = r.json()
        record(key, r.status_code, data.get("usage"), started)
        return JSONResponse(data, status_code=r.status_code)

    # Streaming: pass tokens through as they arrive, and ask the server
    # to report usage in its final chunk so we can still meter it.
    body["stream_options"] = {"include_usage": True}

    async def relay():
        usage, status = None, 200
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            status = r.status_code
            async for line in r.aiter_lines():
                if line.startswith("data: {"):
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usage") or usage
                yield line + "\n"
        record(key, status, usage, started)

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.get("/v1/models")
async def models(request: Request):
    authorise(request)
    r = await client.get("/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/usage")
async def usage():
    """Tokens used per caller since the gateway started."""
    return {KEYS[k]["name"]: n for k, n in tokens_used.items()}
