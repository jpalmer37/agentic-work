# 8. Inference and generation

## Why this matters

A model is trained once and then run millions of times. Running it is
called **inference**, and it is where your latency, your bill and your
users' experience are decided. A few ideas explain almost everything
about why inference is fast or slow, cheap or expensive.

## The idea in plain words

A language model does not write a response. It produces a probability
for every possible next token. Something outside the model picks one,
appends it to the text, and asks again.

```python
tokens = tokenize(prompt)
while not finished(tokens):
    # scores for the next token
    logits = model(tokens)[-1]
    next_id = sample(logits)
    tokens.append(next_id)
```

That loop is called **autoregressive generation**. One token per trip
round the loop, each depending on all the tokens before it. Every
performance topic in this module is a consequence of that loop.

## Choosing the next token

The model gives probabilities. How you pick from them shapes the output.

### Greedy

Always take the most likely token. Predictable and repeatable, but
prone to dull, repetitive text.

### Temperature

Divide the logits by a number `T` before the softmax.

```text
logits = [2.0, 1.0, 0.1]

T = 0.5   [0.864, 0.117, 0.019]
          sharper
T = 1.0   [0.659, 0.242, 0.099]
          unchanged
T = 2.0   [0.502, 0.304, 0.194]
          flatter
```

Low temperature exaggerates the gaps and makes the model more
conservative. High temperature shrinks them and makes it more varied.
A temperature of 0 means greedy.

This is the same lever as the `sqrt(d)` in attention
([Module 5](05-attention.md)): scaling the inputs to a softmax controls
how peaked the output is.

### Top-k and top-p

Even at moderate temperature, the long tail of unlikely tokens
occasionally gets picked and derails the text. Two filters cut the tail
before sampling.

**Top-k:** keep only the `k` most likely tokens.

**Top-p** (also called nucleus sampling): keep the smallest set of
tokens whose probabilities add up to at least `p`.

```text
probability   running total
   0.636          0.636
   0.234          0.871
   0.086          0.957   <- stop
   0.032          0.988
   0.012          1.000

top_p = 0.9 keeps the first three
```

Top-p adapts itself: when the model is sure, it keeps one or two
tokens. When the model is unsure, it keeps many.

### Practical settings

| Task | Temperature |
|---|---|
| Extraction, classification, code | 0 to 0.3 |
| General question answering | 0.3 to 0.7 |
| Brainstorming, creative writing | 0.7 to 1.0 |

Two cautions. Temperature 0 makes output *repeatable*, not *correct*.
And it is not perfectly repeatable either: tiny floating-point
differences between hardware or batch sizes can still change a token.

### In code

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

rng = np.random.default_rng(0)

def sample(logits, temp=1.0, top_p=1):
    if temp == 0:
        return int(np.argmax(logits))
    p = softmax(logits / temp)
    order = np.argsort(-p)
    total = np.cumsum(p[order])
    n = np.searchsorted(total, top_p)
    keep = order[: n + 1]
    q = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=q))
```

## The KV cache

Look at the loop again. On every step it runs the model over the
**entire** sequence to get one new token. Generating 1,000 tokens that
way would reprocess the early tokens a thousand times.

The causal mask rescues us. A token can only attend to earlier tokens,
so once a token has been processed, **nothing that comes later can
change its keys and values**. They can be computed once and saved.

That saved store is the **KV cache**. With it, each step only has to:

1. Run the one new token through each layer.
2. Compute its query, and attend over the cached keys and values.
3. Append its own key and value to the cache.

Per-token work drops from "reprocess everything" to "process one token
and look back". Every serious inference engine does this.

### The cache costs memory

The cache holds a key and a value for every token, in every layer.

```text
bytes per token =
  2 * layers * kv_heads
    * head_size * bytes_per_number
```

For the 8-billion-parameter model from
[Module 7](07-llm-architectures.md), with 32 layers, 8 key/value heads
of size 128, in 16-bit numbers:

```text
2 * 32 * 8 * 128 * 2 bytes
  = 131,072 bytes
  = 0.125 MB per token

8,000 tokens of context
  = 1 GB per conversation
```

Without grouped-query attention (32 key/value heads instead of 8), that
would be 4 GB per conversation.

This is the practical reason grouped-query attention exists, and it
means GPU memory must cover **the weights plus a cache for every
request being served at once**.

## Two phases: prefill and decode

Inference on one request has two distinct stages.

**Prefill.** The whole prompt is processed in a single parallel pass to
fill the KV cache. Thousands of tokens are handled together, so it is
fast per token.

**Decode.** Output tokens are produced one at a time. Each needs its own
pass. This is slow per token and cannot be parallelized within a single
response.

```text
prompt:  [==========]
         one parallel pass

output:  [=][=][=][=][=]
         one token at a time
```

This is why hosted APIs charge more for output tokens than input
tokens, and why the fastest way to cut latency is usually to ask for a
shorter answer.

### The metrics that matter

| Metric | What drives it |
|---|---|
| Time to first token | Queueing plus prefill (prompt length) |
| Tokens per second, per user | Decode speed |
| Throughput, all users | Batching efficiency |
| Total latency | First token + output length / speed |

Always look at the slow tail (the 95th or 99th percentile), not only
the average. Averages hide the requests your users complain about.

### Prompt caching

If many requests begin with the same text, such as a long system prompt
or a shared document, the KV cache for that prefix can be reused instead
of recomputed. Cost and time to first token both fall.

Caching matches from the start of the prompt, so **put stable content
first and the part that changes last.**

## Batching

A GPU is built to do a great deal of arithmetic in parallel. A single
request uses a small part of that capacity. Running many requests
together in a **batch** costs only a little more time per step and
multiplies the total tokens produced.

The old approach waited to collect a batch, ran it, and returned
everything at the end. That is poor for language models, because
responses have different lengths and a short reply would wait for the
longest one.

**Continuous batching** fixes this. The batch is re-formed at every
token step. Finished responses leave at once. New requests join as soon
as there is room.

```text
step:   1  2  3  4  5  6  7  8
req A:  x  x  x  x  .  .  .  .
req B:  x  x  x  x  x  x  x  x
req C:  .  .  x  x  x  x  .  .
req D:  .  .  .  .  x  x  x  x

A finishes at step 4.
C joins at step 3.
D takes A's slot at step 5.
```

This single technique is the main reason a dedicated serving engine can
deliver many times the throughput of a simple loop. It is the first
thing to look for when choosing one
([Module 13](13-hosting-your-own-models.md)).

The trade-off is between **throughput** and **per-user speed**. More
requests in the batch means more total tokens per second and a lower
cost per token, but each individual stream slows a little.

## Quantization

A weight is normally stored as a 16-bit number. **Quantization** stores
it in fewer bits: 8, or 4. The model becomes smaller, and often faster
too, because generating a token means reading every weight once, and
there are now fewer bytes to read.

Memory for an 8-billion-parameter model:

| Precision | Bytes per weight | Size |
|---|---|---|
| 32-bit float | 4 | 32 GB |
| 16-bit float | 2 | 16 GB |
| 8-bit | 1 | 8 GB |
| 4-bit | about 0.6 | about 5 GB |

Effect on quality, as a rough guide:

- **16-bit:** the standard. No meaningful loss.
- **8-bit:** nearly indistinguishable from 16-bit.
- **4-bit:** a small, usually acceptable drop.
- **Below 4-bit:** quality falls away quickly.

A useful rule: for a fixed memory budget, **a larger model at 4-bit
usually beats a smaller model at 16-bit.**

Always measure on your own task. Quantization hurts some tasks more
than others, and averages hide that.

## Other ways to go faster

**Speculative decoding.** A small, fast model drafts several tokens
ahead. The large model checks them all in one parallel pass and keeps
the ones it agrees with. When the draft is right, which is often, you
get several tokens for the price of one large-model step. Done properly
the output is the same as the large model alone would produce.

**Distillation.** Train a small model to imitate a large one on your
task ([Module 9](09-adapting-models.md)). The best speedup is a model
that is simply smaller.

**Shorter prompts and outputs.** Not glamorous, and often the largest
saving available.

## Structured output

Applications often need valid JSON, not prose. Asking nicely works most
of the time. **Constrained decoding** makes it work every time.

At each step, the engine works out which tokens would keep the output
valid under your schema, and sets the logit of every other token to
negative infinity before sampling.

It is the same masking trick as the causal mask in attention: invalid
choices are given zero probability. The model can only produce text
that fits the schema.

## Knowing when to stop

Generation ends when one of these happens:

- The model emits its special **end-of-sequence** token.
- A **maximum token** limit is reached.
- A **stop sequence** you supplied appears in the output.

Always set a sensible maximum. A model that fails to stop will happily
generate until the limit, and you pay for every token.

## Streaming

Tokens can be sent to the user as they are produced, instead of waiting
for the whole response. Total time is unchanged, but the user starts
reading within a moment. For interactive applications this is the
difference between feeling instant and feeling broken.

## Common mistakes

- **Benchmarking with one request at a time.** That measures the worst
  case for a GPU. Test at realistic concurrency.
- **Reporting only the average latency.** Track the 95th and 99th
  percentiles.
- **Ignoring output length.** Decode dominates. A prompt change that
  doubles the answer length doubles the wait.
- **Comparing tokens per second across models.** Different tokenizers
  split text differently. Compare time to complete the same task.
- **Leaving the maximum token limit very high.** One runaway response
  can hold a batch slot for a long time.
- **Quantizing without re-evaluating.** Check quality on your own data.

## Check yourself

<details>
<summary>1. Why can keys and values be cached during
generation?</summary>

Because of the causal mask, a token never attends to later tokens. Once
it has been processed, its keys and values at every layer are final, so
they can be stored and reused for every subsequent step.
</details>

<details>
<summary>2. A model has 24 layers, 4 key/value heads of size 128, and
uses 16-bit numbers. How much cache does a 10,000-token conversation
need?</summary>

Per token: `2 * 24 * 4 * 128 * 2 = 49,152` bytes. Times 10,000 tokens
is about 492 MB.
</details>

<details>
<summary>3. Why are output tokens more expensive than input
tokens?</summary>

Input tokens are processed together in one parallel prefill pass. Output
tokens must be generated one at a time, each needing its own pass
through the model.
</details>

<details>
<summary>4. What does continuous batching improve over simple
batching?</summary>

Requests join and leave the batch at every token step. Short responses
return as soon as they finish, and new requests do not wait for the
whole batch to complete, so the GPU stays full and latency stays low.
</details>

<details>
<summary>5. You have 24 GB of GPU memory. Would you rather run a
12B-parameter model at 16-bit or a 30B-parameter model at 4-bit?</summary>

Usually the larger model at 4-bit. It needs roughly 18 to 20 GB, leaving
room for the cache, and a bigger model with mild quantization generally
outperforms a smaller one at full precision. Confirm on your own task.
</details>

<details>
<summary>6. How does constrained decoding guarantee valid JSON?</summary>

At each step it masks every token that would break the schema by
setting its logit to negative infinity, so only valid continuations can
be sampled.
</details>

## Takeaways

- Generation is a loop: predict, pick, append, repeat.
- Temperature and top-p control how adventurous the picks are.
- The KV cache avoids reprocessing, and costs memory per request.
- Prefill is parallel and cheap. Decode is serial and expensive.
- Continuous batching is what makes serving efficient.
- Quantization trades a little quality for a lot of memory.

---

[← LLM architectures](07-llm-architectures.md) ·
[Course home](README.md) ·
[Next: Adapting models →](09-adapting-models.md)
