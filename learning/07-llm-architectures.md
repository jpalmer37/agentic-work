# 7. LLM architectures

## Why this matters

When you choose a model, read a model card, or size hardware, you meet a
wall of terms: RMSNorm, SwiGLU, RoPE, GQA, MoE, "8B", "128k context".
Each is a small, understandable decision. This module decodes them, and
explains the training stages that turn a raw network into an assistant.

## The idea in plain words

A modern large language model is the decoder-only transformer from
[Module 6](06-transformers.md), scaled up, with about half a dozen
refinements.

The surprising truth is that architectures across model families are
very similar. What separates a strong model from a weak one is mostly
**data, scale and training recipe**, not a secret block design.

## How an LLM is made: the stages

### Stage 1: Pretraining

Train on a vast amount of text with one objective: predict the next
token ([Module 2](02-loss-functions.md)). This is where almost all the
compute goes.

The result is a **base model**. It does not answer questions. It
continues documents. Give it a question and it may reply with another
question, because that is a plausible continuation.

### Stage 2: Instruction tuning

Fine-tune the base model on examples of conversations: an instruction,
then a good response. The model learns the *format* of being helpful.

Conversations are marked up with special tokens that separate the roles.
This markup is the **chat template**:

```text
<|system|> You are a helpful assistant.
<|user|> What is a dot product?
<|assistant|> A dot product is ...
```

Every model family uses different markers. Using the wrong template is
one of the most common reasons a self-hosted model gives poor answers.

### Stage 3: Preference tuning

Teach the model which of several possible answers people prefer. Two
common approaches:

- **RLHF** (reinforcement learning from human feedback): train a
  separate model to score answers the way human raters do, then tune the
  LLM to score well.
- **DPO** (direct preference optimization): skip the separate scorer and
  train directly on pairs of "better answer, worse answer".

### Stage 4: Reinforcement learning on tasks

Increasingly, models are also trained on problems with checkable
outcomes: code that passes tests, maths with a known answer. The model
tries, gets scored, and is updated toward what worked. This is a major
source of recent gains in reasoning and coding.

## What changed since the original transformer

| Part | Original (2017) | Typical now |
|---|---|---|
| Layout | Encoder-decoder | Decoder-only |
| Norm position | After sub-layer | Before (pre-norm) |
| Norm type | LayerNorm | RMSNorm |
| Activation | ReLU | GELU or SwiGLU |
| Position | Sinusoidal, added | Rotary (RoPE) |
| Attention | Full multi-head | Grouped-query |
| MLP | Dense | Dense, or mixture of experts |

Each is explained below.

### RMSNorm

LayerNorm subtracts the mean and divides by the standard deviation.
**RMSNorm** skips the mean and just divides by the root-mean-square size
of the vector:

```text
rmsnorm(x) = scale * x / rms(x)

rms(x) = sqrt(mean of x^2)
scale  = a learned weight
```

It is simpler, slightly faster and works just as well.

### SwiGLU

The feed-forward layer's activation moved from ReLU to smoother
functions. **SwiGLU** adds a gate: one projection of the input decides
how much of another projection gets through.

```text
gate   = silu(W1 @ x)
mlp(x) = W3 @ (gate * (W2 @ x))
```

It uses three matrices instead of two, so the hidden size is reduced to
keep the weight count about the same. It gives a small, consistent
improvement in quality.

### Rotary positions (RoPE)

Covered in [Module 6](06-transformers.md): queries and keys are rotated
by position so attention depends on the distance between tokens.

### Grouped-query attention (GQA)

In standard attention every head has its own queries, keys and values.
In **grouped-query attention**, several query heads share one set of
keys and values.

```text
standard:  32 query heads
           32 key/value heads

grouped:   32 query heads
            8 key/value heads
```

Quality barely changes. The benefit shows up at inference: the cache of
keys and values that a model keeps while generating becomes four times
smaller in this example. [Module 8](08-inference-and-generation.md)
explains why that cache matters so much.

### Mixture of experts (MoE)

Replace the single feed-forward layer in each block with many smaller
ones, called **experts**, plus a small **router**. For each token, the
router picks a few experts to run and skips the rest.

```text
                +--> expert 1
token -> router-+--> expert 2
                +--> ...
                +--> expert 16

(only the top 2 of 16 are run)
```

The model has many weights in total but uses only a fraction for any one
token. That gives two different sizes to keep straight:

- **Total parameters:** set memory needs. All experts must be loaded.
- **Active parameters:** set compute per token. Only chosen experts run.

MoE models offer the knowledge of a large model at closer to the running
cost of a small one, in exchange for more memory and more complicated
serving.

## Reading a model's configuration

Open-weight models ship a `config.json`. Here are the fields that
matter, with values from a well-known 8-billion-parameter model:

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  "max_position_embeddings": 8192
}
```

| Field | Meaning |
|---|---|
| `hidden_size` | `d`, the residual stream width |
| `num_hidden_layers` | Number of blocks |
| `num_attention_heads` | Query heads per block |
| `num_key_value_heads` | Key/value heads (GQA) |
| `intermediate_size` | MLP hidden width |
| `vocab_size` | Number of tokens |
| `max_position_embeddings` | Context window in tokens |

You can now check the advertised size yourself. Head size is
`4096 / 32 = 128`, so the shared keys and values are `8 * 128 = 1024`
wide.

```text
attention, per block
  Q and output   2 * 4096 * 4096
                 =  33.6M
  K and V        2 * 4096 * 1024
                 =   8.4M
MLP, per block (three matrices)
                 3 * 4096 * 14336
                 = 176.2M

per block        218.1M
32 blocks        6.98B

embedding and unembedding
                 2 * 128256 * 4096
                 = 1.05B

total            8.03B
```

The "8B" in the model's name is exactly this arithmetic.

## Context window

The context window is the maximum number of tokens the model can attend
to at once: your prompt plus everything it generates.

It is limited by three things:

- **Training.** A model only handles lengths it has learned to handle.
- **Memory.** The key/value cache grows with every token.
- **Compute.** Attention cost grows with the square of length.

A large window does not mean perfect recall. Models tend to use
information at the start and end of a long prompt better than
information buried in the middle. Putting the important material near
the end, and keeping prompts focused, still pays off.

## Scaling

Language models improve in a remarkably predictable way as you increase
three things together: **parameters, data and compute**. Loss falls
smoothly as each one grows. These regularities are called
**scaling laws**.

Two practical lessons:

- **Balance matters.** A huge model on too little data wastes its
  capacity. Size and data should grow together.
- **Small models keep improving.** With better data and longer training,
  today's small models outperform much larger ones from a few years ago.
  Do not assume you need the biggest model.

## Reasoning models

Each generated token gets a fixed amount of computation: one trip
through the stack of blocks. A hard problem may need more steps than
that.

**Reasoning models** are trained to write out intermediate work before
giving a final answer. The architecture is the same. The model simply
uses extra tokens as a scratchpad, and because each new token can
attend to everything written so far, a chain of tokens becomes a chain
of computation.

The trade-off is direct: better answers on hard problems, in exchange
for more output tokens, higher cost and longer waits. For simple tasks
it is wasted effort.

## Multimodal models

To let a language model see images, turn the image into things that look
like tokens:

1. Cut the image into small square patches.
2. Pass the patches through an image encoder, itself a transformer.
3. Project the encoder's output vectors to the LLM's width `d`.
4. Place them in the sequence alongside the text token vectors.

From there the LLM attends to image vectors exactly as it attends to
words. Audio and video follow the same pattern. The recurring idea is
that **everything becomes a sequence of vectors**, and the transformer
does not care where they came from.

## Generative models and embedding models

These are different tools built from the same parts.

| | Generative LLM | Embedding model |
|---|---|---|
| Output | Next-token probabilities | One vector per input |
| Trained with | Next-token cross-entropy | Contrastive loss |
| Used for | Writing, reasoning, chat | Search, clustering |
| Typical size | Billions of weights | Often far smaller |

A retrieval system usually uses both: an embedding model to find
relevant text, then a generative model to write the answer
([Module 9](09-adapting-models.md)).

## Open weights and hosted APIs

**Hosted models** are reached through an API. You send text and get
text. You never see the weights.

**Open-weight models** can be downloaded and run on your own hardware.
When you evaluate one, read these on its model card:

- **Size** in parameters, and whether it is a mixture of experts.
- **Context window.**
- **License.** Some forbid commercial use or add conditions.
- **Chat template.** Use the one it was trained with.
- **Available quantized versions**
  ([Module 8](08-inference-and-generation.md)).
- **Intended use and known limits.**

[Module 13](13-hosting-your-own-models.md) covers how to choose between
hosting yourself and using an API.

## Common misconceptions

- **"The model looks answers up."** There is no database inside. What it
  knows is spread across its weights, which is why it can be confidently
  wrong.
- **"More parameters means smarter."** Only when data and training keep
  up. A well-trained small model often beats a careless large one.
- **"A bigger context window means it remembers everything."** It means
  it *can* attend to more. Recall degrades in long, cluttered prompts.
- **"The model knows about itself."** It only knows what was in its
  training data or prompt. Ask it about its own architecture and it will
  guess.
- **"Base models and chat models are interchangeable."** A base model
  continues text. A chat model expects its template.

## Check yourself

<details>
<summary>1. What is the difference between a base model and a chat
model?</summary>

A base model has only been pretrained to continue text. A chat model has
additionally been tuned on conversations, using a specific template of
role markers, so that it responds to instructions.
</details>

<details>
<summary>2. What does grouped-query attention save, and when?</summary>

Several query heads share one set of keys and values, so the cache of
keys and values kept during generation is several times smaller. The
saving is in inference memory.
</details>

<details>
<summary>3. A mixture-of-experts model has 60B total parameters and 10B
active. Which number sizes your GPU memory, and which sets speed?</summary>

Memory is set by the total, because every expert must be loaded. Speed
per token is set mainly by the active count, since only the chosen
experts run.
</details>

<details>
<summary>4. Why can writing out intermediate reasoning improve
answers?</summary>

Each token gets one fixed pass through the network. Writing steps out
lets later tokens build on earlier ones, so the model performs a longer
chain of computation than a single pass allows.
</details>

<details>
<summary>5. How does a language model "see" an image?</summary>

The image is cut into patches, encoded into vectors, and projected to
the same width as token vectors. Those vectors are placed in the
sequence and attended to like any other token.
</details>

<details>
<summary>6. You self-host a model and its answers are rambling and
oddly formatted. What do you check first?</summary>

The chat template. If the prompt is not wrapped in the role markers the
model was trained with, it behaves more like a base model continuing
text.
</details>

## Takeaways

- Modern LLMs are decoder-only transformers with small refinements.
- Training is staged: pretrain, instruction-tune, preference-tune, and
  often reinforcement learning on checkable tasks.
- RMSNorm, SwiGLU, RoPE and GQA are tidy upgrades to the same block.
- Mixture of experts separates total parameters from active ones.
- You can verify a model's size from its config with simple arithmetic.
- Data and training recipe matter as much as architecture.

---

[← Transformers](06-transformers.md) ·
[Course home](README.md) ·
[Next: Inference and generation →](08-inference-and-generation.md)
