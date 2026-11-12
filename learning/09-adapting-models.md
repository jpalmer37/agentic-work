# 9. Adapting models

## Why this matters

A general-purpose model rarely does exactly what your product needs. It
does not know your documents, your formats or your tone. There are
several ways to close that gap, and they differ enormously in cost and
effort. Choosing the right one, in the right order, is one of the most
valuable skills in applied AI.

## The idea in plain words

Think of a ladder. Each rung is more powerful and more expensive than
the one below.

```text
5. Train your own model
     almost never
4. Fine-tune
     change the weights
3. Give it tools
     let it act
2. Give it data (retrieval)
     let it look things up
1. Prompt well
     ask clearly, show examples
```

**Start at the bottom. Climb only when you can show, with measurements,
that the current rung is not enough.** Most teams that jump straight to
fine-tuning discover later that a better prompt or retrieval would have
done the job.

## Diagnose before you choose

Ask *why* the model is failing. The cause points to the remedy.

| The problem | The remedy |
|---|---|
| It does not understand the task | Better prompt, examples |
| It lacks facts, or facts change | Retrieval |
| It needs to act or compute | Tools |
| Style or format is inconsistent | Examples, then fine-tuning |
| It is too slow or too expensive | Smaller fine-tuned model |

A short way to remember it: **retrieval changes what the model knows.
Fine-tuning changes how the model behaves.**

## Rung 1: Prompting

The prompt is the cheapest thing to change and often the most
effective. What reliably helps:

- **Say exactly what you want**, including the audience and purpose.
- **Supply the context** the model needs. It knows nothing about your
  situation except what you tell it.
- **Show examples** of good input and output. This is called
  **few-shot prompting**.
- **Specify the output format** precisely.
- **Give room to reason** on hard problems before the final answer.

Few-shot prompting works through attention. The model has learned,
during pretraining, to notice patterns earlier in its context and
continue them. No weights change. This ability to adapt from examples in
the prompt is called **in-context learning**.

## Rung 2: Retrieval (RAG)

**Retrieval-augmented generation** means: find relevant text first, put
it in the prompt, then ask the model to answer using it.

It is the right answer when the model needs knowledge it was not
trained on: your internal documents, recent events, a user's own data.

### The pipeline

```text
OFFLINE, once per document
  documents
    -> split into chunks
    -> embed each chunk
    -> store vectors in an index

ONLINE, once per question
  question
    -> embed
    -> find nearest chunks
    -> (rerank)
    -> build prompt: chunks + question
    -> generate answer with sources
```

The "find nearest" step is the dot product from
[Module 1](01-vectors-and-dot-products.md): score the question's vector
against every chunk's vector and take the highest.

```python
import numpy as np

def unit(M):
    n = np.linalg.norm(M, axis=-1,
                       keepdims=True)
    return M / n

# docs:  (N, d) chunk embeddings
# query: (d,)   question embedding
def top_k(query, docs, k=5):
    scores = unit(docs) @ unit(query)
    best = np.argsort(-scores)[:k]
    return best, scores[best]
```

For up to a few hundred thousand chunks, this brute-force search is fast
enough and exactly correct. Beyond that, a vector database uses an
**approximate nearest neighbour** index, which gives up a sliver of
accuracy for a large gain in speed.

### What makes retrieval work

**Chunking.** Split documents into pieces of a few hundred tokens. Too
large and the relevant sentence is diluted. Too small and it loses its
context. Split on natural boundaries such as headings and paragraphs,
and keep metadata (title, section, date) with each chunk.

**Hybrid search.** Embeddings capture meaning but can miss exact
strings: product codes, error numbers, names. Keyword search has the
opposite strengths. Combining both is more robust than either alone.

**Reranking.** A first pass retrieves, say, 50 candidates quickly. A
slower, more accurate **reranker** model then reads the question and
each candidate together and reorders them. The top 5 go to the LLM.

**Access control.** Filter by what the user is allowed to see **at
retrieval time**. If a chunk reaches the prompt, assume the user can
get the model to reveal it.

### Evaluate the two halves separately

When a RAG answer is wrong, there are two possible culprits:

1. **Retrieval failed:** the right chunk was not in the prompt.
2. **Generation failed:** the right chunk was there, and the model
   ignored or misread it.

They need different fixes. Measure retrieval alone first ("is the right
chunk in the top 5?"). Only then examine the answers.

## Rung 3: Tools

A model can only produce text. **Tool use**, also called function
calling, lets that text trigger actions.

1. You describe the available tools to the model.
2. The model replies with a structured request, such as
   `get_order_status(order_id="A123")`.
3. Your code runs the function.
4. You send the result back to the model.
5. The model continues, using the result.

The model never runs anything itself. It asks, and your code decides
whether to comply. That boundary is where you enforce permissions.

An **agent** is this loop repeated: the model plans, calls a tool, reads
the result, and decides what to do next, until the task is done.

Agents are powerful and need care:

- **Errors compound.** A small mistake early can derail everything
  after it. Keep tasks bounded and add checks.
- **Costs multiply.** Every step is another model call with a growing
  context.
- **Tool output is untrusted.** A web page or document the agent reads
  may contain text written to hijack it. This is **prompt injection**,
  covered in [Module 13](13-hosting-your-own-models.md).

## Rung 4: Fine-tuning

Fine-tuning continues training a pretrained model on your own examples,
so its weights shift toward your task.

**Good reasons to fine-tune:**

- You need a consistent style, tone or output format that prompting
  cannot hold reliably.
- You have a narrow task with many good examples.
- You want a small, cheap model to match a large one on that task.
- Your prompts have grown huge with instructions and examples, and you
  want that behaviour built in.

**Poor reasons:**

- To teach facts that change. Use retrieval.
- Because prompting "felt" insufficient, without measurements.
- When you have only a handful of examples.

### LoRA: fine-tuning without the cost

Updating every weight in a large model needs a lot of memory and yields
a full-size copy of the model for each task. **LoRA** (low-rank
adaptation) avoids both.

The idea: freeze the original weight matrix `W`. Learn a small
*correction* to it, built from two thin matrices.

```text
W_new = W + B @ A

W: (d, d)   frozen
A: (r, d)   trained
B: (d, r)   trained

r is small, for example 8
```

Count the weights for `d = 4096` and `r = 8`:

```text
full matrix   4096 * 4096
              = 16,777,216
LoRA          2 * 4096 * 8
              =     65,536

LoRA trains 0.4% as many weights.
```

Why can so few weights be enough? Adapting a model to a new task turns
out to need changes in only a few directions, not a rewrite of
everything. A low-rank update is exactly "a change in a few
directions".

`B` starts as all zeros, so at the beginning `B @ A` is zero and the
model is unchanged. Training then grows the correction from nothing.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base, r=8):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        out, inp = base.weight.shape
        A = torch.randn(r, inp) * 0.01
        B = torch.zeros(out, r)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        # strength of the correction
        self.scale = 16 / r

    def forward(self, x):
        delta = x @ self.A.T @ self.B.T
        y = self.base(x)
        return y + self.scale * delta
```

What LoRA gives you in practice:

- **Small files.** An adapter is megabytes, not gigabytes.
- **Many tasks, one base model.** Swap adapters without reloading.
- **No inference cost.** The correction can be merged into `W`.
- **Modest hardware.** Combined with a 4-bit base model, a technique
  called **QLoRA**, a multi-billion-parameter model can be fine-tuned
  on a single consumer GPU.

### Fine-tuning data

Quality matters far more than quantity.

- A few hundred to a few thousand excellent examples is often enough.
- Examples should look exactly like real inputs and ideal outputs.
- Format them with the model's own chat template.
- Hold some back for evaluation, and never train on those.
- Read your data. A model faithfully learns your mistakes too.

### Risks

- **Forgetting.** Tuning hard on a narrow task can erode general
  ability. Keep learning rates low and check broader behaviour.
- **Overfitting.** With little data, the model memorizes. Watch the
  validation loss.
- **Maintenance.** A fine-tuned model is an asset you must now
  re-evaluate and possibly retrain whenever the base model changes.

## Distillation

Use a large, capable **teacher** model to produce high-quality outputs
for many inputs from your task. Then fine-tune a small **student** model
on those outputs.

The student ends up nearly as good as the teacher *on that task*, at a
fraction of the cost and latency. This is one of the most practical ways
to make a production system cheap: prototype with the best model
available, then distill once the task is stable.

## Rung 5: Training your own

Training a language model from scratch needs enormous data, compute and
expertise. For nearly every organization it is the wrong choice.
Consider it only for a truly different kind of data, such as a new
scientific modality, where no pretrained model exists.

## Putting it together

The approaches combine. A mature system often uses all of them:

- A carefully written **prompt**,
- with **retrieved** documents inserted,
- and **tools** for live data and actions,
- running on a small model **fine-tuned** or distilled for the task.

## Common mistakes

- **Fine-tuning first.** It is the slowest loop to iterate on. Exhaust
  prompting and retrieval before it.
- **No evaluation set.** Without one you cannot tell if any change
  helped. See [Module 11](11-evaluation.md).
- **Blaming the model for a retrieval failure.** Check what was actually
  in the prompt before changing anything else.
- **Chunking blindly.** Splitting every 500 characters cuts sentences
  and tables in half.
- **Using the wrong template when fine-tuning.** The model must be
  trained and served with the same formatting.
- **Trusting tool output.** Treat anything fetched from outside as data,
  never as instructions.

## Check yourself

<details>
<summary>1. Your support bot gives outdated answers about pricing.
Retrieval or fine-tuning?</summary>

Retrieval. Prices are facts that change. Keep them in a document store
and retrieve the current version. Fine-tuning would bake in whatever was
true on training day.
</details>

<details>
<summary>2. Your model's answers are correct but ignore your required
JSON structure about one time in ten. What do you try, in order?</summary>

First tighten the prompt and add examples. Then use constrained decoding
so the structure is guaranteed. Fine-tune only if those are not enough.
</details>

<details>
<summary>3. A LoRA adapter with r = 16 is applied to a 4096 by 4096
matrix. How many weights are trained?</summary>

`2 * 4096 * 16 = 131,072`, compared with 16.8 million in the full
matrix. That is under 1%.
</details>

<details>
<summary>4. Why does LoRA start with B set to zero?</summary>

So that the correction `B @ A` is zero at the start and the model begins
exactly as the pretrained one. Training then moves it away gradually.
</details>

<details>
<summary>5. A RAG answer is wrong. What is the first thing to
check?</summary>

Whether the chunk containing the right answer was actually retrieved and
placed in the prompt. That tells you whether to fix retrieval or
generation.
</details>

<details>
<summary>6. When does distillation make sense?</summary>

When a large model already does your task well but is too slow or
costly to run at volume. Its outputs become training data for a small
model that is cheap to serve.
</details>

> **Try it.** [Lab 04](labs/04-rag-over-your-docs/README.md) builds retrieval over this course, and [Lab 05](labs/05-lora-fine-tune/README.md) fine-tunes a small model with LoRA.

## Takeaways

- Climb the ladder from the bottom: prompt, retrieve, tools, fine-tune.
- Retrieval changes what a model knows. Fine-tuning changes how it
  behaves.
- RAG is embeddings plus dot-product search plus a good prompt.
- Tools let a model act, with your code as the gatekeeper.
- LoRA trains a tiny low-rank correction instead of the whole model.
- Distill a large model into a small one once the task is stable.

---

[← Inference and generation](08-inference-and-generation.md) ·
[Course home](README.md) ·
[Next: Classical ML and data science →](10-classical-ml-and-data-science.md)
