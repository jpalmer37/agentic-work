# 14. Projects and cheat sheet

Reading builds a map. Building things makes it yours. This page has
five projects that exercise every module, followed by a one-page
reference and a glossary.

## Five projects

They rise in difficulty. Each fits in a weekend, and none needs more
than a laptop for the first three.

### Project 1: Train a network with your own autograd

**Exercises:** Modules 2, 3, 4.

Start from the `Value` class in
[Module 3](03-backpropagation.md).

1. Add subtraction, power and `tanh`.
2. Build a `Neuron`, a `Layer` and a small network from `Value`s.
3. Generate a toy dataset: 100 points in 2D in two classes that a
   straight line cannot separate.
4. Write a training loop with plain gradient descent.
5. Plot the loss, and the decision boundary as it forms.

**You are done when** the loss falls smoothly and the network separates
the two classes.

**Stretch:** check your gradients numerically. Nudge one weight by a
tiny amount, measure the change in loss, and compare with the gradient
your engine reports.

### Project 2: A tiny transformer from scratch

**Exercises:** Modules 5, 6, 8.
**Walkthrough:** [Lab 01](labs/01-train-a-tiny-gpt/README.md).

Train a character-level language model on a small text file, such as a
book from the public domain.

1. Tokenize by character. Your vocabulary will be under 100 symbols.
2. Write the attention function yourself, including the causal mask
   and the `sqrt(d)`.
3. Assemble blocks: about 4 layers, `d = 128`, 4 heads.
4. Train with AdamW, warmup and cosine decay.
5. Generate text at temperatures 0.5, 1.0 and 1.5.

**You are done when** the first logged loss is close to
`ln(vocabulary size)`, it falls well below that, and the samples look
like the source text.

**Stretch:** remove the `sqrt(d)` scaling and compare training curves.
Then remove positional information and watch what the model can no
longer do.

### Project 3: Retrieval with real evaluation

**Exercises:** Modules 1, 9, 11.
**Walkthrough:** [Lab 04](labs/04-rag-over-your-docs/README.md).

Build a question-answering system over documents you know well: a
project's documentation, a handbook, your own notes.

1. Chunk the documents on headings and paragraphs.
2. Embed the chunks and search them with a NumPy dot product.
3. Write 30 questions, each with the chunk that answers it.
4. Measure retrieval: how often is the right chunk in the top 5?
5. Add generation. Grade answers for faithfulness to the sources.
6. Change one thing (chunk size, hybrid search, a reranker) and
   measure again.

**You are done when** you can state two numbers, retrieval hit rate and
answer pass rate, and show how one change moved them.

**Stretch:** categorize every failure and fix the largest category.

### Project 4: Fine-tune with LoRA and prove it helped

**Exercises:** Modules 4, 9, 11.
**Walkthrough:** [Lab 05](labs/05-lora-fine-tune/README.md).

Pick a narrow task with a clear right answer, such as extracting fields
from messy text into JSON.

1. Collect or create 300 to 500 examples. Hold back 100 for testing.
2. Measure a baseline: the same small model with a good prompt only.
3. Fine-tune with LoRA, using the model's chat template.
4. Evaluate on the held-back set. Compare with the baseline.
5. Quantize the result to 4-bit and evaluate again.

**You are done when** you have a table with three rows (prompted,
fine-tuned, fine-tuned and quantized) and can say whether the
differences are larger than the noise.

**Stretch:** distill. Use a large model to label more data, retrain,
and see whether the small model improves.

### Project 5: Serve a model like a product

**Exercises:** Modules 8, 12, 13.
**Walkthrough:** [Lab 02](labs/02-run-a-local-llm/README.md) and [Lab 03](labs/03-serve-like-a-product/README.md).

1. Run an open-weight model behind an inference server that speaks the
   standard API.
2. Put a small gateway in front: API keys, a per-key token limit, and
   request logging.
3. Load-test at 1, 4, 16 and 64 concurrent requests. Record time to
   first token, tokens per second, and 95th-percentile latency.
4. Work out memory use and the cost per million tokens at each level.
5. Add an eval that runs automatically before any prompt or model
   change.

**You are done when** you can draw the throughput-versus-concurrency
curve and name the concurrency where latency becomes unacceptable.

**Stretch:** add a second model and route between them. Add input and
output checks, then try to break them.

## Cheat sheet

### Core formulas

```text
dot product
  a · b = sum of a_i * b_i
        = |a| |b| cos(angle)

cosine
  (a · b) / (|a| |b|)

softmax
  exp(z_i) / sum of exp(z_j)

cross-entropy
  loss      -ln(p_correct)
  gradient  p - y
  at start  ln(number of classes)

perplexity
  exp(loss)

MSE
  mean of (pred - target)^2
```

### Training

```text
gradient descent
  w = w - lr * grad

Adam
  m = avg(grad)
  v = avg(grad^2)
  w = w - lr * m / (sqrt(v) + eps)

residual
  out = x + f(x)
```

### Transformers

```text
attention
  softmax(Q K^T / sqrt(d)) V

block
  x = x + attention(norm(x))
  x = x + mlp(norm(x))

parameters
  about 12 * layers * d^2
  plus embeddings

cost
  attention grows with T^2
  MLP grows with T
```

### Adapting

```text
LoRA
  W_new = W + B @ A
  trains 2 * d * r weights
  per matrix
```

### Serving

```text
weights memory
  params * bytes per param

KV cache per token
  2 * layers * kv_heads
    * head_size * bytes

GPU memory
  weights
  + cache * concurrency
  + overhead

cost per million tokens
  hourly price
  / million tokens per hour
  / utilization
```

### Statistics

```text
standard error of a rate
  sqrt(p * (1 - p) / n)

95% range
  p +/- 1.96 * SE
```

### Numbers worth remembering

| Fact | Value |
|---|---|
| One token of English | About 4 characters |
| Memory at 16-bit | 2 GB per billion weights |
| Memory at 4-bit | About 0.6 GB per billion |
| Starting loss, 50k vocab | About 10.8 |
| Eval of 200 cases | About +/- 5 points |
| Eval of 1,000 cases | About +/- 2 points |
| Gradient clip | 1.0 |
| LR, fine-tuning | 1e-5 to 5e-5 |
| LR, LoRA | 1e-4 to 3e-4 |

## Glossary

**Activation.** The output of a layer. Also the non-linear function
applied to it, such as ReLU or GELU.

**Agent.** A model in a loop that plans, calls tools, reads results and
decides what to do next.

**Attention.** The mechanism by which each token gathers information
from other tokens, weighted by dot-product similarity.

**Autoregressive.** Generating one token at a time, each conditioned on
all the tokens before it.

**Backpropagation.** The algorithm that computes the gradient of the
loss for every weight in one backward pass.

**Base model.** A model that has only been pretrained. It continues
text and has not been taught to follow instructions.

**Batch.** A group of examples processed together.

**Causal mask.** The mask that stops a token attending to later
tokens.

**Chat template.** The special tokens and layout that mark roles in a
conversation for a particular model.

**Context window.** The maximum number of tokens a model can attend to
at once.

**Continuous batching.** Re-forming the batch at every token step so
requests can join and leave freely.

**Cross-entropy.** The loss `-ln(p_correct)`. The standard for
classification and language modelling.

**Distillation.** Training a small model to imitate a large one.

**Drift.** A change over time in input data, or in the relationship
between inputs and correct outputs.

**Embedding.** A learned vector representing a token, sentence, image
or other item.

**Eval.** A fixed set of inputs with a scoring method, used to track
quality.

**Fine-tuning.** Continuing to train a pretrained model on new data.

**Gradient.** How much the loss changes when a weight changes slightly.

**GQA.** Grouped-query attention. Several query heads share one set of
keys and values, shrinking the KV cache.

**Inference.** Running a trained model to get outputs.

**KV cache.** Saved keys and values for tokens already processed, so
they are not recomputed during generation.

**Leakage.** Information reaching the model in training that would not
be available at prediction time.

**Logits.** Raw scores before softmax.

**LoRA.** Low-rank adaptation. Fine-tuning by training two small
matrices whose product is added to a frozen weight matrix.

**Loss.** A single number measuring how wrong the model is.

**MLP.** The feed-forward layer in a transformer block.

**MoE.** Mixture of experts. Many feed-forward layers, with only a few
chosen per token.

**Overfitting.** Memorizing training data instead of learning the
general pattern.

**Perplexity.** `exp(loss)`. The effective number of choices the model
is unsure between.

**Prefill.** The parallel pass over the prompt at the start of
inference.

**Prompt injection.** Text in a model's input that hijacks its
behaviour.

**Quantization.** Storing weights in fewer bits to save memory.

**RAG.** Retrieval-augmented generation. Finding relevant text and
placing it in the prompt before generating.

**Residual connection.** Adding a layer's input to its output:
`x + f(x)`.

**Residual stream.** The per-token vector that runs through a
transformer and that every layer adds to.

**RMSNorm.** A simple normalization that divides a vector by its
root-mean-square size.

**RoPE.** Rotary position embedding. Encodes position by rotating
queries and keys.

**Softmax.** Turns a list of scores into probabilities that sum to 1.

**Temperature.** A divisor applied to logits before sampling. Lower is
more conservative.

**Token.** A piece of text, often part of a word, that the model treats
as one unit.

**Top-p.** Sampling only from the smallest set of tokens whose
probabilities reach a threshold.

**Weight decay.** Shrinking weights slightly toward zero on every
step, as regularization.

## Further reading

Search for these by title. All have stood the test of time.

**Foundations**

- *Neural Networks: Zero to Hero*, video series by Andrej Karpathy.
  Builds an autograd engine and a GPT from nothing.
- *Neural Networks*, video series by 3Blue1Brown. The best visual
  intuition for backpropagation and attention.
- *Dive into Deep Learning*, by Zhang, Lipton, Li and Smola. A free
  book with runnable code.
- *A Recipe for Training Neural Networks*, essay by Andrej Karpathy.

**Transformers and LLMs**

- *Attention Is All You Need*, Vaswani and others, 2017. The original
  paper. Readable after Modules 5 and 6.
- *The Illustrated Transformer*, by Jay Alammar.
- *LoRA: Low-Rank Adaptation of Large Language Models*, Hu and others,
  2021.

**Production**

- *Designing Machine Learning Systems* and *AI Engineering*, both by
  Chip Huyen.
- *Hidden Technical Debt in Machine Learning Systems*, Sculley and
  others, 2015.
- *Rules of Machine Learning*, by Martin Zinkevich.
- *OWASP Top 10 for Large Language Model Applications*.

**Experiments**

- *Trustworthy Online Controlled Experiments*, by Kohavi, Tang and Xu.

## Where to go from here

Pick the project closest to your work and build it. Then come back to
the module that felt least solid and read it again. It will read
differently once you have code that depends on it.

---

[← Hosting your own models](13-hosting-your-own-models.md) ·
[Course home](README.md)
