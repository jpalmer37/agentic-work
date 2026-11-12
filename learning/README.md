# ML Deep Dive

A course for software and data engineers who know the basics of machine
learning and want to understand what is really going on underneath, all
the way to running models in production.

## Who this is for

You have trained a model or two. You know roughly what a neural network,
a loss, and a transformer are. You can read Python.

What you want now is the next layer down:

- Why does backpropagation work, and what does it cost?
- Why is attention divided by a square root?
- What actually differs between one LLM architecture and another?
- How do you host a model yourself without it falling over?

Each module answers questions like these in plain language, with small
numbers you can check by hand and code you can run.

## How to read it

- Each module is a 10 to 15 minute read, written to be comfortable on a
  phone. Formulas are plain text. Code and diagrams are narrow.
- Modules build on each other, so the first pass is best in order.
  After that, each one stands alone as a reference.
- Every module ends with **Check yourself** questions. Tap a question to
  reveal the answer. If you can answer them without peeking, move on.

## Course map

### Part 1: Foundations

1. [Vectors and dot products](01-vectors-and-dot-products.md):
   the one operation that everything else is built from.
2. [Loss functions](02-loss-functions.md):
   how a model is told what "wrong" means.
3. [Backpropagation](03-backpropagation.md):
   how blame flows backward through a network.
4. [Optimization and training](04-optimization-and-training.md):
   turning gradients into a model that works.

### Part 2: Transformers and LLMs

5. [Attention](05-attention.md):
   queries, keys, values, and the square root.
6. [Transformers](06-transformers.md):
   the full block, piece by piece.
7. [LLM architectures](07-llm-architectures.md):
   what modern large language models changed, and why.
8. [Inference and generation](08-inference-and-generation.md):
   how text comes out, and what makes it fast or slow.

### Part 3: Applying models

9. [Adapting models](09-adapting-models.md):
   prompting, retrieval, and fine-tuning, and when to use each.
10. [Classical ML and data science](10-classical-ml-and-data-science.md):
    the toolkit that still wins on tables and experiments.
11. [Evaluation](11-evaluation.md):
    knowing whether a change made things better.

### Part 4: Production

12. [MLOps](12-mlops.md):
    treating models like software that also depends on data.
13. [Hosting your own models](13-hosting-your-own-models.md):
    sizing, serving, cost, and safety for internal and public apps.

### Wrap-up

14. [Projects and cheat sheet](14-projects-and-cheatsheet.md):
    five hands-on projects, the key formulas on one page, a glossary.

## Hands-on labs

The modules are for reading. The [labs](labs/README.md) are for doing,
at your PC: train a tiny GPT, run a real model locally and remove it,
serve one behind a gateway and load-test it, build retrieval over this
course, and fine-tune with LoRA. They are written for Windows with WSL2
and an NVIDIA GPU, and each lab ends by cleaning up after itself.

## Notation

The course avoids heavy maths notation. A few conventions:

| Written | Means |
|---|---|
| `x`, `w` | a vector (a list of numbers) |
| `W` | a matrix (a grid of numbers) |
| `a · b` | dot product of `a` and `b` |
| `W @ x` | matrix multiply, as in Python |
| `d` | the size of a vector, its "dimension" |
| `ln` | natural logarithm |
| `(B, T, d)` | array shape: batch, tokens, dimension |

## Running the code

The NumPy snippets run as written with Python 3.10 or later:

```bash
pip install numpy
```

A few snippets use PyTorch (`pip install torch`). They are marked, and
none of them need a GPU.

## A note on dates

The ideas in Parts 1 to 3 are stable and will stay true for years.
Tools named in Part 4 are examples, checked in November 2026. That part
of the field moves quickly, so learn the pattern and confirm the
specific tool before you commit to it.

---

[Start with Module 1: Vectors and dot products →](01-vectors-and-dot-products.md)
