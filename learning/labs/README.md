# Hands-on labs

The [course](../README.md) explains how things work. These labs let
you watch them work on your own machine: train a small model, run a real
language model, serve it like a product, and then take it all down
again.

The course is for reading anywhere. The labs are for sitting at your PC.
Each lab's README still reads comfortably on a phone, so you can look
ahead on the train and do the steps later.

## What you need

The labs are written for:

- **Windows 10 or 11 with WSL2** (Ubuntu).
- **An NVIDIA GPU with about 12 GB of memory**, such as an RTX 4070.
- **About 40 GB of free disk** if you do every lab. Model downloads and
  the serving container are large. Each lab ends by deleting what it
  downloaded.

Labs 01 and 04 also run on a CPU, just more slowly.

## The labs

| Lab | You will | Time |
|---|---|---|
| [00 Setup](00-setup/README.md) | Check GPU, Python, Docker | 30 min |
| [01 Train a tiny GPT](01-train-a-tiny-gpt/README.md) | Train a model from nothing | 30 min |
| [02 Run a local LLM](02-run-a-local-llm/README.md) | Run a real model, then remove it | 30 min |
| [03 Serve like a product](03-serve-like-a-product/README.md) | Serve, protect and load-test | 1 to 2 h |
| [04 RAG over your docs](04-rag-over-your-docs/README.md) | Search and answer from the course | 1 h |
| [05 LoRA fine-tune](05-lora-fine-tune/README.md) | Fine-tune and prove it helped | 1 to 2 h |

Do Lab 00 first. After that, each lab stands alone, although 04 is
easiest straight after 02, because it uses the same local model server.

## How each lab is laid out

Every lab README has the same sections:

1. **Goal**: what you will build.
2. **What you will see**: the result.
3. **Time and hardware**.
4. **Read first**: the course modules it puts into practice.
5. **Steps**: numbered commands to run.
6. **What to notice**: where your numbers connect to the theory.
7. **Spin down and clean up**: how to remove everything.
8. **Troubleshooting**: common problems, especially on WSL2.

The Python scripts sit next to each README. You do not need to read
them to do the lab, but they are short and commented if you want to.

## Where to run commands

Unless a step says otherwise, run every command in your **WSL2 Ubuntu
terminal**, from inside the lab's folder:

```bash
cd learning/labs/01-train-a-tiny-gpt
```

All the labs share one Python environment, created in Lab 00 at
`learning/labs/.venv`. Activate it whenever you open a new terminal:

```bash
source ../.venv/bin/activate
```

## How these labs were checked

The Python in Labs 01, 04 and 05 was run end to end on a CPU before
publishing. Labs 04 and 05 used small stand-ins for the real models,
because the test machine could not download them.

The GPU, Ollama and Docker steps in Labs 00, 02, 03 and 05 could not be
run on the test machine. They follow the official documentation as of
November 2026: Ollama, vLLM, Docker Desktop and NVIDIA. If a command
has changed since, the official page linked in that lab wins.

Model names and package versions are pinned and dated "checked November
2026". Newer ones will usually work too.

---

[Course home](../README.md) ·
[Start with Lab 00: Setup →](00-setup/README.md)
