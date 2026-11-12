# Lab 05: LoRA fine-tune, and prove it helped

## Goal

Teach a small open model a narrow task with LoRA, then answer the only
question that matters: did it really get better, by more than the noise,
and at what cost?

## What you will see

- A base model following an extraction instruction, imperfectly.
- LoRA training under 2% of the model's weights, in minutes.
- An adapter of a few tens of megabytes, beside a model of about a
  gigabyte.
- A results table with error bars, and a case-by-case count of what
  the fine-tune fixed and what it broke.
- What 4-bit loading does to the fine-tuned model's accuracy.

## Time and hardware

One to two hours, including a download of about 1 GB. Needs the NVIDIA
GPU. The 4-bit steps need it. The rest runs on a CPU, very slowly.

## Read first

- [Module 4: Optimization](../../04-optimization-and-training.md), for
  learning rates and warmup.
- [Module 9: Adapting models](../../09-adapting-models.md), for when
  to fine-tune and how LoRA works.
- [Module 11: Evaluation](../../11-evaluation.md), for error bars and
  case-by-case comparison.

## The task

Customers write messages in every style imaginable. You need clean
JSON out:

```text
Customer Haddad, Felix, based in
Quebec City. Total: 131.73 CAD.
Ref 30379. felix.haddad@...
```

```json
{"name": "Felix Haddad",
 "email": "felix.haddad@...",
 "city": "Quebec City",
 "total": 131.73}
```

The traps: surname-first names, capitals, currency symbols, thousands
separators, and order and phone numbers that look like totals.
`make_data.py` generates 500 training and 100 test messages.

**The model** is `Qwen/Qwen2.5-0.5B-Instruct`: 0.5 billion parameters,
Apache 2.0 licence. Small enough that a fine-tune is quick, and small
enough that it needs one. Checked November 2026.

## Steps

### 1. Make the data

```bash
cd ~/agent-skills/learning/labs
source .venv/bin/activate
cd 05-lora-fine-tune
python make_data.py
```

Read a few lines of `data/train.jsonl`. Knowing your data is step one
of every fine-tune.

### 2. Measure the base model

```bash
python lab.py baseline
```

The first run downloads the model. It then answers all 100 test
messages with the instruction alone and prints:

- the share with **all four fields right**,
- the score for each field,
- how often the output was valid JSON,
- three failures, with what was wanted and what came back.

Read the failures. Which trap caught it?

### 3. Fine-tune

```bash
python lab.py train
```

```text
trainable params: ... || all params:
  ... || trainable%: 1.7...
step    0  loss 2.1..
step   10  loss 0.9..
...
Saved the adapter to out/adapter
```

The defaults are 150 steps of 8 examples, a peak learning rate of
`2e-4` with warmup and cosine decay, and LoRA rank 16 on every linear
layer. These are the starting points from
[Module 4](../../04-optimization-and-training.md).

### 4. Measure the fine-tuned model

```bash
python lab.py evaluate
```

Same 100 messages, same scoring, now with the adapter loaded.

### 5. Measure it in 4-bit

```bash
python lab.py evaluate --4bit
```

The base weights are loaded in 4-bit, with the same adapter on top.

### 6. Read the report

```bash
python lab.py report
```

```text
run        exact  +/-  JSON
base         ..%  ..%   ..%
fine-tuned   ..%  ..%   ..%
fine-tuned,
  4-bit      ..%  ..%   ..%

fine-tuned vs base:
  fixed .., broke ..
```

### 7. Experiments (optional)

Each changes one thing. Re-run `evaluate` and `report` after each.

**Less data.** How few steps are enough?

```bash
python lab.py train --steps 20
```

**Lower rank.** How small can the adapter be?

```bash
python lab.py train --rank 2
```

**QLoRA.** Train on a 4-bit base, the technique that lets far larger
models fit on one GPU:

```bash
python lab.py train --4bit
python lab.py evaluate --4bit
```

**A bigger model, prompt only.** Is a fine-tuned small model better than
a larger one used as-is?

```bash
python lab.py baseline \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

An adapter only fits the model it was trained on. Always evaluate with
the same `--model` you trained with.

## What to notice

**The error bars.** With 100 test messages, a score near 50% carries a
95% range of about plus or minus 10 points. A score near 95% carries
about plus or minus 4. A gap smaller than that is not evidence.

**Fixed versus broke.** Two overall percentages hide the detail. The
case-by-case line shows how many messages the fine-tune got right that
the base got wrong, and the reverse. A fine-tune that fixes 40 and
breaks 0 is convincing even on a small test set.

**Where the gain comes from.** Compare the per-field scores. Formatting
rules such as "first name then last name" and "a number, no currency
symbol" are exactly what fine-tuning teaches well and what instructions
alone convey poorly. This is "fine-tuning changes how the model
behaves" from [Module 9](../../09-adapting-models.md).

**The adapter is tiny.** Compare the size of `out/adapter` with the
model in `~/.cache/huggingface`. One base model can carry many small
adapters, one per task.

**The loss falls fast, then flattens.** The model already knew how to
read and write JSON. It only had to learn your conventions. A few
hundred examples are plenty for that.

**4-bit costs a little.** Usually a point or two on a task like this,
sometimes nothing. Measure on your own task rather than assuming, as
[Module 8](../../08-inference-and-generation.md) advises.

**Training data quality is everything.** The model learns exactly what
`make_data.py` produces. Change a rule in the generator, for example
surnames first, and the fine-tuned model follows it faithfully. It
learns your mistakes as faithfully as your intentions.

## Spin down and clean up

Nothing runs after each command finishes. To free the disk:

```bash
rm -rf out data
rm -rf ~/.cache/huggingface/hub/\
models--Qwen--Qwen2.5-0.5B-Instruct
rm -rf ~/.cache/huggingface/hub/\
models--Qwen--Qwen2.5-1.5B-Instruct
```

When you have finished every lab, remove the shared Python environment
too:

```bash
cd ~/agent-skills/learning/labs
rm -rf .venv
```

## Troubleshooting

**`CUDA out of memory`.** Another program holds the GPU: often vLLM from
Lab 03 or Ollama from Lab 02. Check `nvidia-smi`, then
`docker compose down` in Lab 03's folder or
`sudo systemctl stop ollama`. Or use `--batch 4`.

**`4-bit loading needs an NVIDIA GPU`.** PyTorch cannot see the GPU.
Re-run the Lab 00 check.

**bitsandbytes errors.** Make sure you installed the pinned versions
from `00-setup/requirements.txt` into `.venv`.

**Loss does not fall.** Check that `make_data.py` ran and the learning
rate is `2e-4`. A rate of `2e-6` barely moves. A rate of `2e-2` usually
diverges.

**The fine-tuned model is worse.** It happens. Look at what it broke in
the report, check the training data for the same mistakes, and try
fewer steps. This is why you measure.

## How this lab was checked

Every command except the 4-bit ones ran on a CPU before publishing,
using a tiny stand-in model built locally, because the real model could
not be downloaded on the test machine. That proves the code path. It
says nothing about the real model's scores, which you will be the first
to see.

---

[← Lab 04: RAG over your docs](../04-rag-over-your-docs/README.md) ·
[Labs home](../README.md)
