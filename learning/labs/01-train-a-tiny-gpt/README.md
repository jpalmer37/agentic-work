# Lab 01: Train a tiny GPT

## Goal

Train a small language model from random numbers to recognisable
Shakespeare, using exactly the pieces described in the course, and
watch each piece behave as the theory says it should.

## What you will see

- A first loss almost exactly equal to `ln(vocabulary size)`.
- The loss falling steeply, then flattening.
- Generated text that goes from noise to words, to something that
  looks like a play.
- Temperature changing the text from repetitive to chaotic.

## Time and hardware

- **GPU:** about 10.7 million parameters, roughly 5 to 10 minutes on an
  RTX 4070.
- **CPU:** a smaller model of about 0.8 million parameters, roughly 10
  minutes on a laptop. Add `--cpu`.

## Read first

- [Module 2: Loss functions](../../02-loss-functions.md), for the
  starting loss and perplexity.
- [Module 5: Attention](../../05-attention.md), for the causal mask and
  the square root.
- [Module 6: Transformers](../../06-transformers.md), for the block and
  the residual stream.
- [Module 8: Inference](../../08-inference-and-generation.md), for
  temperature.

## Steps

### 1. Go to the lab

```bash
cd ~/agent-skills/learning/labs
source .venv/bin/activate
cd 01-train-a-tiny-gpt
```

### 2. Train

```bash
python train.py
```

The first run downloads the Complete Works of Shakespeare (about 5 MB)
from Project Gutenberg into `data/`.

The opening lines describe the model:

```text
device       cuda (bf16)
vocabulary   ... tokens
parameters   10.7...M
ln(vocab)    4.5...
```

Then a progress line every 250 steps:

```text
step     0  train 4.6..  val 4.6..
step   250  train 2.4..  val 2.4..
step   500  train 2.0..  val 2.0..
...
```

At the end it prints a sample and saves the model, the loss history and
a plot to `out/`.

### 3. Look at the loss curve

Open `out/loss.png`. From Windows Explorer, type `\\wsl$` in the address
bar to browse your Ubuntu files, or run:

```bash
explorer.exe out
```

### 4. Generate at different temperatures

```bash
python generate.py
```

This prints the same prompt continued at temperatures 0.5, 1.0 and 1.5.
Try your own:

```bash
python generate.py \
  --prompt "JULIET:" \
  --temps 0 0.8
```

### 5. Experiment: remove the square root

```bash
python train.py --no-scale \
  --out out-noscale
```

Compare its loss curve with the first run.

### 6. Experiment: remove position

```bash
python train.py --no-pos \
  --out out-nopos
python generate.py \
  --model out-nopos/model.pt
```

### 7. Train on your own text (optional)

Any plain-text file works: your notes, a codebase, song lyrics.

```bash
python train.py \
  --text ~/my-notes.txt \
  --out out-mine
```

## What to notice

**The first loss.** Before training, the model knows nothing, so it
spreads probability evenly across the vocabulary. Cross-entropy for an
even guess is `ln(vocabulary size)`. The script prints that value next
to the first loss so you can compare. If they did not match, something
would be wrong before training even began.

**The shape of the curve.** A steep drop as the model learns which
characters are common, then spaces and short words, then a long slow
slope as it learns spelling and structure. The warmup at the start
and the cosine decay at the end are from
[Module 4](../../04-optimization-and-training.md).

**Train and validation together.** The two curves should stay close.
If you train much longer, or on a short text, validation stops falling
while training keeps going down. That gap is overfitting.

**Perplexity.** A final loss of about 1.5 is `exp(1.5) = 4.5`. The
model is about as unsure as a choice among four or five characters,
down from a choice among about 100 at the start.

**Temperature.** At 0.5 the text is safe and often repeats itself. At
1.5 it invents words and loses its thread. Around 0.8 to 1.0 usually
reads best. Same model, same probabilities, different sampling.

**Without the square root.** With only 64 dimensions per head the
damage is modest, but the loss usually starts less smoothly and ends a
little higher. The effect grows with the head size, as the table in
[Module 5](../../05-attention.md) shows.

**Without position.** The model can still learn which characters
are common, but it cannot tell where it is in a word or line. The loss
stays clearly higher, and the text is a jumble of plausible fragments.
Attention on its own sees a set, not a sequence.

## Spin down and clean up

Nothing keeps running after the script ends. To free the disk space:

```bash
rm -rf data out out-*
```

## Troubleshooting

**Download fails.** Download the plain-text file of any Project
Gutenberg book in your browser and pass it with `--text`.

**`CUDA out of memory`.** Something else is using the GPU. Check with
`nvidia-smi`, close it, or run with `--cpu`.

**Very slow on the GPU.** Check that the first line says `cuda`. If it
says `cpu`, PyTorch cannot see the GPU. Re-run the Lab 00 check.

**Loss becomes `nan`.** Rare at these settings. Run again. If it
repeats, lower `lr` in the `CONFIGS` table at the top of `train.py`.

---

[← Lab 00: Setup](../00-setup/README.md) ·
[Labs home](../README.md) ·
[Next: Lab 02, Run a local LLM →](../02-run-a-local-llm/README.md)
