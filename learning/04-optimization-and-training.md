# 4. Optimization and training

## Why this matters

Backpropagation hands you a gradient. It does not tell you what to do
with it. The optimizer, the learning rate schedule and a handful of
stabilizing tricks decide whether a model trains smoothly, trains
slowly, or blows up. Most "my model will not train" problems are solved
here, not in the architecture.

## The idea in plain words

Picture walking down a mountain in thick fog. You cannot see the valley.
You can only feel the slope under your feet. So you take a step
downhill, feel again, and repeat.

The gradient is the slope. The **learning rate** is the size of your
step. Everything else in this module is a refinement of that walk:
using memory of past slopes, adapting the step for each weight, and
changing step size over time.

## Gradient descent

The basic update for every weight:

```text
w = w - learning_rate * gradient
```

The learning rate is the most important setting in training.

- **Too small:** progress is painfully slow, and the model may settle
  somewhere mediocre.
- **Too large:** each step overshoots. The loss bounces around or
  diverges to `NaN`.
- **About right:** the loss falls quickly, then steadily.

When a run misbehaves, the learning rate is the first thing to change.

## Mini-batches: a noisy but cheap slope

Computing the exact gradient means running the whole dataset. Instead we
take a small random **batch**, compute the gradient on that, and step.
This is stochastic gradient descent (SGD).

The batch gradient is a noisy estimate of the true one. That turns out
to be fine, and even helpful: the noise shakes the model out of narrow,
fragile solutions and toward broad ones that generalize better.

**Batch size trade-offs:**

- Larger batches give steadier gradients and use hardware efficiently,
  but need more memory.
- Smaller batches are noisier and fit in less memory.
- If you make the batch bigger, you can usually raise the learning rate
  roughly in proportion.

**Gradient accumulation** fakes a large batch on a small GPU: run
several small batches, let their gradients add up, then step once.

## Momentum: remember where you were going

Plain SGD zig-zags in narrow valleys, bouncing between the walls while
crawling along the floor. Momentum fixes this by stepping along a
running average of recent gradients instead of only the latest one.

```text
velocity = 0.9 * velocity + gradient
w        = w - learning_rate * velocity
```

The side-to-side components cancel out. The consistent downhill
component builds up. Think of a heavy ball rolling rather than a
hiker stepping.

## Adam: a separate step size for every weight

Different weights see very different gradients. The embedding for a
rare token gets a signal once in a while. A weight in the final layer
gets a strong signal on every step. One shared learning rate cannot suit
both.

Adam tracks two running averages per weight:

```text
m = average of recent gradients
    (the direction)
v = average of recent gradients^2
    (their typical size)

w = w - lr * m / (sqrt(v) + tiny)
```

Dividing by `sqrt(v)` cancels the scale of the gradient. What remains,
`m / sqrt(v)`, is close to +1 or -1 when recent gradients agree, and
close to 0 when they keep flipping sign.

So with Adam, **each weight moves by about one learning rate per step
when it is confident, and barely moves when it is not.** The raw
magnitude of the gradient stops mattering. That robustness is why Adam
works out of the box on problems where SGD needs careful tuning.

## AdamW and weight decay

**Weight decay** shrinks every weight slightly toward zero on each
step. It is a form of regularization: weights have to keep earning
their size, which discourages the model from leaning on a few very
large ones.

**AdamW** is Adam with weight decay applied as its own separate step
rather than mixed into the gradient. That small change makes it behave
properly, and AdamW is the standard optimizer for transformers.

## Learning rate schedules

The best step size is not constant over a run.

**Warmup.** For the first few hundred or thousand steps, ramp the
learning rate up from zero. Early gradients are large and unreliable,
and Adam's running averages have not seen enough data to be trusted.
Big steps at this point can damage the model in ways it never recovers
from.

**Decay.** After warmup, lower the learning rate gradually. Big steps
find the right region. Small steps settle into it.

The usual shape is linear warmup followed by a cosine curve down:

```python
import math

PEAK = 3e-4
WARMUP = 1_000
TOTAL = 100_000

def lr_at(step):
    if step < WARMUP:
        return PEAK * step / WARMUP
    span = TOTAL - WARMUP
    done = (step - WARMUP) / span
    wave = math.cos(math.pi * done)
    return PEAK * 0.5 * (1 + wave)
```

```text
lr
 |    .-''-.
 |   /      '-.
 |  /          '-.
 | /              '-._
 |/                   '--..__
 +-----------------------------> step
  warmup        cosine decay
```

## Reasonable starting points

These are places to begin, not rules. Always tune the learning rate.

| Setting | Typical start |
|---|---|
| Optimizer | AdamW |
| Peak LR, training from scratch | 1e-4 to 1e-3 |
| Peak LR, full fine-tuning | 1e-5 to 5e-5 |
| Peak LR, LoRA fine-tuning | 1e-4 to 3e-4 |
| Weight decay | 0.01 to 0.1 |
| Warmup | 1% to 5% of steps |
| Gradient clip (norm) | 1.0 |

Fine-tuning uses a much smaller learning rate than training from
scratch. The model already knows a great deal, and big steps would
overwrite it.

## Regularization: fighting memorization

A model **overfits** when it memorizes the training examples instead of
learning the pattern behind them. Training loss keeps falling while loss
on unseen data starts rising.

Tools against it:

- **More data.** The best fix, when you can get it.
- **Weight decay.** Described above.
- **Dropout.** During training, randomly zero a fraction of
  activations. The network cannot rely on any single unit, so it learns
  redundant, sturdier features. Dropout is switched off at inference.
- **Early stopping.** Keep the checkpoint with the best validation
  score, not the last one.
- **Data augmentation.** Create varied versions of each example.

Very large language models trained on enormous datasets use little or no
dropout. With that much data, memorizing is harder than learning.

## Normalization

Normalization layers rescale activations to a steady range so that each
layer sees inputs of a consistent size. This keeps gradients well
behaved (see [Module 3](03-backpropagation.md)) and allows larger
learning rates.

- **Batch norm** normalizes across the batch. Common in image models. It
  behaves differently in training and inference, which causes bugs.
- **Layer norm** normalizes across the features of each token
  independently. It does not depend on the batch at all, and it is what
  transformers use.

## Two stabilizers used everywhere

**Gradient clipping.** If the length of the full gradient exceeds a
limit (often 1.0), scale it down to that limit. One bad batch can no
longer wreck the model.

**Mixed precision.** Do most arithmetic in 16-bit floating point instead
of 32-bit. Memory roughly halves and speed improves on modern GPUs,
with almost no loss in quality. The `bfloat16` format is preferred
because it handles a wide range of magnitudes safely.

## Reading a loss curve

Plot training loss and validation loss together. The shapes are a
diagnostic language.

| What you see | Likely cause |
|---|---|
| Both fall, then flatten together | Healthy. Model or data is the limit |
| Train falls, validation rises | Overfitting |
| Both stay high and flat | LR too low, or a bug |
| Loss spikes, then recovers | LR a little too high |
| Loss jumps to `NaN` | Exploding gradients |
| Validation far below train | Leakage, or heavy dropout |
| Loss starts far from `ln(V)` | Bad init or bad labels |

## The most useful debugging habit

Before a long run, **try to overfit a single small batch.** Train on the
same 8 examples for a few hundred steps with regularization off.

A correct model and training loop will drive the loss to nearly zero. If
it cannot memorize 8 examples, there is a bug in the data, the loss or
the model, and no amount of extra compute will fix it. This check takes
a minute and saves days.

## Common mistakes

- **Forgetting `model.eval()`.** Dropout stays on during evaluation and
  your scores are noisy and too low.
- **Tuning on the test set.** Every decision made by peeking at test
  results leaks it. Tune on validation and touch test once.
- **Trusting a single run.** Two runs with different random seeds can
  differ by more than the improvement you think you found.
- **Changing several things at once.** Then you cannot tell which change
  mattered.
- **Skipping warmup with Adam.** A common cause of early divergence in
  transformers.

## Check yourself

<details>
<summary>1. Your loss falls for a while, then suddenly becomes NaN. What
do you try first?</summary>

Lower the peak learning rate, make sure there is warmup, and add
gradient clipping at a norm of 1.0. Logging the gradient norm will show
it climbing before the failure.
</details>

<details>
<summary>2. Why does Adam cope with gradients of very different sizes
across weights?</summary>

It divides each weight's averaged gradient by the typical size of that
weight's gradient. The scale cancels, so every weight moves by roughly
one learning rate per step when its gradients are consistent.
</details>

<details>
<summary>3. Why is the learning rate for fine-tuning much lower than for
training from scratch?</summary>

A pretrained model's weights are already in a good place. Large steps
would move them far away and destroy what was learned. Small steps
adjust the model without overwriting it.
</details>

<details>
<summary>4. Training loss is 0.1 and validation loss is 1.5 and rising.
What is happening, and what are three responses?</summary>

Overfitting. Get more data or augment it, add regularization (weight
decay, dropout), stop earlier, or use a smaller model.
</details>

<details>
<summary>5. What does overfitting a single batch tell you?</summary>

That the pipeline is wired correctly: data, labels, model, loss and
optimizer can together drive the loss to zero. If they cannot, there is
a bug, and it is far cheaper to find it now than after a long run.
</details>

## Takeaways

- The learning rate matters more than any other setting.
- Momentum smooths the path. Adam adapts the step per weight.
- AdamW with warmup and cosine decay is the default for transformers.
- Overfitting shows up as training and validation loss parting ways.
- Clip gradients, use mixed precision, and watch both loss curves.
- Overfit one batch before you spend real compute.

---

[← Backpropagation](03-backpropagation.md) ·
[Course home](README.md) ·
[Next: Attention →](05-attention.md)
