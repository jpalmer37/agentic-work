# 2. Loss functions

## Why this matters

A model has no idea what you want. The loss function is the only place
you tell it. Everything the model learns is a side effect of making one
number smaller, so choosing that number is choosing what the model
becomes good at.

## The idea in plain words

A loss function takes the model's output and the right answer and
returns a single number: how wrong was that? Zero means perfect. Bigger
means worse. Training is the process of adjusting weights to push the
average of that number down.

The model optimizes exactly what you wrote, not what you meant. A loss
that is slightly off from your real goal produces a model that is
slightly off in the same way, very reliably.

## What makes a usable loss

- **One number.** You can only minimize one thing at a time.
- **Smooth.** A small change in the weights should cause a small change
  in the loss, so there is a slope to follow. This is why we do not
  train directly on accuracy: nudging a weight usually changes no
  predictions at all, so accuracy is flat almost everywhere.
- **Honest.** Lower loss should really mean better behaviour.

## Regression: predicting a number

**Mean squared error (MSE).** Square each error, then average.

```text
MSE = mean of (pred - target)^2
```

Squaring means big misses dominate. An error of 1 costs 1. An error of
10 costs 100. One wild outlier can outweigh a hundred good predictions.

**Mean absolute error (MAE).** Average the size of each error, without
squaring. An error of 10 costs 10. It is much calmer about outliers.

A useful way to remember the difference:

- Minimizing MSE pulls predictions toward the **mean** of the targets.
- Minimizing MAE pulls predictions toward the **median**.

**Huber loss** is the compromise: squared for small errors, absolute for
large ones. A good default when your data has occasional junk in it.

## Classification: from scores to probabilities

A classifier's last layer outputs one raw score per class. These scores
are called **logits**. They can be any number, positive or negative.

**Softmax** turns logits into probabilities that are positive and sum to
one: exponentiate each, then divide by the total.

```text
softmax(z)_i = exp(z_i) / total

total = sum of exp(z_j) over all j
```

Worked example with three classes:

```text
logits  = [2.0,  1.0,  0.1]
exp     = [7.39, 2.72, 1.11]
total   = 11.21
softmax = [0.659, 0.242, 0.099]
```

Two things to notice. Only the **differences** between logits matter:
add 100 to all of them and the output is identical. And the exponential
exaggerates gaps, so a modest lead in logits becomes a large lead in
probability.

## Cross-entropy

Look at the probability the model gave to the correct class. Take the
negative log of it. That is the loss.

```text
loss = -ln(p_correct)
```

If the correct class in the example above is the first one:

```text
loss = -ln(0.659) = 0.417
```

Why the log? Look at how it prices different levels of confidence:

| p for the right answer | Loss |
|---|---|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

Being right and confident costs almost nothing. Being wrong and
confident is punished severely, and the penalty has no ceiling. This
pushes the model to be accurate **and** to hold back when it is unsure.

## Where cross-entropy comes from

It is not an arbitrary choice. It falls out of one question: *which
weights make the training data most probable?*

1. The probability of the whole dataset is the product of the
   probability of each example.
2. Products of many small numbers underflow, so take the log. The log
   turns the product into a sum.
3. We minimize rather than maximize, so flip the sign.

What is left is the average of `-ln(p_correct)`. Minimizing
cross-entropy and "maximum likelihood" are the same thing, said two
ways. MSE comes from the same argument, if you assume errors follow a
bell curve.

## The gradient is beautifully simple

Combine softmax and cross-entropy, and the gradient of the loss with
respect to the logits is:

```text
gradient = predictions - truth
```

In the example, the true answer as a vector is `[1, 0, 0]`:

```text
[0.659, 0.242, 0.099] - [1, 0, 0]
  = [-0.341, 0.242, 0.099]
```

Read it as an instruction: raise the first logit, lower the others, each
in proportion to how wrong it was. When the model is right, the gradient
shrinks to zero by itself. This clean signal is a large part of why
classification networks train so well.

## Two sanity checks worth memorizing

**The starting loss.** An untrained model should guess evenly across
`V` classes, which gives a loss of `ln(V)`.

```text
10 classes      ->  ln(10)    =  2.30
50,257 tokens   ->  ln(50257) = 10.82
```

If your first logged loss is far from this, something is wrong with the
initialization or the data before training even begins.

**Perplexity.** For language models, people often report
`exp(loss)` instead of the loss.

```text
loss 3.0  ->  perplexity 20.1
loss 2.0  ->  perplexity  7.4
```

Read it as: "the model is as unsure as if it were choosing evenly among
this many tokens." Going from loss 3.0 to 2.0 sounds small, but it cuts
the effective number of choices from 20 to 7.

## The loss behind every LLM

A language model is a classifier whose classes are the tokens in its
vocabulary. At every position in the text it predicts the next token,
and the loss is cross-entropy against the token that really came next.

```text
loss = average over all positions of
       -ln(p of the actual next token)
```

That is the entire training objective. Grammar, facts, reasoning and
style all emerge because they help predict the next token.

## Other losses you will meet

**Binary cross-entropy.** For yes/no questions, or when several labels
can be true at once. Each output goes through a sigmoid instead of a
shared softmax, and each is scored independently.

**KL divergence.** Measures how far one probability distribution is from
another. Cross-entropy against a full distribution, rather than a single
right answer, is KL plus a constant. It is the loss used in
**distillation**, where a small model learns to copy a large model's
output probabilities.

**Contrastive loss.** Used to train embedding models. Take a batch of
matching pairs, such as a question and its answer. Compute the dot
product of every question with every answer. Then apply cross-entropy,
where the "correct class" for each question is its own answer.

```text
scores = questions @ answers.T
         shape (B, B)
target = the diagonal
         (pair i matches pair i)
loss   = cross_entropy(scores, target)
```

It is the same loss as before, built on the dot product from
[Module 1](01-vectors-and-dot-products.md). Matching pairs get pulled
together, and everything else in the batch gets pushed apart.

**Label smoothing.** Instead of a target of exactly 1 for the correct
class, use 0.9 and spread the rest. It stops the model from chasing
infinite confidence and often improves how well its probabilities match
reality.

## In code

```python
import numpy as np

def softmax(z):
    z = z - z.max()   # for stability
    e = np.exp(z)
    return e / e.sum()

logits = np.array([2.0, 1.0, 0.1])
target = 0

p = softmax(logits)
loss = -np.log(p[target])
grad = p.copy()
grad[target] -= 1.0

print(p.round(3))
print(loss.round(3))
print(grad.round(3))
```

```text
[0.659 0.242 0.099]
0.417
[-0.341  0.242  0.099]
```

Subtracting the maximum first changes nothing mathematically (only
differences matter) but stops `exp` from overflowing.

In PyTorch, the built-in loss expects raw logits:

```python
import torch.nn.functional as F

loss = F.cross_entropy(logits, targets)
```

## Common mistakes

- **Double softmax.** `F.cross_entropy` applies softmax internally.
  Applying your own first still trains, just badly, with no error.
- **Counting padding.** In a batch of uneven-length sequences, padded
  positions must be masked out of the loss, or the model is rewarded for
  predicting padding.
- **Confusing loss with the metric.** Loss is what training minimizes.
  The metric is what you care about. Watch both. They can disagree.
- **Ignoring class imbalance.** If 99% of examples are one class, low
  loss can mean the model has learned to always say that class.
- **Computing `log(softmax(x))` yourself.** It is numerically fragile.
  Use the library's combined function.

## Check yourself

<details>
<summary>1. A 1,000-class classifier logs a loss of 6.9 on its first
batch. Is that a problem?</summary>

No. `ln(1000)` is 6.91, which is what an evenly-guessing model should
score. A first loss of 20 or 0.5 would be the warning sign.
</details>

<details>
<summary>2. Why not train directly on accuracy?</summary>

Accuracy only changes when a prediction flips from wrong to right. For
almost every small change in a weight, no prediction flips, so the slope
is zero and there is nothing for gradient descent to follow.
Cross-entropy changes smoothly with every weight.
</details>

<details>
<summary>3. Your regression data has a few wildly wrong labels. Which
loss do you avoid, and what do you use?</summary>

Avoid plain MSE: squaring lets a handful of outliers dominate the
average. Use MAE or Huber loss, and also go and look at those labels.
</details>

<details>
<summary>4. A language model's loss falls from 2.5 to 2.2. What happens
to perplexity?</summary>

It goes from `exp(2.5) = 12.2` to `exp(2.2) = 9.0`. The model has gone
from being as unsure as a choice among 12 tokens to a choice among 9.
</details>

<details>
<summary>5. How is training an embedding model related to training a
classifier?</summary>

It is the same cross-entropy loss. The "logits" are dot products between
each item and every candidate in the batch, and the "correct class" is
the item's true partner.
</details>

## Takeaways

- The loss is the only way a model learns what you want.
- MSE chases the mean and hates outliers. MAE chases the median.
- Softmax turns scores into probabilities. Only differences matter.
- Cross-entropy is `-ln(p_correct)`: confident mistakes cost the most.
- Softmax plus cross-entropy gives the gradient `p - y`.
- An LLM is trained on next-token cross-entropy and nothing else.

---

[← Vectors and dot products](01-vectors-and-dot-products.md) ·
[Course home](README.md) ·
[Next: Backpropagation →](03-backpropagation.md)
