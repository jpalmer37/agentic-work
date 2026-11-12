# 3. Backpropagation

## Why this matters

Backpropagation is the algorithm that makes deep learning possible. It
is also the source of most practical limits you will hit: memory use in
training, unstable training in deep networks, and the design of nearly
every modern architecture. Residual connections and normalization layers
exist largely to keep backpropagation healthy.

## The idea in plain words

A neural network is one very long formula. To improve it, we need to
know, for **every** weight: if I nudge this up a little, does the loss
go up or down, and by how much? That sensitivity is the weight's
**gradient**.

The obvious approach is to nudge each weight and rerun the network. With
a billion weights, that is a billion runs for a single update.

Backpropagation gets every gradient in **one backward sweep** that costs
about the same as running the network forward once more. It does this by
working backward from the loss and reusing intermediate results, the
same way dynamic programming avoids recomputing subproblems.

## The chain rule, without calculus notation

Suppose changing A by 1 changes B by 3, and changing B by 1 changes C by
2. Then changing A by 1 changes C by 6.

That is the whole chain rule: **multiply sensitivities along a path.**
If A reaches C by more than one path, **add the paths together.**

## Networks are graphs

Any computation can be drawn as a graph of small steps. Here is a tiny
two-layer network with one ReLU:

```text
x --+
    (*)--> a --relu--> h --+
w1 -+                      (*)--> yhat
                      w2 --+

loss = (yhat - y)^2
```

The forward pass runs left to right and computes values. The backward
pass runs right to left and computes gradients.

## Worked example, by hand

Use these numbers:

```text
x = 2     w1 = 0.5     w2 = 3
target y = 5
```

**Forward.** Compute and remember every intermediate value.

```text
a    = w1 * x      = 1
h    = relu(a)     = 1
yhat = w2 * h      = 3
loss = (3 - 5)^2   = 4
```

**Backward.** Start at the loss and step back one node at a time. Write
`d(thing)` for "how much the loss changes per unit change in thing".

```text
d(yhat) = 2 * (yhat - y) = -4

d(w2) = d(yhat) * h   = -4 * 1 = -4
d(h)  = d(yhat) * w2  = -4 * 3 = -12

d(a)  = d(h) * 1      = -12
        (relu is open, since a > 0)

d(w1) = d(a) * x      = -12 * 2 = -24
d(x)  = d(a) * w1     = -12 * 0.5 = -6
```

Notice that `d(h)` was computed once and then reused for everything
upstream of it. That reuse is the entire trick.

**Update.** Both gradients are negative: increasing either weight
lowers the loss. Take a small step against the gradient, with a learning
rate of 0.01:

```text
w1 = 0.5 - 0.01 * (-24) = 0.74
w2 = 3   - 0.01 * (-4)  = 3.04
```

Run forward again: `yhat = 3.04 * 0.74 * 2 = 4.50`, and the loss drops
from 4 to 0.25.

## Each operation only needs a local rule

Look back at the example. No step needed to understand the whole
network. Each one took the gradient arriving from the right, applied a
small local rule, and passed the result left.

| Operation | What it does to the incoming gradient |
|---|---|
| Add | Copies it to both inputs unchanged |
| Multiply | Sends each input the gradient times the *other* input |
| ReLU | Passes it if the input was positive, else blocks it |
| Copy / reuse | Adds up the gradients from every place it was used |

For a matrix layer `out = W @ x`, the same multiply rule applies in
matrix form:

```text
d(W) = outer(d(out), x)
d(x) = W.T @ d(out)
```

This is why automatic differentiation works. A framework only needs to
know, for each primitive operation, how to run it forward and how to
pass a gradient backward. Chain enough of those and you can
differentiate anything.

## A complete autograd engine

The whole idea fits in under fifty lines. Each `Value` remembers the
values it was made from (`prev`) and a small function (`back`) that
applies its local rule.

```python
class Value:
    def __init__(self, data, prev=()):
        self.data = data
        self.grad = 0.0
        self.prev = prev
        self.back = lambda: None

    def __add__(self, o):
        out = Value(self.data + o.data,
                    (self, o))
        def back():
            self.grad += out.grad
            o.grad += out.grad
        out.back = back
        return out

    def __mul__(self, o):
        out = Value(self.data * o.data,
                    (self, o))
        def back():
            g = out.grad
            self.grad += o.data * g
            o.grad += self.data * g
        out.back = back
        return out

    def relu(self):
        x = max(0.0, self.data)
        out = Value(x, (self,))
        def back():
            if out.data > 0:
                self.grad += out.grad
        out.back = back
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v.prev:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v.back()
```

Run the worked example through it:

```python
x = Value(2.0)
w1 = Value(0.5)
w2 = Value(3.0)
neg_y = Value(-5.0)

h = (w1 * x).relu()
err = w2 * h + neg_y
loss = err * err
loss.backward()

print(loss.data, w1.grad, w2.grad)
# 4.0 -24.0 -4.0
```

The same numbers as the hand calculation. PyTorch does exactly this,
with tensors instead of single numbers and with every operation
implemented in fast compiled code.

Two details in that code are worth a second look:

- Gradients use `+=`, not `=`. A value used in two places receives
  gradient from both, and they must add.
- `backward` first sorts the graph so that every node is processed only
  after everything that depends on it. That ordering is what lets each
  gradient be computed once.

## What backpropagation costs

**Time:** roughly two to three times a forward pass. Cheap.

**Memory:** expensive. Every backward rule above needed a value from the
forward pass (`h`, `w2`, `x`, whether the ReLU was open). So the forward
pass must **keep every intermediate activation** until the backward pass
has used it.

This is the reason:

- Training needs several times more memory than inference.
- Batch size and sequence length are limited by GPU memory.
- Inference wraps code in `torch.no_grad()`, which tells the framework
  not to keep anything.
- "Gradient checkpointing" exists: it throws activations away and
  recomputes them during the backward pass, trading time for memory.

## Why deep networks are hard: vanishing and exploding

The gradient that reaches an early layer is a **product** of one factor
per layer it passed through. Products are unstable.

```text
0.9 ^ 50  = 0.005     (vanishes)
1.1 ^ 50  = 117       (explodes)
```

Factors only slightly below 1 starve the early layers of signal, so they
stop learning. Factors slightly above 1 produce enormous updates and the
loss turns into `NaN`.

The classic example is the sigmoid activation. Its slope is never more
than 0.25, so ten layers shrink the gradient by at least
`0.25^10`, about one millionth.

## How modern networks keep gradients healthy

Each of these is a direct answer to the product problem.

**ReLU-style activations.** Slope of exactly 1 when active, so no
shrinkage on the active path.

**Careful initialization.** Starting weights are scaled so that signals
keep roughly the same size from layer to layer, in both directions.

**Normalization layers.** Rescale activations to a steady range at every
layer, so factors cannot drift far from 1.

**Residual connections.** Instead of `out = f(x)`, compute
`out = x + f(x)`. Recall the rule for add: it copies the gradient to
both inputs unchanged. So there is always a direct path from the loss
back to every layer that involves no multiplication at all.

```text
without:
  x --> f --> f --> f --> loss
  gradient is multiplied at each f

with:
  x -+--> + -+--> + -+--> + --> loss
     |    ^  |    ^  |    ^
     +-f--+  +-f--+  +-f--+
  a clear highway along the top
```

This is the single most important structural idea in deep learning. It
is what allows networks with hundreds of layers, and every transformer
depends on it.

**Gradient clipping.** If the overall gradient is larger than some
limit, scale it down. A blunt but effective guard against explosions.

## In PyTorch

```python
# clear old gradients
optimizer.zero_grad()
loss = loss_fn(model(x), y)
# fill .grad on every weight
loss.backward()
# nudge the weights
optimizer.step()
```

Four statements, and now you know what each one does. `zero_grad` exists
because gradients accumulate with `+=`, exactly as in the small engine
above.

## Common mistakes

- **Forgetting `zero_grad`.** Gradients from previous steps pile up and
  training behaves erratically.
- **Evaluating without `no_grad`.** It works, but memory climbs because
  the graph is being stored for a backward pass that never comes.
- **Holding on to the loss tensor.** Appending `loss` to a list keeps
  its whole graph alive. Append `loss.item()` instead.
- **Dead ReLUs.** A unit whose input is always negative passes no
  gradient and can never recover. Too high a learning rate causes this.
- **Assuming a `NaN` loss is a data bug.** Often it is exploding
  gradients. Log the gradient norm and you will see it climb first.

## Check yourself

<details>
<summary>1. Why is one backward pass enough to get the gradient for
every weight?</summary>

Because gradients are shared. The sensitivity of the loss to a late
value is computed once, then reused by everything that feeds into it.
Working backward means each node's gradient is ready by the time the
nodes before it need it.
</details>

<details>
<summary>2. In `z = x * y` with x = 3 and y = 4, the gradient arriving
at z is 2. What gradients do x and y receive?</summary>

Multiply swaps: each input gets the incoming gradient times the other
input. `x` receives `2 * 4 = 8` and `y` receives `2 * 3 = 6`.
</details>

<details>
<summary>3. Why does training use so much more memory than
inference?</summary>

The backward pass needs the intermediate values from the forward pass,
so they must all be stored until it runs. Inference can discard each
layer's output as soon as the next layer has used it.
</details>

<details>
<summary>4. How does a residual connection help gradients?</summary>

Addition passes the gradient through unchanged. With `x + f(x)`, there
is always a path from the loss back to earlier layers that skips every
multiplication, so the signal cannot vanish along it.
</details>

<details>
<summary>5. Your loss becomes NaN after a few hundred steps. Name two
things to try.</summary>

Lower the learning rate (or add warmup), and add gradient clipping.
Logging the gradient norm will confirm whether it was exploding.
</details>

## Takeaways

- A gradient says how the loss responds to a nudge in one weight.
- Backpropagation gets all of them in one backward sweep by reuse.
- Each operation needs only a small local rule. Add copies, multiply
  swaps, ReLU gates.
- The cost is memory: forward activations must be kept.
- Gradients are products across layers, so they vanish or explode.
- Residual connections give gradients a multiplication-free highway.

---

[← Loss functions](02-loss-functions.md) ·
[Course home](README.md) ·
[Next: Optimization and training →](04-optimization-and-training.md)
