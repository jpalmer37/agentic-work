# 5. Attention

## Why this matters

Attention is the mechanism that lets each token gather information from
other tokens. It is the heart of the transformer and the reason language
models can use context. It is also where most of the cost of long
prompts comes from. Understanding it properly explains both what these
models can do and what they charge you for.

## The idea in plain words

The meaning of a word depends on the words around it. In "the bank of
the river" and "the bank raised rates", the token "bank" needs to end up
with different internal representations.

Attention is how that happens. Each token **asks a question**, every
token **advertises what it has**, and the asker pulls in a blend of
information weighted toward the tokens whose advertisement best matches
the question.

"Best matches" is measured with a dot product.

## Attention as a soft lookup

You know how a dictionary lookup works:

```python
table = {"colour": "red",
         "size": "large"}
table["colour"]   # exact match: "red"
```

The key must match exactly, and you get exactly one value back.

Attention is the same idea made soft:

- The **query** is compared with **every key**, giving a score each.
- Scores become weights that sum to 1.
- The result is a **weighted average of all the values**.

Instead of "the one entry that matches", you get "mostly the entries
that match well, plus a little of the rest". Because it is soft, it is
differentiable, so the model can learn what to look up.

## Query, key, value

Every token's vector is passed through three learned matrices to
produce three new vectors:

| Vector | Role |
|---|---|
| Query | What am I looking for? |
| Key | What do I contain? |
| Value | What will I hand over if chosen? |

Keys and values are separate so that *how a token is found* can differ
from *what it contributes once found*.

## The four steps

```text
1. scores  = Q @ K.T
   every query dotted with every key

2. scores  = scores / sqrt(d)
   keep them in a sane range

3. weights = softmax(scores)
   each row sums to 1

4. output  = weights @ V
   blend the values
```

Or as one formula, the most famous in modern AI:

```text
Attention(Q, K, V)
  = softmax(Q K^T / sqrt(d)) V
```

Here `d` is the size of the query and key vectors.

## Worked example

One query and three tokens to look at, with `d = 2`:

```text
query  q = [1, 0]

keys       values
[1, 0]     [10,  0]
[0, 1]     [ 0, 10]
[1, 1]     [ 5,  5]
```

**Step 1: dot the query with each key.**

```text
scores = [1, 0, 1]
```

The query matches the first and third keys, and not the second.

**Step 2: divide by sqrt(2) = 1.414.**

```text
scaled = [0.707, 0, 0.707]
```

**Step 3: softmax.**

```text
weights = [0.401, 0.198, 0.401]
```

**Step 4: weighted average of the values.**

```text
output = 0.401 * [10,  0]
       + 0.198 * [ 0, 10]
       + 0.401 * [ 5,  5]
       = [6.02, 3.98]
```

The output leans toward the values whose keys matched the query. That
is all attention does.

## Why divide by the square root?

This is the question people most often get wrong, and you already have
the answer from [Module 1](01-vectors-and-dot-products.md).

**The problem.** A dot product is a sum of `d` terms. More terms, bigger
sums. For vectors with entries of spread around 1, the typical size of
the dot product is `sqrt(d)`. With `d = 1024`, scores are routinely
around 32, and the gaps between them are just as large.

Softmax exponentiates its inputs. Feed it gaps of size 30 and one
weight becomes essentially 1 while every other becomes essentially 0.

That causes two failures:

1. **Attention goes hard.** Each token can only look at one other token,
   instead of blending several.
2. **Gradients vanish.** A saturated softmax is flat: nudging the scores
   changes nothing. The model starts training with attention frozen in
   random, useless patterns.

**The fix.** Divide scores by `sqrt(d)`. The typical size goes back to
about 1, whatever the dimension, and softmax starts in its responsive
range.

**Why the square root and not `d`?** Each of the `d` terms contributes
some variance, and variances add, so the total variance is about `d`.
The typical size is the square root of the variance. Dividing by
`sqrt(d)` restores a spread of exactly 1. Dividing by `d` would squash
scores too far and make attention almost uniform.

You can measure the effect. For each dimension, this takes a random
query and 8 random keys and reports the largest attention weight,
averaged over many trials:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

rng = np.random.default_rng(0)
for d in (16, 64, 256, 1024):
    raw, scaled = [], []
    for _ in range(2000):
        q = rng.standard_normal(d)
        K = rng.standard_normal((8, d))
        s = K @ q
        t = s / np.sqrt(d)
        raw.append(softmax(s).max())
        scaled.append(softmax(t).max())
    print(d, round(np.mean(raw), 2),
          round(np.mean(scaled), 2))
```

```text
16   0.75 0.36
64   0.88 0.36
256  0.94 0.36
1024 0.97 0.36
```

Without scaling (middle column), one key takes nearly all the weight as
the dimension grows. With scaling (right column), the top weight sits at
0.36 every time: focused, but still sharing.

Think of it as setting the default temperature so that training starts
somewhere sensible. The model can still learn sharper or softer
attention later by growing or shrinking its weights.

## Shapes

For a sequence of `T` tokens with vectors of size `d`:

```text
Q, K, V    (T, d)
scores     (T, T)
weights    (T, T)
output     (T, d)
```

Row `i` of the weights says how much token `i` attends to each token.
Every token produces a query, so every token gets its own row of
weights and its own blended output. All of it happens in one matrix
multiplication, in parallel.

## Masks: controlling who can see whom

**The causal mask.** A language model predicts the next token, so a
token must not look at tokens that come after it. We enforce this by
setting those scores to negative infinity before the softmax, which
turns their weights into exactly zero.

```text
        can attend to
        t1  t2  t3  t4
  t1  [  x   .   .   . ]
  t2  [  x   x   .   . ]
  t3  [  x   x   x   . ]
  t4  [  x   x   x   x ]
```

This buys something valuable. Because position 3 can only see positions
1 to 3, its output is a fair prediction of token 4. So a single forward
pass over a sequence of `T` tokens yields `T` separate training
examples, all computed at once.

**The padding mask.** Sequences in a batch are padded to equal length.
Padding positions are masked the same way so nobody attends to them.

## Multi-head attention

One set of attention weights captures one kind of relationship. But a
token might need several at once: its grammatical subject, the thing a
pronoun refers to, and the previous word.

So the vector is split into several **heads**. With `d = 768` and 12
heads, each head works in 64 dimensions with its own query, key and
value matrices. The heads run in parallel, their outputs are joined back
together, and one more matrix mixes them.

```text
x --> split into heads
        head 1 --+
        head 2 --+--> join --> mix
        ...    --+
        head h --+
```

The total cost is about the same as one big head. You get many
specialists for the price of one generalist.

Note that the `d` in `sqrt(d)` is the **per-head** size, 64 in this
example, not 768.

## Self-attention and cross-attention

**Self-attention:** queries, keys and values all come from the same
sequence. Tokens look at each other. This is what a language model uses.

**Cross-attention:** queries come from one sequence, keys and values
from another. This is how a translation model's output looks at the
source sentence, or how text tokens look at image features.

The arithmetic is identical. Only the source of `K` and `V` changes.

## Attention has no sense of order

Look at the four steps again. Nothing in them uses a token's position.
Shuffle the input and each token's output is unchanged, merely
shuffled the same way.

To attention, the input is a **set**, not a sequence. "Dog bites man"
and "man bites dog" would look the same. Position has to be injected
separately, which is covered in [Module 6](06-transformers.md).

## What it costs

The score matrix is `T` by `T`. So compute and memory for attention grow
with the **square** of sequence length.

```text
 1,000 tokens ->   1,000,000 scores
10,000 tokens -> 100,000,000 scores
```

That is per head, in every layer. Ten times the context costs a hundred times the attention work. This is
the fundamental reason long contexts are expensive and why context
windows have limits.

You will often hear about **flash attention**. It is not a different
kind of attention and not an approximation. It computes the same result
in small blocks so that the full `T` by `T` matrix never has to sit in
memory at once.

## What attention does not do

Attention only **moves** information between positions. Its output is a
weighted average, which is a mild operation. The heavier per-token
processing happens in the feed-forward layer that follows it.

A useful division of labour to keep in mind:

- **Attention:** tokens communicate.
- **Feed-forward layer:** each token thinks about what it gathered.

## In code

The full mechanism in a dozen lines of NumPy:

```python
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    Kt = K.swapaxes(-1, -2)
    scores = Q @ Kt / np.sqrt(d)
    if mask is not None:
        # hide the masked positions
        scores[..., ~mask] = -1e9
    weights = softmax(scores)
    return weights @ V, weights

rng = np.random.default_rng(0)
# 4 tokens, d = 8
X = rng.standard_normal((4, 8))
ones = np.ones((4, 4), dtype=bool)
causal = np.tril(ones)

out, w = attention(X, X, X, causal)
print(out.shape)    # (4, 8)
print(w.round(2))
```

```text
[[1.   0.   0.   0.  ]
 [0.02 0.98 0.   0.  ]
 [0.17 0.22 0.62 0.  ]
 [0.08 0.3  0.11 0.51]]
```

Each row sums to 1, and the upper triangle is zero: no token sees the
future.

Multi-head attention reuses the same function. Splitting into heads is
only a reshape. Here `W` holds the four matrices: query, key, value and
the output mix.

```python
def multi_head(X, W, heads, mask=None):
    Wq, Wk, Wv, Wo = W
    T, d = X.shape

    def split(M):
        # (T, d) -> (heads, T, d/heads)
        M = M.reshape(T, heads, -1)
        return M.swapaxes(0, 1)

    Q = split(X @ Wq)
    K = split(X @ Wk)
    V = split(X @ Wv)
    out, _ = attention(Q, K, V, mask)
    out = out.swapaxes(0, 1)
    return out.reshape(T, d) @ Wo
```

## Common mistakes

- **Forgetting the scale, or using the wrong `d`.** It is the per-head
  dimension of the queries and keys.
- **Masking after the softmax.** Zeroing weights afterwards leaves rows
  that no longer sum to 1. Mask the scores before the softmax.
- **Treating attention weights as an explanation.** They show where
  information was read from in one layer, not why the model produced
  its answer. Use them as a hint, not as proof.
- **Assuming long context is free.** Cost grows with the square of
  length, and quality often drops when the relevant fact is buried in
  the middle of a long prompt.

## Check yourself

<details>
<summary>1. In one sentence each, what are the query, key and
value?</summary>

The query is what a token is looking for. The key is what a token
advertises about itself. The value is what a token contributes to anyone
who attends to it.
</details>

<details>
<summary>2. What goes wrong if you remove the division by
sqrt(d)?</summary>

Scores grow with dimension, softmax saturates so that nearly all weight
lands on one token, and its gradient becomes almost zero. Attention
starts out frozen and learns very slowly.
</details>

<details>
<summary>3. Why sqrt(d) rather than d?</summary>

A dot product of `d` terms has variance about `d`, so its typical size
is `sqrt(d)`. Dividing by that restores a spread of 1. Dividing by `d`
would shrink scores too far and flatten attention toward uniform.
</details>

<details>
<summary>4. What does the causal mask make possible during
training?</summary>

Every position predicts its next token using only earlier tokens, so one
forward pass over `T` tokens gives `T` valid training examples at once,
with no cheating.
</details>

<details>
<summary>5. You go from a 4,000-token prompt to a 16,000-token prompt.
How does attention cost change?</summary>

The length is 4 times larger, so the number of scores is 16 times
larger.
</details>

<details>
<summary>6. Why use 12 small heads instead of one large one?</summary>

One head gives one pattern of weights per token. Several heads let a
token attend to different things for different reasons at the same time,
for about the same total cost.
</details>

## Takeaways

- Attention is a soft dictionary lookup, scored with dot products.
- `softmax(Q K^T / sqrt(d)) V` is four simple steps.
- The square root undoes the growth of dot products with dimension, so
  softmax starts responsive rather than saturated.
- The causal mask hides the future and makes training parallel.
- Multiple heads look for different relationships at once.
- Attention ignores order and costs the square of sequence length.

---

[← Optimization and training](04-optimization-and-training.md) ·
[Course home](README.md) ·
[Next: Transformers →](06-transformers.md)
