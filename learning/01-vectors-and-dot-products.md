# 1. Vectors and dot products

## Why this matters

Strip a neural network down and you find one operation repeated billions
of times: the dot product. A neuron is a dot product. Attention is a
table of dot products. Semantic search is a dot product. If this one
idea is solid, the rest of the course is mostly bookkeeping.

## The idea in plain words

A vector is a list of numbers. In machine learning we treat that list as
a **point in space**, and we arrange things so that **direction carries
meaning**: similar things point in similar directions.

The dot product takes two vectors and returns one number that says how
much they agree. Large and positive: they point the same way. Near zero:
they are unrelated. Negative: they point in opposite directions.

## Two views of a vector

**As data.** `[0.2, -1.3, 0.7]` is three numbers in a row. This is how
the computer sees it.

**As an arrow.** The same three numbers are an arrow from the origin to
a point in 3D space. It has a length and a direction. This is how you
should picture it.

Real models use hundreds or thousands of dimensions. You cannot picture
that, and you do not need to. Picture 2D or 3D and trust that the
arithmetic carries over.

## The dot product

Multiply matching entries and add them up:

```text
a · b = a1*b1 + a2*b2 + ... + ad*bd
```

Worked example:

```text
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1*4 + 2*5 + 3*6
      = 4 + 10 + 18
      = 32
```

There is a second formula that gives the same number and explains what
it means:

```text
a · b = |a| * |b| * cos(angle)
```

`|a|` is the length of `a`, and the angle is the one between the two
vectors. So the dot product is large when both vectors are long **and**
they point the same way.

## Three ways to read a dot product

All three are the same arithmetic. Which one helps depends on context.

**1. Similarity.** How aligned are these two things? This is the reading
used in search and in attention.

**2. Projection.** How far does `a` reach along the direction of `b`?
This is the reading used when asking "how much of feature `b` is present
in `a`?"

**3. Weighted sum.** Treat `w` as weights and `x` as inputs:
`w · x` adds up the inputs, each scaled by how much it matters. This is
exactly what a neuron computes before its activation function.

> A neuron is a detector. Its weight vector is the pattern it looks for.
> The dot product measures how strongly the input matches that pattern.

## Cosine similarity

The dot product mixes two things: direction and length. Often you only
care about direction. Divide the lengths out:

```text
cosine(a, b) = (a · b) / (|a| * |b|)
```

The result is always between -1 and 1.

Continuing the example:

```text
|a| = sqrt(1 + 4 + 9)    = 3.742
|b| = sqrt(16 + 25 + 36) = 8.775

cosine = 32 / (3.742 * 8.775) = 0.975
```

Nearly 1, so these point in almost the same direction.

**Which to use?**

- If your vectors are normalized to length 1, dot product and cosine are
  the same number. Most embedding models are used this way.
- If length carries information (for example, confidence or frequency),
  the plain dot product keeps it and cosine throws it away.

## Matrices: many dot products at once

A matrix is a stack of vectors. Multiplying a matrix by a vector takes
the dot product of each row with that vector:

```text
        [ row1 · x ]
W @ x = [ row2 · x ]
        [ row3 · x ]
```

So a layer with weight matrix `W` asks one question per row: "how much
does the input match my pattern?" The output is the list of answers.

That is all a linear layer is: a bank of pattern detectors running in
parallel.

**The shape rule.** `(m, n) @ (n, k)` gives `(m, k)`. The inner numbers
must match. Most bugs in ML code are shape bugs, so it pays to write the
shape in a comment beside every tensor.

## Embeddings: meaning as direction

A model cannot do arithmetic on the word "cat". So each token gets a
row in a big table, and that row is its **embedding**: a learned vector.

```text
token id
  -> row of the table
  -> vector of size d
```

The values start random. Training moves them so that tokens used in
similar ways end up pointing in similar directions. Nobody tells the
model that "cat" and "kitten" are related. It falls out of the training
objective, because giving them similar vectors makes predictions easier.

The same idea extends beyond words. Sentences, images, users and
products can all be embedded. Once they are vectors, "find similar
items" becomes "find the largest dot products", which is fast.

## High dimensions behave strangely

Two facts about high-dimensional space matter later in the course.

**Random vectors are almost perpendicular.** In 2D, two random arrows
often point in similar directions. In 1,000 dimensions they almost never
do. There is a huge amount of room, so a model can store many more
distinct "directions of meaning" than it has dimensions, as long as they
are only nearly perpendicular.

**Dot products grow with dimension.** If each entry is a random number
with spread around 1, the dot product of two such vectors has a typical
size of about `sqrt(d)`. More dimensions, more terms in the sum, bigger
totals.

You can see both effects directly:

```python
import numpy as np

rng = np.random.default_rng(0)
for d in (2, 64, 1024):
    a = rng.standard_normal((2000, d))
    b = rng.standard_normal((2000, d))
    dots = (a * b).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.abs(dots / (na * nb))
    print(d, dots.std().round(1),
          cos.mean().round(3))
```

```text
2    1.4  0.645
64   7.9  0.098
1024 32.6 0.026
```

The middle column is the typical size of the dot product: about
`sqrt(2) = 1.4`, `sqrt(64) = 8`, `sqrt(1024) = 32`. The right column is
the average cosine: it shrinks toward zero as dimension grows.

Hold on to that `sqrt(d)`. It is the reason attention has a square root
in it, which is the subject of [Module 5](05-attention.md).

## In code

```python
import numpy as np
from numpy.linalg import norm

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

dot = a @ b                   # 32.0
cos = dot / (norm(a) * norm(b))
print(round(cos, 3))          # 0.975

# A layer: 4 detectors, 3 inputs.
W = np.ones((4, 3))   # shape (4, 3)
out = W @ a           # shape (4,)
```

## Common mistakes

- **Mixing normalized and unnormalized vectors.** If your index stores
  unit-length vectors, normalize the query too, or scores are skewed.
- **Comparing embeddings from different models.** Each model has its own
  space. A vector from model A means nothing to model B.
- **Silent broadcasting.** `(n,)` and `(n, 1)` look alike and behave
  differently. Adding them gives an `(n, n)` grid without any error.
- **Reading too much into one dimension.** Meaning lives in directions,
  which are usually blends of many dimensions, not in single entries.

## Check yourself

<details>
<summary>1. Two vectors have a dot product of zero. What does that tell
you?</summary>

They are perpendicular: neither has any component along the other. In
the similarity reading, they are unrelated. (It could also mean one of
them is all zeros.)
</details>

<details>
<summary>2. You double the length of one vector. What happens to the dot
product, and to the cosine similarity?</summary>

The dot product doubles. The cosine does not change, because the length
is divided out.
</details>

<details>
<summary>3. A layer has weight shape (512, 768). What size input does it
take, and what size output does it give?</summary>

It takes a vector of size 768 and returns one of size 512. Each of the
512 rows is one detector that is dotted with the input.
</details>

<details>
<summary>4. Why can a 1,000-dimension space hold far more than 1,000
distinguishable concepts?</summary>

Only 1,000 directions can be exactly perpendicular, but a vastly larger
number can be nearly perpendicular. Random directions in high dimensions
barely overlap, so a model can pack many features in with only a little
interference between them.
</details>

## Takeaways

- A vector is a point in space, and direction carries meaning.
- The dot product measures agreement between two vectors.
- A neuron is a dot product. A layer is many of them at once.
- Cosine similarity is the dot product with length removed.
- In high dimensions, dot products grow like `sqrt(d)`. Remember this.

---

[Course home](README.md) · [Next: Loss functions →](02-loss-functions.md)
