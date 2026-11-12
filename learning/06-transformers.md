# 6. Transformers

## Why this matters

Nearly every model you will work with, from chat assistants to embedding
models to speech recognition, is a transformer. They share one simple
repeating unit. Once you can draw that unit and say what each part is
for, model cards, configuration files and research papers stop being
intimidating.

## The idea in plain words

A transformer is a tall stack of identical **blocks**. Each block does
two things to every token:

1. **Attention:** gather relevant information from other tokens.
2. **Feed-forward layer:** process what was gathered.

Around each of those sits a residual connection and a normalization
layer to keep training stable. Text goes in as tokens. A probability for
every possible next token comes out.

## The whole pipeline

```text
text
 -> tokenizer     ids       (T,)
 -> embedding     vectors   (T, d)
 -> block 1
 -> block 2       vectors   (T, d)
 -> ...
 -> block N
 -> final norm
 -> unembedding   logits    (T, vocab)
 -> softmax       probabilities
```

The shape `(T, d)` never changes through the stack. Every block takes
`T` vectors of size `d` and returns `T` vectors of size `d`. That
uniformity is why you can stack as many blocks as you can afford.

## Tokenization

Models read **tokens**, not characters or words. A tokenizer splits text
into pieces from a fixed vocabulary, usually 30,000 to 200,000 entries.

```text
"unbelievable results"
 -> "un" "believ" "able" " results"
 -> 403  12011    481    3719
```

Why pieces of words?

- **Characters** would make sequences very long, and attention cost
  grows with the square of length.
- **Whole words** would need an endless vocabulary and could not handle
  typos, names or new terms.
- **Subwords** are the compromise: common words are one token, rare
  words are built from parts, and nothing is ever unrepresentable.

Practical consequences:

- In English, one token is roughly four characters, or about three
  quarters of a word. Code and other languages often use more.
- Cost, speed and context limits are all counted in tokens.
- Each model family has its own tokenizer. Counts are not transferable.
- Models are weak at tasks about letters, like counting the r's in a
  word, because they never see the letters.

## The residual stream

This is the most useful mental model for a transformer.

Each token owns one vector of size `d`. That vector travels straight up
through the whole network. Blocks do not replace it. They **read** from
it, compute something, and **add** the result back.

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

Think of it as a shared whiteboard for each token. Early layers jot down
simple facts ("this is a noun", "previous token was 'the'"). Later
layers read those notes and add more abstract ones. The final layer
reads the whiteboard to predict the next token.

It is also the gradient highway from
[Module 3](03-backpropagation.md): the plus signs give gradients a
direct route to every layer.

## Inside one block

```text
    x  (T, d)
    |
    +----------+
    |          |
  norm         |
    |          |
 attention     |
    |          |
    +<---------+   x = x + attention
    |
    +----------+
    |          |
  norm         |
    |          |
   MLP         |
    |          |
    +<---------+   x = x + mlp
    |
    x  (T, d)
```

### The attention sub-layer

Exactly what [Module 5](05-attention.md) described: multi-head
self-attention with a causal mask. This is the only place where tokens
exchange information.

### The feed-forward sub-layer (MLP)

A small two-layer network applied to **each token separately**:

```text
mlp(x) = W2 @ activation(W1 @ x)

W1: d  -> 4d     expand
W2: 4d -> d      project back
```

It expands the vector to four times its size, applies a non-linearity,
and shrinks it back.

Why expand? Recall that a layer is a bank of pattern detectors. The wide
middle layer gives the block a lot of detectors. Each one fires on some
pattern in the token's vector, and the second matrix writes associated
information back to the residual stream.

A helpful picture: the MLP acts like a learned memory. "If the vector
looks like *the capital of France*, add the direction for *Paris*."

The MLP holds about two thirds of a block's weights. Much of what a
model "knows" is stored here.

### Normalization and where it goes

The original transformer normalized **after** each addition. Modern
models normalize **before** each sub-layer, as in the diagram above.
This is called **pre-norm**.

The difference matters. With pre-norm, the residual path from input to
output is completely clean: nothing but additions. Deep stacks train far
more reliably, with less need for delicate warmup.

## Telling the model about order

Attention treats its input as an unordered set
([Module 5](05-attention.md)). Position must be supplied separately.
There are three generations of solution.

**1. Sinusoidal positions (the original).** Add a fixed pattern of sine
and cosine waves to each embedding. Each position gets a unique
fingerprint. No learned weights.

**2. Learned positions (GPT-2, BERT).** A second embedding table with
one learned row per position, added to the token embedding. Simple, but
the model cannot handle sequences longer than the table.

**3. Rotary positions, "RoPE" (most current LLMs).** Do not add anything
to the embeddings. Instead, inside every attention layer, **rotate** each
query and key by an angle proportional to its position.

The intuition is a clock. Rotate one hand by an amount for position 5
and another by an amount for position 8. The angle between them depends
only on the gap of 3, not on 5 or 8 themselves. Since a dot product
depends on the angle between vectors, attention scores now depend on
**how far apart** two tokens are.

That is usually what matters in language: "the word two places back" is
more useful than "the word at position 4,817". Relative schemes also
cope better with lengths not seen in training.

## Three families of transformer

| Family | Sees | Good for |
|---|---|---|
| Encoder-only | Whole input, both directions | Embeddings, classification, search |
| Decoder-only | Only the past | Text generation. All current LLMs |
| Encoder-decoder | Encoder: all. Decoder: past plus encoder output | Translation, speech to text |

**Encoder-only** models (the BERT family) have no causal mask. Every
token sees the whole input, which is ideal for understanding a fixed
piece of text.

**Decoder-only** models (the GPT family and nearly every LLM since) use
the causal mask and generate one token at a time.

**Encoder-decoder** models (the original transformer, T5, Whisper) read
the input with an encoder, then a decoder generates output while using
cross-attention to look back at the encoder's result.

Decoder-only won for general use because it is the simplest: any task
can be phrased as "continue this text".

## Getting an answer out

After the last block, each token's final vector is turned into one score
per vocabulary entry by the **unembedding** matrix. That is one more
bank of dot products: how well does this final vector match each
token's output direction?

Many models reuse the input embedding table for this, a trick called
**weight tying**. It saves a large number of weights and works well.

## Counting parameters: GPT-2 small

A good way to check your understanding is to count the weights. GPT-2
small has `d = 768`, 12 layers, a vocabulary of 50,257 and a context of
1,024 tokens.

One block:

```text
attention  4*d*d + 4*d  = 2,362,368
MLP        8*d*d + 5*d  = 4,722,432
norms      4*d          =     3,072
                          ---------
one block                 7,087,872
```

Attention has four `d` by `d` matrices: queries, keys, values and the
output mix. The MLP has two matrices of `d` by `4d`. The smaller terms
are biases.

The whole model:

```text
12 blocks             85,054,464
token embeddings      38,597,376
position embeddings      786,432
final norm                 1,536
                     -----------
total                124,439,808
```

The token table is 50,257 rows of 768. The position table is 1,024
rows of 768.

That is the published figure. No hidden machinery: 124 million numbers
arranged as described on this page.

A rule of thumb follows from the block count. Ignoring small terms, each
block has about `12 * d * d` weights, so:

```text
parameters  ~=  12 * layers * d^2
```

## Where the compute goes

Per block, for `T` tokens of size `d`:

```text
attention   grows like  T^2 * d
MLP         grows like  T * d^2
```

For short inputs the MLP dominates. As the sequence gets long, attention
takes over because of the `T^2`. This is why long prompts cost more than
their length alone suggests.

## In code (PyTorch)

A complete pre-norm block, with the attention from
[Module 5](05-attention.md) written out:

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def split(self, t):
        # (B,T,d) -> (B,h,T,d/h)
        B, T, _ = t.shape
        h = self.heads
        t = t.view(B, T, h, -1)
        return t.transpose(1, 2)

    def attend(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.split, qkv)
        hd = d // self.heads
        s = q @ k.transpose(-1, -2)
        s = s / hd ** 0.5
        # causal mask: hide the future
        mask = torch.ones(T, T).triu(1)
        s = s - 1e9 * mask
        y = s.softmax(-1) @ v
        y = y.transpose(1, 2)
        y = y.reshape(B, T, d)
        return self.proj(y)

    def forward(self, x):
        a = self.attend(self.norm1(x))
        x = x + a
        m = self.mlp(self.norm2(x))
        return x + m

block = Block(768, 12)
params = block.parameters()
print(sum(p.numel() for p in params))
# 7087872
```

The count matches the hand calculation above. A full language model is
an embedding table, a stack of these, a final norm and an unembedding.

In real code you would call PyTorch's built-in
`scaled_dot_product_attention` instead of writing the middle of
`attend` by hand. It computes the same thing, much faster.

## Common mistakes

- **Thinking the model sees words or characters.** It sees token ids.
  Many odd behaviours trace back to how text was split.
- **Estimating cost in words.** Budgets, limits and bills are in tokens.
- **Assuming every transformer generates text.** Encoder-only models do
  not. They are the usual choice for embeddings and classification.
- **Forgetting position.** If you build a block yourself and skip
  positional information, the model cannot tell word order at all.
- **Comparing models by parameter count alone.** Training data and
  recipe matter at least as much.

## Check yourself

<details>
<summary>1. What are the two sub-layers in a block, and what is each
for?</summary>

Attention lets tokens exchange information. The feed-forward layer (MLP)
processes each token on its own, using what attention gathered. Each is
wrapped in a residual add with a normalization step.
</details>

<details>
<summary>2. What is the residual stream?</summary>

The per-token vector that runs through the whole network. Every
sub-layer reads from it and adds its output back, so information
accumulates rather than being replaced. It is also the direct path that
gradients follow back to early layers.
</details>

<details>
<summary>3. Why do transformers need positional information at
all?</summary>

Attention computes the same result whatever order the tokens are in.
Without position, "dog bites man" and "man bites dog" are identical to
the model.
</details>

<details>
<summary>4. How does rotary position encoding make attention depend on
relative distance?</summary>

Queries and keys are rotated by an angle proportional to their position.
The dot product between them depends on the difference of the two
angles, which is the distance between the tokens.
</details>

<details>
<summary>5. Roughly how many parameters does a 24-layer model with
d = 1024 have, ignoring embeddings?</summary>

About `12 * 24 * 1024^2`, which is roughly 302 million.
</details>

<details>
<summary>6. Which transformer family would you choose for semantic
search over documents, and why?</summary>

An encoder-style embedding model. It sees the whole text in both
directions and produces a single vector per input, which is what search
needs. A generative decoder is the wrong tool for that step.
</details>

## Takeaways

- A transformer is a stack of identical blocks with constant shape.
- Each block: attention to communicate, MLP to compute.
- The residual stream is a per-token whiteboard that layers add to.
- Pre-norm keeps the residual path clean and training stable.
- Position must be added. Rotary encoding is the current default.
- Parameters are about `12 * layers * d^2`, plus embeddings.

---

[← Attention](05-attention.md) ·
[Course home](README.md) ·
[Next: LLM architectures →](07-llm-architectures.md)
