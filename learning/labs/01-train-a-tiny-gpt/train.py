"""Lab 01: train a tiny character-level GPT from nothing.

Every piece here is described in the course:
  tokenizer      -> Module 6 (here: one token per character)
  attention      -> Module 5 (causal, scaled by sqrt(d))
  block          -> Module 6 (pre-norm, residual stream, MLP)
  loss           -> Module 2 (next-token cross-entropy)
  optimizer      -> Module 4 (AdamW, warmup + cosine, clipping)

Usage (from this folder, with ../.venv active):
    python train.py                 # GPU config
    python train.py --cpu           # small config for a CPU
    python train.py --no-scale      # experiment: drop the 1/sqrt(d)
    python train.py --no-pos        # experiment: drop position info
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DATA_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"  # Shakespeare

CONFIGS = {
    # About 10.7M parameters. A few minutes on an RTX 4070.
    "gpu": dict(d=384, layers=6, heads=6, context=256, batch=64,
                steps=3000, lr=1e-3, eval_every=250),
    # About 0.8M parameters. A few minutes on a laptop CPU.
    "cpu": dict(d=128, layers=4, heads=4, context=64, batch=32,
                steps=1500, lr=2e-3, eval_every=150),
}


# ---------------------------------------------------------------- data

def load_text(path):
    """Use --text if given, else download Shakespeare once."""
    if path:
        return Path(path).read_text(encoding="utf-8")
    cache = HERE / "data" / "shakespeare.txt"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        print(f"Downloading {DATA_URL}")
        raw = urllib.request.urlopen(DATA_URL).read().decode("utf-8")
        # Keep the plays, drop the Project Gutenberg header and licence.
        start = raw.find("*** START")
        end = raw.find("*** END")
        body = raw[raw.find("\n", start) + 1:end] if start >= 0 else raw
        cache.write_text(body, encoding="utf-8")
    return cache.read_text(encoding="utf-8")


class CharTokenizer:
    """The simplest tokenizer: every distinct character is one token."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.index[c] for c in s if c in self.index]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def batches(data, context, batch, device):
    """Random windows. Targets are the inputs shifted by one token."""
    starts = torch.randint(len(data) - context - 1, (batch,))
    x = torch.stack([data[s:s + context] for s in starts])
    y = torch.stack([data[s + 1:s + context + 1] for s in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model

class Block(nn.Module):
    """One pre-norm transformer block: x += attn(norm(x)); x += mlp(norm(x))."""

    def __init__(self, d, heads, scale_attention):
        super().__init__()
        self.heads = heads
        self.scale_attention = scale_attention
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def attend(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, d) -> (B, heads, T, d / heads)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2)
                   for t in (q, k, v))
        # The fused kernel computes softmax(q k^T * scale) v with a causal
        # mask. Its default scale is 1/sqrt(head size); --no-scale sets 1.
        scale = None if self.scale_attention else 1.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           scale=scale)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))

    def forward(self, x):
        x = x + self.attend(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, context,
                 scale_attention=True, use_positions=True):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab, d)
        self.positions = nn.Embedding(context, d) if use_positions else None
        self.blocks = nn.ModuleList(
            Block(d, heads, scale_attention) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, vocab, bias=False)
        self.unembed.weight = self.tokens.weight  # weight tying, Module 6
        self.apply(self._init)

    @staticmethod
    def _init(module):
        # Small random starting weights, as in GPT-2. PyTorch's default
        # embedding init has a spread of 1, which makes the first logits
        # huge and the first loss far above ln(vocab). Try removing this
        # line and look at step 0: that is the "bad init" warning sign
        # from Module 2.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids):
        x = self.tokens(ids)
        if self.positions is not None:
            pos = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.positions(pos)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.norm(x))  # logits (B, T, vocab)

    @torch.no_grad()
    def generate(self, ids, steps, temperature=1.0):
        """Module 8: predict, sample, append, repeat."""
        for _ in range(steps):
            logits = self(ids[:, -self.context:])[:, -1]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


# ---------------------------------------------------------------- training

def lr_at(step, peak, total, warmup):
    """Linear warmup, then cosine decay to zero (Module 4)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    done = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * done))


@torch.no_grad()
def estimate_loss(model, data, cfg, device, autocast, n=20):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = batches(data, cfg["context"], cfg["batch"], device)
        with autocast():
            logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1).float(),
                                      y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="small CPU config")
    p.add_argument("--text", help="train on your own .txt file instead")
    p.add_argument("--steps", type=int, help="override number of steps")
    p.add_argument("--no-scale", action="store_true",
                   help="experiment: remove the 1/sqrt(d) in attention")
    p.add_argument("--no-pos", action="store_true",
                   help="experiment: remove position embeddings")
    p.add_argument("--out", default="out", help="folder for results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cfg = dict(CONFIGS["cpu" if device == "cpu" else "gpu"])
    if args.steps:
        cfg["steps"] = args.steps
    torch.manual_seed(1337)

    # bfloat16 mixed precision on the GPU (Module 4). Plain float32 on CPU.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type=device, dtype=torch.bfloat16,
                              enabled=use_bf16)

    text = load_text(args.text)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = TinyGPT(len(tok.chars), cfg["d"], cfg["layers"], cfg["heads"],
                    cfg["context"], scale_attention=not args.no_scale,
                    use_positions=not args.no_pos).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"device       {device}" + (" (bf16)" if use_bf16 else ""))
    print(f"text         {len(text):,} characters")
    print(f"vocabulary   {len(tok.chars)} tokens")
    print(f"parameters   {n_params / 1e6:.2f}M")
    print(f"ln(vocab)    {math.log(len(tok.chars)):.2f}  "
          "<- expected starting loss (Module 2)")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=0.1)
    warmup = max(1, cfg["steps"] // 20)
    history = []
    out = HERE / args.out
    out.mkdir(exist_ok=True)
    start = time.time()

    for step in range(cfg["steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            train_loss = estimate_loss(model, train_data, cfg, device, autocast)
            val_loss = estimate_loss(model, val_data, cfg, device, autocast)
            history.append(dict(step=step, train=train_loss, val=val_loss))
            print(f"step {step:5d}  train {train_loss:.3f}  "
                  f"val {val_loss:.3f}  {time.time() - start:6.0f}s")
        if step == cfg["steps"]:
            break

        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg["lr"], cfg["steps"], warmup)
        x, y = batches(train_data, cfg["context"], cfg["batch"], device)
        with autocast():
            logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Save everything generate.py needs.
    torch.save(dict(model=model.state_dict(), chars=tok.chars,
                    config=cfg, n_params=n_params,
                    scale_attention=not args.no_scale,
                    use_positions=not args.no_pos),
               out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    plot(history, out / "loss.png", math.log(len(tok.chars)))

    print("\nSample at temperature 0.8:\n")
    prompt = torch.tensor([tok.encode("\n")], device=device)
    model.eval()
    print(tok.decode(model.generate(prompt, 300, 0.8)[0].tolist()))
    print(f"\nSaved {out / 'model.pt'}. Try: python generate.py")


def plot(history, path, start_loss):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, [h["train"] for h in history], label="train")
    ax.plot(steps, [h["val"] for h in history], label="validation")
    ax.axhline(start_loss, ls="--", c="grey", label="ln(vocab)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Loss curve saved to {path}")


if __name__ == "__main__":
    main()
