"""Lab 01: generate text from the model that train.py saved.

    python generate.py                       # temperatures 0.5, 1.0, 1.5
    python generate.py --prompt "ROMEO:" --temps 0 0.8
"""

import argparse
from pathlib import Path

import torch

from train import TinyGPT

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(HERE / "out" / "model.pt"))
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--length", type=int, default=250)
    p.add_argument("--temps", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    chars, cfg = ckpt["chars"], ckpt["config"]
    index = {c: i for i, c in enumerate(chars)}

    model = TinyGPT(len(chars), cfg["d"], cfg["layers"], cfg["heads"],
                    cfg["context"], ckpt["scale_attention"],
                    ckpt["use_positions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = [index[c] for c in args.prompt if c in index]
    for t in args.temps:
        torch.manual_seed(0)
        prompt = torch.tensor([ids], device=device)
        out = model.generate(prompt, args.length, t)[0].tolist()
        print(f"===== temperature {t} =====")
        print("".join(chars[i] for i in out))
        print()


if __name__ == "__main__":
    main()
