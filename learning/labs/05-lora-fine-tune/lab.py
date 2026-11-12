"""Lab 05: LoRA fine-tuning, measured honestly.

    python lab.py baseline              # the base model, prompt only
    python lab.py train                 # LoRA fine-tune, saves an adapter
    python lab.py evaluate              # the fine-tuned model
    python lab.py evaluate --4bit       # the same, base weights in 4-bit
    python lab.py report                # table, error bars, case-by-case

Every run appends to out/results.json, so report can compare them.
Covers Module 4 (training loop), Module 9 (LoRA) and Module 11 (evals).
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # Apache 2.0, checked Nov 2026

INSTRUCTION = (
    "Extract the customer's name, email, city and order total from the "
    "message. Reply with JSON only, using exactly these keys: "
    '"name" (first name then last name, normal capitalisation), '
    '"email", "city" (normal capitalisation), '
    '"total" (a number, no currency symbol).'
)


# ------------------------------------------------------------- loading

def load_rows(name):
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit("No data yet. Run: python make_data.py")
    return [json.loads(line) for line in path.open()]


def load_model(name, four_bit=False, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"           # generation needs left padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"dtype": torch.bfloat16 if cuda else torch.float32}
    if four_bit:
        if not cuda:
            raise SystemExit("4-bit loading needs an NVIDIA GPU.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not four_bit:
        model.to("cuda" if cuda else "cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def prompt_text(tok, message):
    """The chat template the model was trained with (Module 7)."""
    msgs = [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": message}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def answer_text(target):
    return json.dumps(target)


# ------------------------------------------------------------- scoring

def parse(output):
    """Pull the first {...} out of the output and parse it."""
    m = re.search(r"\{.*?\}", output, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def field_ok(key, got, want):
    if key == "total":
        try:
            return abs(float(got) - float(want)) < 0.005
        except (TypeError, ValueError):
            return False
    return isinstance(got, str) and got.strip() == want


def score(output, target):
    got = parse(output)
    if got is None:
        return {k: False for k in target} | {"valid_json": False}
    return {k: field_ok(k, got.get(k), v) for k, v in target.items()} | {
        "valid_json": True}


# ------------------------------------------------------------- evaluation

@torch.no_grad()
def generate_all(model, tok, messages, batch=16, max_new_tokens=96):
    model.eval()
    outputs = []
    device = next(model.parameters()).device
    for i in range(0, len(messages), batch):
        prompts = [prompt_text(tok, m) for m in messages[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        outputs += tok.batch_decode(new, skip_special_tokens=True)
    return outputs


def run_eval(label, model, tok, limit=None):
    rows = load_rows("test")[:limit]
    start = time.time()
    outputs = generate_all(model, tok, [r["message"] for r in rows])
    per_case = [score(o, r["target"]) for o, r in zip(outputs, rows)]
    exact = [all(s[k] for k in ("name", "email", "city", "total"))
             for s in per_case]

    result = {
        "label": label,
        "n": len(rows),
        "exact": sum(exact) / len(rows),
        "valid_json": sum(s["valid_json"] for s in per_case) / len(rows),
        "fields": {k: sum(s[k] for s in per_case) / len(rows)
                   for k in ("name", "email", "city", "total")},
        "cases": exact,
        "seconds": round(time.time() - start, 1),
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "results.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history = [h for h in history if h["label"] != label] + [result]
    path.write_text(json.dumps(history, indent=1))

    print(f"\n{label}: all four fields right in "
          f"{result['exact']:.0%} of {len(rows)} messages")
    print("per field:", {k: f"{v:.0%}" for k, v in result["fields"].items()})
    print(f"valid JSON: {result['valid_json']:.0%}")
    wrong = [i for i, ok in enumerate(exact) if not ok][:3]
    for i in wrong:
        print(f"\n  message: {rows[i]['message'][:120]!r}")
        print(f"  wanted:  {answer_text(rows[i]['target'])}")
        print(f"  got:     {outputs[i].strip()[:120]!r}")
    return result


# ------------------------------------------------------------- training

def make_batch(tok, rows, device):
    """Tokenize prompt + answer. Loss only on the answer tokens."""
    ids, labels = [], []
    for r in rows:
        p = tok(prompt_text(tok, r["message"]), add_special_tokens=False)
        a = tok(answer_text(r["target"]) + tok.eos_token,
                add_special_tokens=False)
        ids.append(p["input_ids"] + a["input_ids"])
        # -100 means "ignore": the model is not graded on the prompt.
        labels.append([-100] * len(p["input_ids"]) + a["input_ids"])
    # Pad on the right for training (generation pads on the left).
    width = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids])
    label_ids = torch.tensor([y + [-100] * (width - len(y)) for y in labels])
    mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
    return (input_ids.to(device), mask.to(device), label_ids.to(device))


def cmd_train(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, tok = load_model(args.model, four_bit=args.four_bit)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model)  # this is QLoRA
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    rows = load_rows("train")
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, args.steps // 10)
    g = torch.Generator().manual_seed(0)
    model.train()
    start = time.time()

    for step in range(args.steps):
        lr = args.lr * min(1, (step + 1) / warmup) * 0.5 * (
            1 + math.cos(math.pi * max(0, step - warmup) /
                         max(1, args.steps - warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        pick = torch.randint(len(rows), (args.batch,), generator=g).tolist()
        input_ids, mask, labels = make_batch(tok, [rows[i] for i in pick],
                                             device)
        loss = model(input_ids=input_ids, attention_mask=mask,
                     labels=labels).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.3f}  "
                  f"lr {lr:.1e}  {time.time() - start:5.0f}s")

    path = OUT / "adapter"
    model.save_pretrained(path)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"\nSaved the adapter to {path} ({size / 1e6:.1f} MB)")


# ------------------------------------------------------------- report

def cmd_report():
    path = OUT / "results.json"
    if not path.exists():
        raise SystemExit("No results yet. Run baseline and evaluate first.")
    history = json.loads(path.read_text())
    print(f"{'run':<22} {'exact':>6} {'+/-':>5} {'JSON':>5} {'secs':>6}")
    for h in history:
        m = 1.96 * math.sqrt(h["exact"] * (1 - h["exact"]) / h["n"])
        print(f"{h['label']:<22} {h['exact']:>6.0%} {m:>5.0%} "
              f"{h['valid_json']:>5.0%} {h['seconds']:>6}")

    # Case by case against the first run (Module 11): which messages did
    # each later run fix, and which did it break?
    base = history[0]
    for h in history[1:]:
        fixed = sum(not a and b for a, b in zip(base["cases"], h["cases"]))
        broke = sum(a and not b for a, b in zip(base["cases"], h["cases"]))
        print(f"\n{h['label']} vs {base['label']}: "
              f"fixed {fixed}, broke {broke}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command",
                   choices=["baseline", "train", "evaluate", "report"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--4bit", dest="four_bit", action="store_true",
                   help="load the base model in 4-bit (needs a GPU)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--limit", type=int, help="evaluate on fewer messages")
    args = p.parse_args()
    torch.manual_seed(0)

    if args.command == "baseline":
        model, tok = load_model(args.model, four_bit=args.four_bit)
        label = "base, prompt only" + (", 4-bit" if args.four_bit else "")
        run_eval(label, model, tok, args.limit)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        model, tok = load_model(args.model, four_bit=args.four_bit,
                                adapter=str(OUT / "adapter"))
        label = "fine-tuned" + (", 4-bit" if args.four_bit else "")
        run_eval(label, model, tok, args.limit)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
