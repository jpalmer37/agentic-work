"""Lab 05: generate a small, messy extraction dataset.

Each example is a customer message written in many different styles.
The target is clean JSON with four fields:

    {"name": "Priya Shah", "email": "priya.shah@example.com",
     "city": "Vancouver", "total": 1234.5}

The messages include traps a model must learn to avoid: order numbers
and phone numbers that look like totals, names written surname-first,
and totals with currency symbols and thousands separators.

    python make_data.py          # writes data/train.jsonl, data/test.jsonl
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent

FIRST = ["Priya", "Liam", "Mei", "Omar", "Sofia", "Kenji", "Amara", "Lucas",
         "Fatima", "Noah", "Ines", "Raj", "Hannah", "Mateo", "Aiko", "Daniel",
         "Zara", "Felix", "Leila", "Jonah", "Chloe", "Tariq", "Elena", "Sam"]
LAST = ["Shah", "Nguyen", "Okafor", "Martin", "Rossi", "Tanaka", "Haddad",
        "Silva", "Kowalski", "Brown", "Dubois", "Patel", "Murphy", "Garcia",
        "Kim", "Johansson", "Ali", "Schmidt", "Moreau", "Chen", "Lopez"]
CITIES = ["Vancouver", "Toronto", "Montreal", "Calgary", "Halifax", "Ottawa",
          "Victoria", "Winnipeg", "Regina", "Edmonton", "Kelowna", "Quebec City"]
DOMAINS = ["example.com", "mail.example.org", "inbox.example.net"]

TEMPLATES = [
    "hi, this is {name_text} from {city_text}. please charge {total_text} "
    "for order #{order}. you can reach me at {email}",
    "Order {order} - total {total_text}. Ship to {city_text}. "
    "Contact: {name_text} <{email}>",
    "{name_text} here ({email}). I'm in {city_text} and my invoice says "
    "{total_text}. My phone is {phone} if needed.",
    "From: {email}\nName: {name_text}\nRe: order {order}\n\nHello, "
    "amount due is {total_text}, delivery to {city_text}. Thanks!",
    "Can you confirm the payment of {total_text}? Ticket {order}. "
    "{name_text}, {city_text}. Email {email}, phone {phone}.",
    "URGENT!!! {name_text} ({city_text}) was billed {total_text} for order "
    "#{order}. reply to {email}",
    "Customer {name_text}, based in {city_text}. Total: {total_text}. "
    "Ref {order}. {email}",
]


def money_text(total, rng):
    whole = f"{total:,.2f}" if rng.random() < 0.5 else f"{total:.2f}"
    style = rng.choice(["${}", "{} dollars", "CAD {}", "{} CAD", "$ {}"])
    if rng.random() < 0.3 and total == int(total):
        whole = whole.split(".")[0]
    return style.format(whole)


def example(rng):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    city = rng.choice(CITIES)
    total = round(rng.choice([rng.uniform(5, 200), rng.uniform(200, 5000)]),
                  rng.choice([0, 2, 2, 2]))
    email = f"{first.lower()}.{last.lower()}@{rng.choice(DOMAINS)}"

    name_text = rng.choice([
        f"{first} {last}", f"{first} {last}", f"{last}, {first}",
        f"{first.upper()} {last.upper()}", f"{first.lower()} {last.lower()}",
    ])
    city_text = rng.choice([city, city, city.lower(), city.upper()])

    message = rng.choice(TEMPLATES).format(
        name_text=name_text, city_text=city_text, email=email,
        total_text=money_text(total, rng),
        order=rng.randint(10000, 99999),
        phone=f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-"
              f"{rng.randint(1000, 9999)}",
    )
    target = {"name": f"{first} {last}", "email": email, "city": city,
              "total": total}
    return {"message": message, "target": target}


def main():
    rng = random.Random(42)
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for name, n in [("train", 500), ("test", 100)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps(example(rng)) + "\n")
        print(f"wrote {n} examples to {out / name}.jsonl")
    print("\nOne example:")
    sample = example(random.Random(7))
    print(sample["message"])
    print("->", json.dumps(sample["target"]))


if __name__ == "__main__":
    main()
