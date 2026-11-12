"""Lab 04: retrieval-augmented generation over the course itself.

Three commands, each one stage of the pipeline in Module 9:

    python rag.py index              # chunk + embed the course, save index
    python rag.py eval-retrieval     # is the right chunk in the top k?
    python rag.py ask "Why divide attention by sqrt(d)?"
    python rag.py eval-answers       # answer all questions, judge them

It talks to any OpenAI-compatible server. By default that is Ollama
from Lab 02. Settings can be changed with environment variables:

    BASE_URL      default http://localhost:11434/v1
    CHAT_MODEL    default llama3.2:3b
    EMBED_MODEL   default nomic-embed-text
"""

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

HERE = Path(__file__).parent
COURSE = HERE.parent.parent           # learning/
OUT = HERE / "out"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:11434/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text was trained with these task prefixes. Other embedding
# models may want different ones, or none.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

MAX_CHUNK_CHARS = 1500
TOP_K = 5

client = OpenAI(base_url=BASE_URL, api_key="unused")


# ------------------------------------------------------------- chunking

def chunk_markdown(path):
    """Split one module on its ## and ### headings.

    Each chunk keeps its file, title and heading as metadata, and the
    title + heading are put at the top of the text, so a chunk still
    makes sense on its own. Long sections are split on blank lines.
    """
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    chunks, heading, lines = [], title, []

    def flush():
        body = "\n".join(lines).strip()
        if len(body) < 80:            # skip near-empty sections
            return
        for part in split_long(body):
            chunks.append({
                "file": path.name,
                "heading": heading,
                "text": f"{title} > {heading}\n\n{part}",
            })

    for line in text.splitlines()[1:]:
        if re.match(r"^#{2,3} ", line):
            flush()
            heading, lines = line.lstrip("# ").strip(), []
        elif line.startswith("[") and "Course home" in line:
            continue                   # navigation links
        else:
            lines.append(line)
    flush()
    return chunks


def split_long(body):
    """Split on paragraph boundaries so no piece is far over the limit."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


# ------------------------------------------------------------- embedding

def embed(texts, prefix):
    """Embed in batches and return unit-length vectors (Module 1)."""
    vectors = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        res = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in res.data)
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_index():
    if not (OUT / "chunks.json").exists():
        sys.exit("No index yet. Run: python rag.py index")
    chunks = json.loads((OUT / "chunks.json").read_text())
    vectors = np.load(OUT / "vectors.npy")
    return chunks, vectors


def search(question, chunks, vectors, k=TOP_K):
    """The whole of vector search: one matrix-vector product."""
    q = embed([question], QUERY_PREFIX)[0]
    scores = vectors @ q
    best = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in best]


# ------------------------------------------------------------- commands

def cmd_index():
    files = sorted(COURSE.glob("[0-9][0-9]-*.md"))
    chunks = [c for f in files for c in chunk_markdown(f)]
    print(f"{len(files)} modules -> {len(chunks)} chunks")
    vectors = embed([c["text"] for c in chunks], DOC_PREFIX)
    OUT.mkdir(exist_ok=True)
    (OUT / "chunks.json").write_text(json.dumps(chunks, indent=1))
    np.save(OUT / "vectors.npy", vectors)
    print(f"Saved {vectors.shape[0]} vectors of size {vectors.shape[1]} "
          f"to {OUT}")


def is_hit(chunk, q):
    return chunk["file"] == q["file"] and chunk["heading"] in q["headings"]


def margin(p, n):
    """95% range of a rate measured on n examples (Module 11)."""
    return 1.96 * math.sqrt(p * (1 - p) / n)


def cmd_eval_retrieval():
    chunks, vectors = load_index()
    questions = json.loads((HERE / "questions.json").read_text())
    hit1 = hitk = 0
    print(f"{'rank':>4}  question")
    for q in questions:
        results = search(q["question"], chunks, vectors)
        ranks = [i + 1 for i, (c, _) in enumerate(results) if is_hit(c, q)]
        rank = ranks[0] if ranks else None
        hit1 += rank == 1
        hitk += rank is not None
        print(f"{rank or '-':>4}  {q['question']}")
        if rank is None:
            got = results[0][0]
            print(f"      top result: {got['file']} > {got['heading']}")
    n = len(questions)
    print()
    print(f"hit@1  {hit1 / n:.0%} +/- {margin(hit1 / n, n):.0%}")
    print(f"hit@{TOP_K}  {hitk / n:.0%} +/- {margin(hitk / n, n):.0%}")


ANSWER_PROMPT = """Answer the question using only the numbered sources.
Cite sources like [1]. If the sources do not contain the answer, say
"The sources do not cover this."

{sources}

Question: {question}"""

JUDGE_PROMPT = """You are grading an answer for faithfulness.

Sources:
{sources}

Answer to grade:
{answer}

Is every claim in the answer supported by the sources?
Reply with exactly PASS or FAIL on the first line,
then one sentence explaining why."""


def format_sources(results):
    return "\n\n".join(f"[{i + 1}] {c['text']}"
                       for i, (c, _) in enumerate(results))


def chat(prompt):
    res = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, max_tokens=400,
        messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content.strip()


def answer(question, chunks, vectors, k=4):
    results = search(question, chunks, vectors, k)
    sources = format_sources(results)
    reply = chat(ANSWER_PROMPT.format(sources=sources, question=question))
    return reply, results, sources


def cmd_ask(question):
    chunks, vectors = load_index()
    reply, results, _ = answer(question, chunks, vectors)
    print(reply)
    print("\nSources:")
    for i, (c, score) in enumerate(results):
        print(f"[{i + 1}] {score:.3f}  {c['file']} > {c['heading']}")


def cmd_eval_answers():
    chunks, vectors = load_index()
    questions = json.loads((HERE / "questions.json").read_text())
    passed = 0
    for q in questions:
        reply, _, sources = answer(q["question"], chunks, vectors)
        verdict = chat(JUDGE_PROMPT.format(sources=sources, answer=reply))
        ok = verdict.upper().startswith("PASS")
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {q['question']}")
        if not ok:
            print(f"      judge: {verdict.splitlines()[-1][:150]}")
    n = len(questions)
    p = passed / n
    print(f"\nfaithful  {p:.0%} +/- {margin(p, n):.0%}  ({passed}/{n})")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "index":
        cmd_index()
    elif cmd == "eval-retrieval":
        cmd_eval_retrieval()
    elif cmd == "ask" and len(sys.argv) > 2:
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "eval-answers":
        cmd_eval_answers()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
