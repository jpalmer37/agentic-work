# Lab 04: RAG over your docs

## Goal

Build a question-answering system over this course, with every stage of
the retrieval pipeline visible, and measure it properly: retrieval and
answers scored separately, with error bars.

## What you will see

- The 14 course modules split into a few hundred chunks.
- Vector search that is a single matrix multiplication.
- A retrieval score on 20 questions, with its error bar.
- Answers that cite their sources, and a judge grading them.
- That changing one thing moves the numbers, and by how much.

## Time and hardware

About an hour. Uses Ollama from Lab 02. Works on a CPU, slowly. The
GPU makes answering much faster.

## Read first

- [Module 1: Vectors](../../01-vectors-and-dot-products.md), for
  embeddings and cosine similarity.
- [Module 9: Adapting models](../../09-adapting-models.md), for the
  RAG pipeline and chunking.
- [Module 11: Evaluation](../../11-evaluation.md), for golden sets,
  judges and error bars.

## The pipeline

```text
OFFLINE: python rag.py index
  14 modules
    -> chunks (one per heading)
    -> embed each chunk
    -> out/vectors.npy

ONLINE: python rag.py ask "..."
  question
    -> embed
    -> vectors @ question
    -> top 4 chunks
    -> prompt: sources + question
    -> answer citing [1], [2]
```

All of it is in `rag.py`, about 200 lines.

## Steps

### 1. Start Ollama and get the models

If you removed Ollama at the end of Lab 02, reinstall it with step 1 of
that lab.

```bash
sudo systemctl start ollama
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`nomic-embed-text` is an embedding model of about 140 million
parameters (about 300 MB). It turns text into a vector rather than
generating text. Checked November 2026.

### 2. Build the index

```bash
cd ~/agent-skills/learning/labs
source .venv/bin/activate
cd 04-rag-over-your-docs
python rag.py index
```

```text
14 modules -> 269 chunks
Saved 269 vectors of size 768
```

Your counts may differ a little if the course has changed.

### 3. Measure retrieval

```bash
python rag.py eval-retrieval
```

For each of the 20 questions in `questions.json`, this shows the rank
at which the correct section appeared, or `-` if it was not in the top
5, and what came first instead:

```text
rank  question
   1  Why is attention divided by...
   3  How many examples does an...
   -  What does a model registry...
      top result: 12-mlops.md >
        Check yourself
```

Then the summary:

```text
hit@1  ..% +/- ..%
hit@5  ..% +/- ..%
```

### 4. Ask questions

```bash
python rag.py ask \
  "Why do LLMs charge more for
   output tokens?"
```

You get an answer with citations, then the sources it was given and
their similarity scores. Try a question the course does not cover, and
check that it says so.

### 5. Measure the answers

```bash
python rag.py eval-answers
```

This answers all 20 questions, then asks the model to judge each answer
for faithfulness: is every claim supported by the sources? It reports a
pass rate with an error bar.

### 6. Change one thing, measure again

Pick one change, re-run `index` and both evals, and compare. Ideas:

- **Smaller chunks.** Set `MAX_CHUNK_CHARS = 600` in `rag.py`.
- **Drop the prefixes.** Set `DOC_PREFIX` and `QUERY_PREFIX` to `""`.
  `nomic-embed-text` was trained to expect them.
- **Skip quiz sections.** In `chunk_markdown`, skip chunks whose
  heading is `Check yourself`. They often outrank the real answer.
- **A different embedding model.** For example:

```bash
ollama pull mxbai-embed-large
EMBED_MODEL=mxbai-embed-large \
  python rag.py index
```

Set both prefixes to `""` for models that do not use them.

## What to notice

**Search is one line.** `vectors @ q` scores every chunk at once. With a
few hundred chunks it takes a fraction of a millisecond. A vector
database only becomes necessary at hundreds of thousands of chunks.

**Error bars are wide.** With 20 questions, a hit rate of 80% has a 95%
range of roughly plus or minus 18 points. Most changes you try will move
the score by less than that. Look at *which* questions changed rank
rather than the headline number, as
[Module 11](../../11-evaluation.md) recommends.

**Two halves, two scores.** If a question fails retrieval, the best
prompt in the world cannot fix it. If retrieval succeeds and the answer
is still wrong, the problem is generation. Measure retrieval first.

**Near misses are informative.** When the wrong chunk wins, read it.
Often it is the module's quiz, or a summary that mentions the same
words. Fixing these is the "error analysis" step that beats clever
tooling.

**The judge is the weakest link.** Here a 3B model grades its own
answers. It will pass some answers it should not. Before trusting its
score, grade ten answers yourself and count how often you agree. That
is how you validate a judge.

**It says when it does not know.** The prompt tells the model to admit
when the sources do not cover a question. Check that it does. A model
that answers anyway is guessing from its training data, not your
documents.

## Spin down and clean up

```bash
rm -rf out
ollama stop llama3.2:3b
ollama rm nomic-embed-text
ollama rm mxbai-embed-large
sudo systemctl stop ollama
```

Ignore an error for a model you never pulled. To remove Ollama
completely, follow step 4 of the clean-up in
[Lab 02](../02-run-a-local-llm/README.md).

## Troubleshooting

**`Connection error`.** Ollama is not running. Run
`sudo systemctl start ollama`.

**`model "nomic-embed-text" not found`.** Pull it first, as in step 1.

**Retrieval is fine but answers are slow.** Check `ollama ps` shows
`100% GPU`. On a CPU, `eval-answers` can take many minutes.

**Scores changed after editing the course.** The index is a snapshot.
Re-run `python rag.py index` after any change to the modules.

**Using vLLM from Lab 03 instead.** vLLM serves one chat model and no
embeddings, so keep Ollama for embeddings. Pointing generation at
another server would mean separate clients for chat and embeddings in
`rag.py`, which is a good extension exercise.

---

[← Lab 03: Serve like a product](../03-serve-like-a-product/README.md) ·
[Labs home](../README.md) ·
[Next: Lab 05, LoRA fine-tune →](../05-lora-fine-tune/README.md)
