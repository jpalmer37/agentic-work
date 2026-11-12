# 11. Evaluation

## Why this matters

With traditional software, a test either passes or fails. With a model,
output is fuzzy, varied and sometimes wrong in subtle ways. Teams
without a way to measure quality change prompts by feel, ship
regressions without noticing, and argue from anecdotes. Teams with good
evaluation move faster than everyone else, because every change gets an
answer: better or worse?

## The idea in plain words

An **eval** is a test suite for model behaviour. It has three parts:

1. A fixed set of **inputs** that represent real use.
2. A way to **score** each output.
3. A **number** you track over time.

With those in place, changing a prompt, swapping a model or adjusting
retrieval stops being a guess.

## Why evaluating language models is hard

- **Many right answers.** A good summary can be worded a thousand ways.
- **Open-ended output.** There is nothing to compare character by
  character.
- **Randomness.** The same input can give different outputs.
- **Subjective quality.** "Helpful" and "clear" are judgements.

None of this makes evaluation impossible. It means you need more than
one kind of scoring.

## Step one: a golden set

Collect 50 to 200 examples that look like real usage. For each, record
the input and either the expected answer or the criteria a good answer
must meet.

Where to find them:

- Real user requests, from logs or interviews.
- Cases that have gone wrong before.
- Deliberately hard and unusual cases.
- Inputs the system should refuse or redirect.

Treat the set as code: keep it in version control, review changes, and
**add a new case every time you find a failure in production.** Over
time it becomes your most valuable asset. Models and prompts come and
go. The eval set is what tells you whether the new one is better.

Fifty careful examples beat five thousand careless ones. Start small.

## Three ways to score

### 1. Checks written in code

The cheapest and most reliable. Use them wherever the task allows.

- Exact or normalized match with the expected answer.
- Valid JSON that fits the schema.
- Contains, or does not contain, a required string.
- A number within tolerance.
- Generated code passes its unit tests.
- A cited source really appears in the retrieved documents.

### 2. A model as the judge

For qualities code cannot check, such as tone, relevance or
faithfulness, ask a strong model to grade the output against a written
rubric.

```text
You are grading an answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Does the answer state the same facts
as the reference, with no
contradictions?

Reply with exactly PASS or FAIL,
then one sentence explaining why.
```

What makes a judge trustworthy:

- **Ask narrow questions.** "Is every claim supported by the source?"
  works better than "Is this good?"
- **Prefer pass/fail** or a short scale over scores out of ten.
- **Give it the reference answer** or source when one exists.
- **Check the judge against people.** Have humans grade 50 to 100
  examples, and measure how often the judge agrees. If it does not
  agree with your experts, fix the rubric before trusting it.

Judges have known biases: they can favour longer answers, the first
option shown, or text that resembles their own style. Validating
against human labels is how you catch this.

### 3. People

Human review is the reference standard and the most expensive. Use it
where it counts:

- To create the labels that calibrate your automated judges.
- For regular audits of a sample of production traffic.
- For high-stakes decisions before a launch.

## What to measure, by kind of system

| System | Measure |
|---|---|
| Classification, extraction | Accuracy, precision, recall |
| Retrieval | Is the right chunk in the top k? |
| RAG answers | Faithful to sources? Answers the question? |
| Agents | Task completed? Steps taken? Cost? |
| Everything | Latency, cost, refusal rate |

For retrieval systems, score the two halves separately, as
[Module 9](09-adapting-models.md) explains. A single end-to-end score
cannot tell you which half broke.

## How many examples do you need?

An eval score is an estimate, and estimates have error bars. For a
pass rate `p` measured on `n` examples:

```text
standard error = sqrt(p * (1 - p) / n)
95% range      = p +/- 1.96 * SE
```

At a pass rate around 85%:

| Examples | 95% range |
|---|---|
| 100 | about +/- 7 points |
| 200 | about +/- 5 points |
| 1,000 | about +/- 2 points |
| 5,000 | about +/- 1 point |

So on 200 examples, moving from 85% to 87% tells you **nothing**. The
difference is well inside the noise. Many "improvements" that teams
celebrate are this kind of mirage.

Two ways to see more with fewer examples:

- **Compare on the same examples.** Run both versions on the identical
  set and look only at the cases where they disagree. If B fixes 12
  cases and breaks 2, that is far more convincing than two overall
  percentages.
- **Reduce randomness.** Evaluate at temperature 0, or run each case
  several times and average.

## A minimal eval harness

You do not need a framework to begin.

```python
import math

def run_eval(system, cases, grade):
    passed = 0
    for case in cases:
        output = system(case["input"])
        passed += grade(output, case)
    n = len(cases)
    p = passed / n
    se = math.sqrt(p * (1 - p) / n)
    return p, 1.96 * se

cases = [
    {"input": "capital of France",
     "expected": "Paris"},
    {"input": "capital of Japan",
     "expected": "Tokyo"},
    {"input": "capital of Canada",
     "expected": "Ottawa"},
    {"input": "capital of Peru",
     "expected": "Lima"},
]

# stand-in for a call to a model
canned = {
    "capital of France": "Paris",
    "capital of Japan": "Tokyo",
    "capital of Canada": "Toronto",
    "capital of Peru": " lima ",
}

def system(text):
    return canned[text]

def grade(output, case):
    got = output.strip().lower()
    want = case["expected"].lower()
    return got == want

p, m = run_eval(system, cases, grade)
print(f"{p:.0%} +/- {m:.0%}")
# 75% +/- 42%
```

Replace `system` with a call to your model and `grade` with your own
check. That is a working eval. Everything else is refinement.

Look at the printed margin: 75% plus or minus 42 points. Four examples
tell you almost nothing, which is exactly the point of the previous
section.

## Offline and online evaluation

**Offline:** run the golden set before deploying. It is fast, cheap and
repeatable. It tells you what will *probably* happen.

**Online:** measure real usage after deploying. It tells you what
*actually* happened.

Online signals worth collecting:

- Explicit feedback: thumbs up or down, ratings.
- Implicit feedback: did the user rephrase and retry, copy the answer,
  or abandon the session?
- Outcomes: was the ticket resolved, or escalated to a person?
- A/B tests between versions
  ([Module 10](10-classical-ml-and-data-science.md)).

You need both. Offline evals stop bad changes before users see them.
Online metrics reveal what your golden set failed to anticipate, and
those surprises become new golden examples.

## Regression testing

Treat prompts, model versions and retrieval settings as code. Any change
to them should run the eval suite automatically, and a drop below a
threshold should block the release.

This matters most when a model provider updates a model, or you
upgrade one yourself. The new version may be better on average and
worse on the particular thing your product depends on. Only your own
eval will tell you.

## Public benchmarks

Leaderboards are useful for drawing up a shortlist. They cannot tell
you how a model will do on your task.

- **They measure someone else's problem.** A model that tops a coding
  benchmark may still be mediocre at your support tickets.
- **Contamination.** Benchmark questions leak into training data, so
  scores can reflect memory rather than ability.
- **Goodhart's law.** When a measure becomes a target, it stops being a
  good measure. Models get tuned toward popular benchmarks.

Use benchmarks to choose three candidates. Use your own eval to choose
one.

## Error analysis: the highest-value hour

Numbers tell you *that* something is wrong. Reading tells you *what*.

1. Pull 50 failing examples.
2. Read each one. Write a short note on what went wrong.
3. Group the notes into categories.
4. Count them.
5. Fix the biggest category first.

This simple routine consistently beats clever tooling. Most teams find
that two or three categories account for the bulk of their failures,
and that they had been working on the wrong problem.

## Common mistakes

- **No eval at all.** Changing prompts based on three examples you
  tried by hand.
- **Testing on your few-shot examples.** If an example is in the
  prompt, it cannot be in the eval.
- **Trusting an unvalidated judge.** Measure its agreement with people.
- **Chasing differences inside the noise.**
- **One number for everything.** An average can rise while an important
  group of users gets worse. Break results down by category.
- **Editing the eval set and comparing with old scores.** When the set
  changes, re-run the old system for a fair comparison.
- **Optimizing the proxy.** Raising a judge's score is not the goal.
  Helping users is.

## Check yourself

<details>
<summary>1. What are the three parts of an eval?</summary>

A fixed set of representative inputs, a method for scoring outputs, and
a tracked number.
</details>

<details>
<summary>2. Your eval has 100 examples. A prompt change moves the pass
rate from 80% to 84%. Is that real?</summary>

You cannot say. On 100 examples the 95% range is roughly plus or minus 8
points. Compare the two versions case by case, or grow the eval set.
</details>

<details>
<summary>3. How do you know whether to trust an LLM judge?</summary>

Have people grade a sample, run the judge on the same sample, and
measure agreement. If it disagrees with your experts, improve the rubric
until it does not.
</details>

<details>
<summary>4. A new model tops a public leaderboard. Should you switch to
it?</summary>

Not on that basis. Run it through your own eval. Leaderboards measure
different tasks and can be inflated by contamination.
</details>

<details>
<summary>5. Why keep both offline and online evaluation?</summary>

Offline evaluation is fast and catches regressions before release, but
only covers what you thought to include. Online evaluation shows real
behaviour and surfaces cases you missed.
</details>

<details>
<summary>6. Your overall score went up but users in one region are
complaining. What did you miss?</summary>

A breakdown by segment. Averages can hide a group that got worse.
Report scores by category, language, customer type and so on.
</details>

> **Try it.** [Lab 04](labs/04-rag-over-your-docs/README.md) and [Lab 05](labs/05-lora-fine-tune/README.md) both report scores with error bars and compare runs case by case.

## Takeaways

- An eval is inputs, a scoring method, and a number you track.
- Start with 50 to 200 real examples and grow the set from failures.
- Score with code where you can, a validated judge where you cannot,
  and people to keep both honest.
- Small evals have wide error bars. Compare case by case.
- Run evals on every change, like tests.
- Read your failures. It is the best use of an hour.

---

[← Classical ML and data science](10-classical-ml-and-data-science.md) ·
[Course home](README.md) ·
[Next: MLOps →](12-mlops.md)
