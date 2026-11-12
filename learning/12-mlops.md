# 12. MLOps

## Why this matters

A model in a notebook helps nobody. Most machine learning projects that
fail do not fail at modelling. They fail on the road to production, or
they succeed at launch and then quietly rot. MLOps is the set of
practices that gets a model into service and keeps it working.

If you already build software or data pipelines, you know most of this.
The interesting part is what is different.

## The idea in plain words

Ordinary software is built from **code**. A machine learning system is
built from **code, data and a trained model**, and all three change.

MLOps is DevOps extended to cover the two extra moving parts. The goals
are the familiar ones:

- **Reproducible:** you can rebuild any past result.
- **Automated:** releases do not depend on someone's laptop.
- **Observable:** you know when it stops working.

## What makes ML different from normal software

**Behaviour comes from data.** Two runs of the same code on different
data produce different products. Reviewing the code is not enough.

**It fails silently.** A broken service throws errors. A broken model
returns confident, well-formed, wrong answers with a 200 status code.

**It decays.** The world changes: customers, prices, language, fraud
tactics. A model frozen in time gets steadily worse without any change
to the code.

**Tests are statistical.** You cannot assert an exact output. You
assert that quality stays above a threshold.

**Artifacts are large.** Datasets and weights are gigabytes and do not
belong in git.

## The lifecycle

```text
 +-> data -> train -> evaluate -+
 |                             |
 |                          register
 |                             |
retrain <- monitor <- serve <- deploy
```

It is a loop, not a line. Monitoring feeds back into data and
retraining. A team that builds only the top half has a demo. The bottom
half is what makes it a product.

## Version everything

To reproduce a model you need every ingredient that produced it.

| Ingredient | How to pin it |
|---|---|
| Code | Git commit |
| Data | Immutable snapshot or content hash |
| Configuration | A file in git, not notebook cells |
| Environment | Container image, locked dependencies |
| Random seed | Recorded with the run |
| Model weights | Stored with a link to all of the above |

For LLM applications, add:

| Ingredient | How to pin it |
|---|---|
| Prompts | In git, reviewed like code |
| Model | Exact version identifier, never "latest" |
| Retrieval index | Version and build date |
| Tool definitions | In git |

**The test:** could you rebuild the model that was in production three
months ago, and explain why it made a particular prediction? If not,
start here.

Data versioning does not need special tooling to begin. Writing each
dataset to a dated, never-modified path and recording that path with
the run is a solid first step. Dedicated tools, such as DVC, add
convenience later.

## Experiment tracking

Every training run should record automatically:

- The settings used.
- The code commit and data version.
- Metrics over time.
- The resulting artifacts.

Without this, you will find yourself unable to say which of last week's
runs produced the good model. Tools such as MLflow and Weights & Biases
do this well. A disciplined spreadsheet is still far better than
nothing.

## The model registry

A registry is a catalogue of trained models. Each entry has a version,
its lineage (which run, data and code made it), its evaluation results,
and a stage such as *staging* or *production*.

It gives you two important things:

- **Promotion is deliberate.** A model reaches production through a
  recorded decision, not a file copy.
- **Rollback is trivial.** Point production back at the previous
  version.

## Pipelines

Turn the steps you ran by hand into an automated sequence:

```text
ingest -> validate -> build features
       -> train -> evaluate -> register
```

Each step has declared inputs and outputs, runs in a defined
environment, and can be rerun on its own. If you have built data
workflows with an orchestrator before, this is the same discipline
applied to training. The same properties matter: steps that can be
safely repeated, pinned environments, and results cached by their
inputs.

### Validate data before training

Bad data produces a bad model with no error message. Check the data at
the door:

- Schema: the expected columns and types.
- Completeness: row counts and missing-value rates.
- Ranges: no negative ages, no dates in the future.
- Distribution: has a column shifted sharply since last time?

Fail the pipeline loudly when a check fails. It is much cheaper than
discovering the problem in production.

### Training and serving must agree

A classic silent bug: a feature is computed one way in the training
pipeline and a slightly different way in the live service. The model
receives inputs unlike anything it learned from, and quality drops with
no error anywhere.

The cure is to have **one implementation** of each feature, used by
both training and serving. A "feature store" is a system built around
that idea.

## CI/CD for models

Continuous integration for ML has more gates than for ordinary code.

```text
on every change:
  1. unit tests
     does the code work?
  2. data checks
     is the data sane?
  3. evaluation suite
     is quality good enough?
  4. performance tests
     are latency and memory OK?
  5. package and deploy
     progressively
```

Gate 3 is the one that is new. The candidate must:

- Beat or match the current production model on the golden set.
- Not regress on any important category of input.
- Pass any safety or policy checks.

[Module 11](11-evaluation.md) covers how to build that suite.

## Deployment patterns

**How predictions are served:**

| Pattern | When to use |
|---|---|
| Batch | Score everything nightly. Simplest. |
| Online | An API answering requests in real time |
| Streaming | React to events as they arrive |
| On-device | Run on a phone or at the edge |

If the business can live with predictions computed in advance, choose
batch. It removes latency requirements, scaling and most on-call
burden.

**How a new version is released:**

| Strategy | How it works |
|---|---|
| Shadow | New model sees real traffic. Output is logged, not used |
| Canary | 1% of traffic, then 10%, then all, watching metrics |
| A/B test | Split users and compare outcomes |
| Blue-green | Run both, switch all traffic, keep the old one ready |

Shadow deployment is especially valuable for models: you learn how the
new version behaves on live inputs with zero risk to users.

Always have a rollback you have actually tried.

## Monitoring

Monitor in four layers, from the machine up to the business.

**1. Service health.** The usual: latency at the 50th, 95th and 99th
percentiles, error rate, throughput, and for GPU services, memory use
and queue length.

**2. Input data.** Are today's inputs like the ones the model was
trained on? Watch schema, missing rates and distributions.

**3. Model outputs.** Has the mix of predictions shifted? If a fraud
model that used to flag 1% suddenly flags 5%, something has changed
even if you cannot yet say what. For LLMs: output length, refusal rate,
tool-call error rate, share of responses failing validation.

**4. Outcomes.** The real measure: accuracy once true labels arrive,
user feedback, task completion, business metrics. These often arrive
late, which is why layers 2 and 3 exist as early warnings.

## Drift

**Data drift:** the inputs change. A new customer segment arrives. A
sensor is recalibrated. An upstream team renames a category.

**Concept drift:** the relationship between inputs and the right answer
changes. Fraudsters adapt. What counted as a normal purchase last year
is unusual now.

Detect drift by comparing a recent window of data with a reference
window from training time. Respond in order: alert, investigate, and
retrain if the shift is real. Many apparent drifts turn out to be
upstream data bugs, so investigate before retraining.

## Retraining

- **On a schedule:** weekly or monthly. Simple and predictable.
- **On a trigger:** when drift or a quality drop is detected.

Either way, a retrained model goes through the same evaluation gates as
any other release. Automatic retraining without automatic evaluation is
a way to ship problems faster.

## LLMOps: what changes with language models

When you build on a pretrained LLM, you usually train nothing. The
"model" you operate is the combination of:

```text
prompt
+ model version
+ retrieval
+ tools
+ settings
```

The principles are identical. The emphasis shifts.

- **Prompts are code.** Version, review and test them.
- **Pin model versions.** Providers update and retire models. Your eval
  suite is how you qualify a replacement.
- **Trace every request.** Record the prompt, retrieved documents, tool
  calls, output, token counts, latency and cost. When an answer is
  wrong, the trace shows which stage failed.
- **Evals are the test suite.** Run them on every change.
- **Watch cost as a first-class metric.** Track tokens by feature and
  by customer. A prompt change can double your bill.
- **Close the loop.** Production failures become eval cases, which
  drive the next improvement.

Tracing and evaluation tools for LLM applications are a fast-moving
area. Langfuse, LangSmith, Arize Phoenix, and the tracing features of
MLflow and Weights & Biases were common choices when this was written.
Many build on the OpenTelemetry standard, which keeps you portable.

## Governance

In regulated settings such as health, finance and the public sector,
you also need to be able to answer questions from outside the team.

- **Documentation:** a model card stating what the model is for, what
  it was trained on, how it was evaluated and its known limits.
- **Lineage:** which data produced which model.
- **Access control:** who can train, deploy and query.
- **Audit logs:** who changed what, and when.
- **Privacy:** what personal data is in training sets, prompts and
  logs, and how long it is kept.
- **Fairness:** evaluation broken down by relevant groups.

It is far easier to build these in from the start than to reconstruct
them for an auditor.

## Start small

You do not need a platform team to begin. Four habits deliver most of
the value:

1. **Pin your data** to an immutable version for every run.
2. **Track every run** with its settings, code and metrics.
3. **Gate releases** on an evaluation suite.
4. **Monitor outputs** and review a sample of them every week.

Add automation as the pain justifies it.

| Level | What it looks like |
|---|---|
| 0 | Notebooks, manual deploys |
| 1 | Automated, reproducible training pipeline |
| 2 | Automated testing, deployment and monitoring |

Most teams should aim to reach level 1 quickly and grow into level 2.

## Common mistakes

- **Treating launch as the finish line.** Launch is where the work of
  keeping it correct begins.
- **Monitoring only uptime.** The service can be perfectly healthy while
  the model is perfectly wrong.
- **Using "latest" for a model version.** Behaviour changes underneath
  you with no deploy on your side.
- **Unversioned prompts** edited directly in production.
- **Retraining on autopilot** with no evaluation gate.
- **Building the platform before the first model.** Ship one model with
  simple tools, and automate what hurt.

## Check yourself

<details>
<summary>1. Name three ways an ML system differs from ordinary software
from an operations point of view.</summary>

Its behaviour depends on data as well as code. It fails silently by
returning plausible wrong answers. It decays over time as the world
changes. Also: its tests are statistical, and its artifacts are large.
</details>

<details>
<summary>2. What must you record to be able to reproduce a
model?</summary>

The code commit, the exact data version, the configuration, the
environment, the random seed, and the resulting weights, all linked
together.
</details>

<details>
<summary>3. What is training/serving skew, and how do you prevent
it?</summary>

A feature being computed differently in training and in production, so
the model sees unfamiliar inputs. Prevent it by using one shared
implementation of each feature in both places.
</details>

<details>
<summary>4. Your service's latency and error rate look perfect, but
users say answers got worse. Which monitoring layers were
missing?</summary>

Input data, model outputs and outcomes. Service health only tells you
the system is running, not that it is right.
</details>

<details>
<summary>5. What is a shadow deployment good for?</summary>

Running a new model on real production inputs without using its output.
You see how it behaves and compare it with the current model at no risk
to users.
</details>

<details>
<summary>6. In an LLM application with no training, what are you
versioning?</summary>

The prompts, the exact model version, the retrieval index, the tool
definitions, and settings such as temperature. Together they are the
"model" you operate.
</details>

## Takeaways

- MLOps is DevOps plus two more moving parts: data and models.
- Models fail silently and decay, so monitoring must go beyond uptime.
- Version code, data, config, environment and weights together.
- Gate every release on an evaluation suite.
- Prefer batch serving when you can, and release with shadow or canary.
- For LLM apps: version prompts, pin models, trace requests, watch
  cost.

---

[← Evaluation](11-evaluation.md) ·
[Course home](README.md) ·
[Next: Hosting your own models →](13-hosting-your-own-models.md)
