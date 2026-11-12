# 10. Classical ML and data science

## Why this matters

Not every problem is a language model problem. Most company data lives
in tables, and for tables, older methods are usually more accurate,
far cheaper and easier to explain. And the discipline of honest
measurement that data science teaches applies to everything, including
every LLM project you will ever ship.

## The idea in plain words

Use the simplest model that solves the problem. Spend the effort you
save on two things that matter more than model choice: **the quality of
your data** and **the honesty of your evaluation**.

## Which tool for which data

| Your data | Reach for first |
|---|---|
| Tables (rows and columns) | Gradient-boosted trees |
| Text, images, audio | A pretrained deep model |
| Very little data | Linear or logistic regression |
| You must explain each decision | Linear model or small tree |

Deep learning wins when the raw input has structure a network can
learn to read: pixels, sounds, words. On a table of customer attributes
there is little such structure, and trees win.

## Always start with a baseline

Before any real model, measure something trivially simple:

- Predict the average, or the most common class.
- A hand-written rule ("flag orders over 1,000").
- Logistic regression on a few obvious columns.

The baseline tells you what "good" means. If your sophisticated model
barely beats "always predict the average", either there is a bug or
the data has little signal. Both are worth knowing early.

## Underfitting and overfitting

Every model sits somewhere between two failures.

```text
error
 |\                         /
 | \   validation          /
 |  \                   _.'
 |   '-.__          __.-'
 |        ''------''
 |  '-._
 |      '--.__   training
 |            '''---......
 +---------------------------->
   too simple        too complex
   (underfit)        (overfit)
```

**Underfitting:** the model is too simple to capture the pattern. It
does poorly on training data and on new data.

**Overfitting:** the model has memorized the noise in its training
data. It does very well on training data and poorly on new data.

More complexity lowers training error forever. Validation error falls,
then rises. You want the bottom of that curve.

## Linear and logistic regression

**Linear regression** predicts a number as a weighted sum of the
inputs: a dot product plus a constant.

```text
prediction = w · x + b
```

**Logistic regression** does the same, then squeezes the result into a
probability between 0 and 1 with a sigmoid.

Both are a single neuron
([Module 1](01-vectors-and-dot-products.md)). They are fast, hard to
overfit, and readable: each weight tells you how much a feature pushes
the prediction, and in which direction. When you need to explain a
decision to a regulator or a customer, that matters.

## Decision trees

A tree is a learned chain of yes/no questions:

```text
          income > 50k?
          /          \
        no            yes
        |              |
   age > 30?      owns home?
    /     \        /      \
  deny  review  approve  review
```

At each step it picks the question that best separates the outcomes. A
single tree is easy to read but overfits badly once it grows deep: it
keeps splitting until it has memorized individual rows.

## Ensembles: many weak models, one strong one

**Random forest.** Train hundreds of trees, each on a random sample of
the rows and columns, and average them. Individual trees overfit in
different ways, and the errors cancel out.

**Gradient boosting.** Train trees one after another. Each new tree is
trained to predict **the errors the ensemble has made so far**.

```text
prediction = tree_1(x)
           + tree_2(x)
           + tree_3(x)
           + ...

tree_2 fixes tree_1's errors.
tree_3 fixes what is still left.
```

Those "errors" are precisely the gradient of the loss
([Module 2](02-loss-functions.md)). Boosting is gradient descent where
each step adds a tree instead of nudging a weight.

Gradient-boosted trees, in libraries such as XGBoost, LightGBM and
CatBoost, are the standard for tabular data.

### Why boosted trees win on tables

- They handle numbers and categories together.
- They do not care about scale. No normalization needed.
- They cope with missing values.
- They capture sharp thresholds and interactions naturally.
- They work with thousands of rows, not millions.
- They train in seconds or minutes on a CPU.
- They need little tuning to be good.

```python
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import (
    cross_val_score,
)

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
)
scores = cross_val_score(
    model, X, y,
    cv=5, scoring="roc_auc",
)
print(scores.mean(), scores.std())
```

## Feature engineering

For tabular problems, the columns you give the model matter more than
the model. Useful patterns:

- **Aggregates:** a customer's average order value, count in 30 days.
- **Ratios:** spend divided by income.
- **Time since:** days since last login.
- **Date parts:** day of week, month, is it a holiday.

A good feature encodes knowledge about the problem that the model would
struggle to work out alone.

## Leakage: the silent killer

**Leakage** is when information that would not be available at
prediction time sneaks into training. The model looks brilliant in
testing and fails in production. It is the most common cause of results
that are too good to be true.

**Target leakage.** A feature that is a *consequence* of the thing you
are predicting. Predicting whether a customer will leave, using a column
`account_closed_date`. Of course it predicts perfectly.

**Contamination between train and test.** The same customer appears in
both sets. Or you computed the average for normalization over all the
data before splitting, so test statistics leaked into training.

**Time leakage.** Splitting rows at random when the data is ordered in
time, so the model trains on next week to predict last week.

Two rules prevent most of it:

1. For every feature, ask: **"at the moment of prediction, would I
   really have this value?"**
2. Split the way production works. Predicting the future? Split by
   time. Predicting for new users? Split by user.

If a model scores far above what seems plausible, assume leakage until
you have ruled it out.

## Splitting data honestly

**Train / validation / test.** Train on one part. Make all your choices
(features, settings, model type) on validation. Report from test,
once, at the end.

**Cross-validation.** With limited data, a single validation split is
noisy. Split the data into five parts, train five times with a
different part held out each time, and average. You get a steadier
estimate and a sense of its spread.

**Grouped and time-based splits.** Keep all rows for one user on the
same side. For time-ordered data, always validate on a later period
than you trained on.

## Choosing the right metric

Take a fraud problem: 1,000 transactions, 10 of them fraudulent. Your
model flags 20, and 8 of those really are fraud.

```text
precision = 8 / 20 = 40%
  of those flagged, how many are real?

recall = 8 / 10 = 80%
  of the real ones, how many caught?

accuracy = 986 / 1000 = 98.6%
```

Accuracy looks superb. But a model that flags **nothing** scores 99.0%
accuracy while catching no fraud at all. When classes are unbalanced,
accuracy is nearly useless.

**Precision and recall pull against each other.** A model outputs a
score, and you choose the threshold. Lower it to catch more fraud
(higher recall) and you raise more false alarms (lower precision).

Pick based on what each mistake costs:

- Missing a case is costly (disease screening): favour recall.
- A false alarm is costly (blocking a real customer): favour precision.

Summary scores when you need one number:

- **F1:** a balance of precision and recall at one threshold.
- **ROC-AUC:** how well the model ranks positives above negatives,
  across all thresholds.
- **PR-AUC:** similar, and more informative when positives are rare.

## Calibration

A model is **calibrated** if its probabilities mean what they say: of
all the cases where it says 70%, about 70% turn out positive.

A model can rank cases well and still be badly calibrated. This matters
whenever the probability itself drives a decision, such as pricing risk
or deciding when to hand a case to a human. Check it by grouping
predictions into buckets and comparing predicted with actual rates.

## A/B testing

Offline metrics predict. Only an experiment on real users confirms.

An **A/B test** randomly assigns users to the current version (A) or
the new one (B) and compares an outcome. Random assignment is what lets
you say the change *caused* the difference.

The hard part is noise. Suppose A converts 100 of 1,000 users (10%)
and B converts 120 of 1,000 (12%). A 20% relative improvement. Ship it?

```python
import math

def ab_test(ca, na, cb, nb):
    pa, pb = ca / na, cb / nb
    pool = (ca + cb) / (na + nb)
    var = pool * (1 - pool)
    se = math.sqrt(var * (1/na + 1/nb))
    z = (pb - pa) / se
    tail = abs(z) / math.sqrt(2)
    p = math.erfc(tail)
    return round(z, 2), round(p, 3)

# conversions and users: A, then B
print(ab_test(100, 1000, 120, 1000))
# (1.43, 0.153)
```

A p-value of 0.15 means a gap this large would show up about 15% of
the time **even if B were no better than A**. That is not strong
evidence. To detect a real move from 10% to 12% reliably, you need
roughly 3,800 users in each group, not 1,000.

Rules for trustworthy experiments:

- **Decide the sample size before you start.**
- **Do not stop early** because the result looks good today. Checking
  repeatedly and stopping at the first "significant" reading produces
  false wins.
- **Name one primary metric** in advance.
- **Watch guardrail metrics** such as latency, errors and complaints,
  so a win on one number does not hide a loss on another.

## Statistical traps

**Correlation is not causation.** Ice cream sales and drownings rise
together. Summer causes both. A hidden common cause is called a
**confounder**. Randomized experiments are the clean way around it.

**Multiple comparisons.** Test 20 metrics and, by chance alone, one
will look "significant". The more you look, the more you will find.

**Simpson's paradox.** A trend that holds within every group can
reverse when the groups are combined, because the group sizes differ.
Always check the breakdown.

**Selection bias.** If you only analyse customers who stayed, you learn
nothing about why others left.

**Regression to the mean.** Extreme results tend to be followed by more
ordinary ones. The worst-performing store last month will probably
improve next month whatever you do.

## Common mistakes

- **Reaching for deep learning on a table.** Try boosted trees first.
- **Skipping the baseline.** You will not know if your model is good.
- **Splitting at random when data has time or groups.** The test score
  will be inflated.
- **Preprocessing before splitting.** Fit scalers and encoders on
  training data only.
- **Reporting accuracy on unbalanced data.**
- **Calling an experiment early.**

## Check yourself

<details>
<summary>1. You are predicting loan default from a table of 50,000
applicants. What model do you try first, and what do you compare it
with?</summary>

Gradient-boosted trees, compared against a simple baseline such as
logistic regression or always predicting the most common outcome.
</details>

<details>
<summary>2. Your churn model reaches 99.8% validation accuracy on the
first attempt. What do you suspect?</summary>

Leakage. Look for a feature that is a consequence of churning, for the
same customers on both sides of the split, or for a random split on
time-ordered data.
</details>

<details>
<summary>3. Only 1% of emails are phishing. A model has 99% accuracy.
Is it good?</summary>

Unknown. A model that never flags anything also scores 99%. Look at
precision and recall, or PR-AUC.
</details>

<details>
<summary>4. How is gradient boosting related to gradient
descent?</summary>

Each new tree is fitted to the gradient of the loss with respect to the
current predictions, which is the remaining error. Adding the tree is a
step downhill, just as a weight update is in a neural network.
</details>

<details>
<summary>5. Your A/B test shows a 5% lift after two days with p = 0.04,
and you had planned to run for two weeks. Do you stop?</summary>

No. Stopping the first time a result crosses the threshold greatly
increases false positives. Run for the planned duration.
</details>

## Takeaways

- For tables, start with gradient-boosted trees and a simple baseline.
- Validation error is U-shaped: too simple underfits, too complex
  overfits.
- Leakage is the usual reason a result looks too good.
- Split the data the way production will use the model.
- Pick the metric from the cost of each kind of mistake.
- Trust experiments over offline numbers, and plan their size first.

---

[← Adapting models](09-adapting-models.md) ·
[Course home](README.md) ·
[Next: Evaluation →](11-evaluation.md)
