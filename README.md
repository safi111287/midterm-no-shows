---
title: Maple Physio No-Show Check
emoji: 📅
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Appointment risk and confirmation-call decisions
---

# Maple Physio no-show check

## What it does

Maple Physio wants to reduce the cost of missed physiotherapy appointments.
A confirmation call costs $8. An uncalled no-show costs $35. Under the assignment's
simplifying assumption, a called patient attends, so a called booking costs only $8.
The intended page estimates no-show probability and recommends whether to call.
It supports reception staff judgment and does not diagnose patients.

**Development status:** The training foundation, saved model, reports, and core
tests are complete. Appointment, NoShowPredictor, the Gradio page, and their
assigned tests are implementation skeletons. This repository is not yet a
completed or deployed submission. No public Space or demo link exists yet.

## How to run it locally

Python 3.12 is used both locally and in the Space settings. Create the environment
from the repository root:

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
python -m noshows.train
```

Train from this environment, not from an older lab environment. If the environment
already exists, activate it; do not recreate it. After dependency changes, run:

```bash
conda env update -f environment.yml --prune
```

The training command recreates the complete pipeline, model metadata, CV table,
grid-search results, threshold costs, and held-out test report. It does not need
the unfinished app or prediction classes.

After the assigned classes and page have been implemented:

```bash
gradio app.py
```

If PowerShell cannot activate conda or cannot find `gradio`, use:

```bash
conda run --no-capture-output -n midterm-no-shows gradio app.py
```

The environment file installs `requirements-dev.txt`, which includes the runtime
requirements. No editable package installation is necessary with this flat layout.

## Team

The following are assigned responsibilities; unfinished tasks are not claimed as completed work.

| Member | Responsibilities | Current state |
|---|---|---|
| Safaa Alhawasli | Environment, shared interfaces, data integration, preprocessing, evaluation, training, generated model/reports, core tests, integration and deployment coordination | Training foundation implemented; final integration pending |
| Md Kawsarul Islam | Appointment, NoShowPredictor, their tests, and class-design/validation report text | Implementation pending |
| Art (full name pending) | Gradio page, callback tests, live valid/invalid/missing-input checks, and interface report text | Implementation pending |
| Rafael (full name pending) | Deployment tests, setup verification, fresh-clone checks, and local-run documentation review | Implementation pending |

Every member must commit from their own GitHub account, make at least three
meaningful commits, review another member's work, and speak in the demo.
The group must have at least fifteen meaningful commits. See `docs/TEAM_HANDOFF.md`.

## Decisions

The generated table contains 4,000 appointments. Only information available when
the booking is entered is used for prediction. Numeric and category lists and
input rules are defined once in `noshows/pipeline.py` and exposed through the
Appointment module for the page.

| Field | Use and accepted values | Reason |
|---|---|---|
| appointment_id | Excluded | Identifies a row; it does not describe an appointment's risk. |
| lead_days | Numeric, whole days 0-120 | Negative booking lead time is impossible; 120 permits a wider but bounded booking horizon than the training range of 0-60. |
| age | Numeric, whole years 12-110 | The assignment bounds the supported patient ages; predictions near the wider limits are extrapolations beyond training ages 16-92. |
| prior_no_shows | Numeric, whole count 0-30 | Past missed visits are known before the appointment; counts cannot be negative or fractional, and the upper limit bounds extrapolation beyond training counts 0-6. |
| hour | Numeric, whole hour 8-17 | Matches the clinic hours represented in the data; appointments outside these hours are unsupported. |
| first_visit | Numeric, 0 or 1 | A binary booking fact; other values and booleans are rejected at the input boundary. |
| distance_km | Numeric, 0.1-150, or missing | Distance must be positive and bounded. The wider allowed range can extrapolate beyond the training range 0.5-60. Missing distance is explicitly accepted because it occurs in training. |
| weekday | mon, tue, wed, thu, fri | These are the exact weekday categories represented by the model; weekends and spelling changes are rejected. |
| booking | online, phone, front_desk | These are the supported booking channels, known before the visit. |
| insurance | private, provincial, self_pay | These are the represented payment/insurance categories; other values are unsupported. This feature may also create fairness concerns. |
| treatment | physio, massage, acupuncture | These are the represented treatment types, chosen at booking. |
| fee_charged | Excluded | Recorded after the visit; it partly reveals the outcome and causes target leakage. |
| no_show | Target only | The future outcome to predict, never an input feature. |

The whole-number requirement applies to counts, whole-day/hour inputs, and the
binary flag. Only distance permits a gap. The Appointment contract accepts None,
blank text, or NaN for distance and converts it to NaN; infinity is invalid.
It must emit a missing-distance warning. The pipeline substitutes the median learned
from the training partition/fold. No median or scaler is fitted on the test set.

`OneHotEncoder(handle_unknown="ignore")` is a technical safeguard, not permission
to accept misspellings. An unseen value is encoded as zeros for that feature, which
can change the probability without raising an error. Appointment must therefore
reject unrecognized category values and name the offending field.

For a fixed example with lead_days=7, age=46, prior_no_shows=0, hour=11,
first_visit=0, distance_km=5, booking=phone, insurance=private, treatment=physio:

| weekday spelling | Raw saved-pipeline no-show probability | Intended input handling |
|---|---:|---|
| mon | 14.5708% | Accepted |
| Mon | 11.9625% | Must be rejected by Appointment |

This is a raw-pipeline diagnostic generated by training, not a claim that the
unfinished app already validates these inputs. It does not change the model.

The cost function is:

```text
total cost = 8 × number called + 35 × actual no-shows not called
```

A worked test example has outcomes `[1, 0, 1, 1, 0]` and call decisions
`[1, 1, 0, 0, 0]`: two calls cost $16 and two uncalled no-shows cost $70, totaling $86.

Thresholds from 0.05 to 0.95 in steps of 0.01 are evaluated on training out-of-fold
probabilities. The selected threshold is **0.25**. Equal probability
means call: `probability >= saved threshold`. Equal-cost thresholds prefer fewer
calls, then the larger threshold. Runtime prediction must read the threshold from
`models/model_info.json`; it must not contain a typed-in decision threshold.
The expected-cost break-even point $8/$35 is about 0.229 for perfectly calibrated
probabilities, but the implemented threshold is selected empirically as required.

## Model comparison

The 75/25 split is stratified with seed 11: 3,000 training rows and 1,000 test rows.
Five-fold stratified cross-validation uses the training partition only. All models
use the same build_pipeline function: median imputation and scaling for numbers,
one-hot encoding for categories, followed by the classifier.

The table reports fold mean ± sample standard deviation. Classification metrics in
this comparison use each estimator's default decision rule, before cost tuning.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| dummy | 0.7383 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| logistic | 0.7580 ± 0.0195 | 0.5918 ± 0.0873 | 0.2293 ± 0.0557 | 0.3299 ± 0.0715 | 0.7273 ± 0.0131 |
| logistic_balanced | 0.6750 ± 0.0096 | 0.4225 ± 0.0117 | 0.6599 ± 0.0342 | 0.5150 ± 0.0169 | 0.7273 ± 0.0134 |
| random_forest | 0.7507 ± 0.0149 | 0.5938 ± 0.1048 | 0.1376 ± 0.0454 | 0.2224 ± 0.0670 | 0.7119 ± 0.0096 |
| hist_gradient_boosting | 0.7460 ± 0.0019 | 0.5291 ± 0.0071 | 0.2662 ± 0.0159 | 0.3540 ± 0.0143 | 0.6920 ± 0.0127 |

Logistic regression has the highest untuned mean training ROC-AUC. The balanced
logistic model has a nearly identical AUC with greater recall under its default
rule. There is no convincing separation between those two AUCs relative to fold
spread. The standard logistic model was selected by the stated AUC ranking,
then its operating threshold was chosen separately for cost.

The majority dummy achieves 73.83% accuracy while recalling no no-shows. This is why
accuracy alone is misleading. The objective is useful risk ranking and lower cost,
not simply predicting the common attendance class.

GridSearchCV uses ROC-AUC to evaluate two parameters with six combinations:
C in [0.1, 1, 10] and fit_intercept in [True, False]. Best settings are
C=1.0 and fit_intercept=False.
Mean CV AUC changes from 0.727299 to 0.727337,
an improvement of only 0.000037, far below the untuned fold
standard deviation of 0.013081. This is not convincing evidence of
an improvement beyond the spread, although the highest-scoring settings are retained.
GridSearchCV's exported standard deviation uses its default population convention;
the comparison table uses sample standard deviation.

Model and parameter selection reuse the training folds. The OOF probabilities
exclude each row from its own model fit, but this is not a fully nested selection
estimate. The untouched held-out test is the final generalization check. The test
set is not used to select a model, tune parameters, or change the threshold.

## Result

On the 1,000 held-out appointments, 262 patients are no-shows. The selected
model and fixed threshold recommend 432 calls.

| Held-out metric | Value |
|---|---:|
| Precision | 0.4144 |
| Recall | 0.6832 |
| F1 | 0.5159 |
| ROC-AUC | 0.7290 |
| Accuracy at the cost threshold | 0.6640 |

Confusion matrix, with actual classes in rows and decisions in columns,
ordered [attend/no-call, no-show/call]: `[[485, 253], [83, 179]]`.
Here call decisions are evaluated against the original no-show outcomes, before
any hypothetical intervention. The cost calculation assumes calls prevent misses.

| Policy | Test cost |
|---|---:|
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| Model recommendation | $6,361 |

The model saves **$2,809** compared with calling nobody and **$1,639** compared
with calling everybody on this fixed test set. It beats both baselines here.

A paired bootstrap with 2,000 resamples gives approximate 95% intervals for the
total savings on a same-size test sample:

- Versus nobody: $2,113.93 to $3,547.12.
- Versus everybody: $1,016.92 to $2,239.00.

These intervals hold the trained model and threshold fixed. They quantify sampling
variation within this synthetic dataset, not training uncertainty, selection
uncertainty, real-clinic distribution shifts, or the realism of the call assumption.
They are not a guarantee of future dollar savings.

The separate leakage experiment uses training CV only:

| Features | Mean CV ROC-AUC | Sample standard deviation |
|---|---:|---:|
| Honest booking features | 0.7273 | 0.0132 |
| Booking features plus fee_charged | 0.9229 | 0.0085 |

The larger leaked score cannot be trusted because fee_charged is only known after
the appointment and partly reveals the outcome. The leaked pipeline is never
saved as the production model.

## Tests

The implemented foundation tests cover reproducible data, excluded fields,
stratified split size and separation, fold reproducibility, training-only
imputation, missing distance, pipeline independence, hand-calculated costs,
call-all/call-none cases, threshold extremes, and invalid probabilities.

During foundation development, run:

```bash
python -m pytest tests/test_pipeline.py tests/test_evaluation.py
```

The other test modules are assigned skeletons that deliberately raise
NotImplementedError until completed. They are not skipped or represented as passing.
The final suite must include validation, immutability, missing-distance warning,
probability/risk ordering, saved-threshold boundaries, one model load per predictor,
interface errors, and deployment agreements including Python 3.12.

After every implementation is complete, all three final commands must pass from
a fresh clone with the environment active:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The instructor's `noshows/data.py` is preserved verbatim and excluded from automatic
Ruff rewriting. All authored project modules are checked and formatted.
No models or generated reports are ignored by git.

## Limitations

- Data are synthetic, not a validated clinical sample; real patients may behave differently.
- Calls are assumed to guarantee attendance, and the $8/$35 costs are fixed assumptions.
- No information about illness, transport disruption, patient contactability,
  call capacity, or actual response to reminders is available.
- Missing distance uses a median and reduces individual information; it must be disclosed.
- Values allowed beyond the training ranges are accepted by the assignment rules
  but may yield unreliable extrapolation.
- Age, insurance, distance, and prior attendance can reflect unequal access to care.
  Group fairness, calibration, and clinical suitability have not been established.
- Predictions support confirmation calls, not denial of appointments or care.
- Saved model/report results must stay consistent with any later retraining and demo.
