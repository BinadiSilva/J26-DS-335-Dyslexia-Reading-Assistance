
# Component 4: Learning Progress Prediction & AI-driven Intervention Planning

**Project:** AI-Based Adaptive Reading Assistance System for Children with Dyslexia  
**Project Code:** J26-DS-335  
**Component:** C4 — Learning Progress Prediction & AI-driven Intervention Planning  
**Branch:** `component4-learning-progress-prediction`

---

## 1. Overview

Component 4 analyzes longitudinal reading-assessment data to understand changes in reading performance, evaluate machine-learning models that estimate the next observed reading score, screen for possible stagnation, and generate preliminary reading-support suggestions.

The component is designed to support the wider adaptive reading assistance system by transforming historical assessment data into structured information that educators can review.

The current implementation uses the LIBS longitudinal reading-assessment dataset for its main prediction and screening workflow. A separate synthetic-data workflow is retained for development compatibility.

## 2. Objectives

- Prepare and validate longitudinal reading-assessment data.
- Analyze changes between consecutive observed reading scores.
- Train and compare reading-progress prediction models.
- Select the prediction model using validation data.
- Evaluate final model performance on held-out participants.
- Screen reading-score trends using preliminary stagnation rules.
- Generate preliminary reading-support suggestions.
- Combine assessment results, historical estimates, screening results, and support suggestions into a report.

## 3. Implemented Features

### 3.1 Data Preparation

The data-preparation service loads and validates the LIBS longitudinal dataset.

Main tasks include:

- Checking required columns and data types.
- Validating reading scores against the documented BAS score range of 0–90.
- Organizing records by participant and assessment time point.
- Preparing previous reading scores and score changes.
- Reporting excluded invalid rows, removing identical duplicate assessments, and rejecting conflicting duplicate scores or mismatched assessment names.

Prepared longitudinal data are stored at:

`datasets/component4/libs_reading_assessments_long.csv`

### 3.2 Reading-Progress Prediction

The prediction service creates supervised learning examples using consecutive observed assessments.

**Input features:**
- Previous observed reading score
- Current observed reading score
- Previous assessment time point
- Current assessment time point

**Target:** The next observed reading score.

The target is the next available observed score, not necessarily a score at a fixed future date. Missing assessment time points can mean that consecutive observations are not consecutive scheduled assessments.

The following approaches are compared:

| Model | Purpose |
|---|---|
| Persistence Baseline | Uses the current score as the estimate of the next score |
| Linear Regression | Provides a linear machine-learning baseline |
| Random Forest Regressor | Models potentially non-linear relationships in the assessment features |

Model selection uses validation MAE among Linear Regression and Random Forest. Persistence is a comparison baseline. Final performance is evaluated on a separate group of held-out participants.

### 3.3 Stagnation Screening

The stagnation-detection service analyzes observed score changes for each participant.

It calculates recent score-change trends and assigns a preliminary category:

- **Low:** Recent scores show progress according to the current rules.
- **Medium:** Recent average change is small and positive.
- **High:** Recent average change is zero or negative.
- **Insufficient Data:** There are not enough observed score changes for screening.

These categories are based on preliminary rules. They are not clinically validated diagnoses or reliable standalone measures of a child's reading needs.

The category uses the mean of the latest up to three observed changes:
`High` for a mean <= 0, `Medium` for 0 < mean < 2, and `Low` for mean >= 2.
At least three assessments are required. No validated clinical cutoff is claimed.

The report now separately shows the latest change and a `recent_decline` flag.
Earlier gains cannot hide this flag, even if the average-based category is Low.
It also reports `recent_change_per_study_year`: total recent change divided by
the summed approximate study-age gaps. The local variable descriptions give
ages 4, 5, 6, and 9 for T1–T4. These rates are descriptive, not individual growth
norms or validated annual thresholds. Exact assessment dates are unavailable.
Missing intermediate time points are flagged and never filled with zero scores.
A score of 90 receives a ceiling-effect explanation.

### 3.4 Intervention Planning

The intervention-planning service generates preliminary reading-support suggestions based on the screening category and available assessment history.

Suggestions can include continuing suitable reading practice, monitoring progress, or seeking educator review when information is insufficient or progress may be limited.

The generated plans are rule-based suggestions. They do not demonstrate that an intervention is effective and must be reviewed by a qualified educator before being used in practice.

| Evidence | Support priority and action |
|---|---|
| Insufficient history | Collect evidence; obtain further assessments before classifying a trend. |
| Latest observed decline, High category, or Medium category | Educator review; inspect assessment conditions and plan supported practice. |
| Negative experimental forecast at a time point seen in training | Educator review; corroborate with further observations before changing support. |
| Low category without those review triggers | Continue and monitor using the existing learning plan. |

A positive forecast never cancels an observed decline. Forecasts at unseen or
unknown training time points do not change priorities. Missing history still
requires evidence collection even when a forecast exists. Numerical scores do
not identify a specific decoding, fluency, or comprehension deficit.

Each plan includes `support_priority`, `forecast_use`, `recommendation_basis`,
`review_required`, and `review_trigger`. The rationale states the actual score
changes and any forecast used. Frequency and duration remain educator decisions.
Activities include educator-selected supported oral reading, constructive
feedback, and recording word-reading errors for educator assessment. These
examples are informed by general literacy guidance in the [IES/WWC foundational
reading guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/21); that source does not
validate this software's mapping, thresholds, or effectiveness for dyslexia.

### 3.5 Combined Progress Reporting

The reporting service combines:

- Latest observed reading scores and assessment time points
- Experimental next-score forecasts from each participant's latest history
- The model used for those estimates
- Recent score-change summaries
- Preliminary stagnation categories
- Suggested reading-support activities
- Recommendation frequency and session-duration guidance

The report is printed in the terminal and returned as a pandas DataFrame.

**Important:** The reporting service generates an experimental next-score forecast from each eligible participant's latest assessment. It predicts the next observed score, which may not correspond to a fixed future date. This forecast has not been independently validated as a real-world future prediction.

Each report row includes `forecast_status` and `forecast_explanation`:

| Status | Meaning |
|---|---|
| `observed_training_time_point` | The latest assessment time point occurred in the final model's training inputs. The forecast remains experimental. |
| `unseen_training_time_point` | The latest assessment time point did not occur in training inputs; historical test results do not validate forecasting from it. |
| `training_time_points_unknown` | Training time-point metadata is unavailable, so support cannot be checked. |
| `no_forecast` | No numerical forecast is available; check assessment history and model availability. |

The check uses only the final model's training data, not the held-out test data.
It checks time-point membership, not the full feature distribution, the time
until the next assessment, or whether this learner's forecast is accurate.
`has_future_forecast` indicates that a numerical estimate exists; it does not
mean that the forecast has been validated.

In the report verified on 2026-10-21, 579 T4 forecasts are flagged as
`unseen_training_time_point`. The other 207 forecasts use T2 (61) or T3 (146),
which occurred in training inputs. All 786 report rows include an explanation.
These status fields do not alter predicted scores. Support planning excludes forecasts with unseen or unknown training time points.

## 4. System Workflow

```text
LIBS Longitudinal Assessment Data
                |
                v
       Data Preparation
                |
                v
   Historical Feature Creation
                |
                v
     Model Training and
      Validation-Based
        Model Selection
                |
                v
    Held-Out Test Evaluation
                |
                v
  Stagnation Screening and
    Intervention Planning
                |
                v
     Combined Progress Report
                |
                v
      Educator Review
```

The current prediction, screening, planning, and reporting scripts run as Python services. The workflow described here does not imply that a live dashboard or automated clinical decision system has been implemented.

## 5. Dataset Information

### LIBS Longitudinal Reading Assessments

The primary workflow uses the LIBS longitudinal reading-assessment data.

The prepared dataset used in the latest successful evaluation contains:

| Item | Count |
|---|---:|
| Observed assessment records | 2,834 |
| Participants with at least two observed scores | 786 |
| Usable next-score prediction examples | 1,262 |
| Participants represented in prediction examples | 709 |

The prediction dataset contains fewer participants than the screening dataset because prediction requires sufficient consecutive observed scores to construct the input features and target.

### Synthetic Development Dataset

The project also retains a synthetic reading-progress dataset for the legacy development workflow.

Synthetic records must be clearly identified as synthetic and must not be presented as real children's assessment records.

## 6. Machine-Learning Evaluation

The latest successful evaluation used participant-level separation across training, validation, and testing.

All three evaluation workflows now share the default outer split: 20% of
eligible participants are reserved using random seed 42. On the current data,
this gives 567 development participants and 142 test participants. Repeated
validation uses only the development group. Interval evaluation divides that
development group into fitting and calibration participants.

Membership is reproducible for the same eligible participant population and
test fraction; this is not a saved participant manifest. Adding or removing
eligible participants can change the split. A custom interval `test_size`
changes its holdout and should be treated as a different experiment.

Earlier repeated-validation runs included the whole dataset, and test results
have already been inspected. This change prevents future cross-workflow use
of test participants for fitting or validation; it cannot make past exploration
unseen. These are exploratory research results, not an untouched confirmatory
test. Further tuning should use development data only.

### Validation Results

The model was selected using validation Mean Absolute Error (MAE).

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Persistence Baseline | 26.19 | 28.52 | -0.936 |
| Linear Regression | 8.08 | 10.04 | 0.760 |
| Random Forest | 6.49 | 8.75 | 0.818 |

**Selected model:** Random Forest Regressor.

### Final Held-Out Test Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Persistence Baseline | 27.06 | 29.64 | -0.953 |
| Linear Regression | 8.69 | 10.82 | 0.740 |
| Random Forest | 7.82 | 10.36 | 0.761 |

### Evaluation Metrics

- **MAE:** Average absolute difference between actual and predicted scores. Lower is better.
- **RMSE:** Error metric that gives greater weight to larger errors. Lower is better.
- **R²:** Indicates how well predictions explain variation in the target values. Its interpretation depends on the dataset and evaluation setting.

The reported values are the results of the current data preparation, model configuration, and participant split. They should not be assumed to generalize to other datasets or populations without further evaluation.

## 7. Stagnation Screening Results

The latest screening run processed 786 participants.

| Preliminary category | Participants |
|---|---:|
| Low | 706 |
| Medium | 3 |
| High | 0 |
| Insufficient Data | 77 |

These counts describe the output of the current rules on this dataset. They do not establish the prevalence of stagnation or reading difficulties in a real-world population.

The screening thresholds require further evaluation before they can be considered reliable.
The updated run found zero latest-score declines and seven participants with
missing intermediate time points. Support priorities were Continue and monitor
(706), Collect evidence (77), and Educator review (3). Decline handling and
forecast-triggered review are tested with explicitly fictional cases; the real
cohort does not validate these rules' sensitivity or specificity.

## 8. Technologies Used

- **Python** — implementation language
- **Pandas** — data loading, cleaning, and manipulation
- **NumPy** — numerical calculations
- **Scikit-learn** — regression models, participant-level splitting, and evaluation metrics

## 9. Project Structure

Relevant files for Component 4:

```text
J26-DS-335-Dyslexia-Reading-Assistance/
|
├── backend/
│   └── app/
│       └── services/
│           └── learning_progress/
│               ├── data_preparation.py
│               ├── progress_prediction.py
│               ├── stagnation_detection.py
│               ├── intervention_planning.py
│               ├── progress_reporting.py
│               └── prepare_libs_longitudinal.py
|
└── datasets/
    └── component4/
        ├── reading_progress.csv
        ├── libs_reading_assessments_long.csv
        └── libs_prediction_results.csv
```

The prepared LIBS dataset must be available at the expected path before running the main LIBS workflow. The prediction-results CSV is generated by the evaluation script when result saving is enabled.

## 10. Setup and Execution

Run commands from the repository root.

### Prerequisites

- Python installed (the current verification environment uses Python 3.13.9)
- Dependencies from `requirements-component4.txt`, including `openpyxl` for Excel and `pytest` for tests
- The prepared BAS dataset, or the source workbook to generate it

Install the required Python packages if they are not already available:

```bash
python -m pip install -r requirements-component4.txt
```

The requirements file records the direct package versions in the verified
environment; it is not a complete dependency lockfile or a claim that a fresh
installation has been tested. Use a dedicated virtual environment for a new
installation. On the current development Mac, `/opt/anaconda3/bin/python`
has these dependencies; the default `python3` points to a different environment.

### Step 1: Validate Python Syntax

```bash
python -m compileall backend/app/services/learning_progress
```

### Step 2: Prepare or Validate Reading Data

If the prepared BAS CSV is missing, generate it from
`LIBS_DataSet_UnLinked_July2024.xlsx` in the repository root:

```bash
python backend/app/services/learning_progress/prepare_libs_longitudinal.py
```

This writes `datasets/component4/libs_reading_assessments_long.csv` only if the
path does not exist. Use `--output PATH` for a new file; `--overwrite` is required
to deliberately replace an existing file. Importing the preparation module
performs no file writes.
The current workbook produces 2,834 records for 786 participants.

If the prepared CSV already exists, validate it without rewriting it:

```bash
python backend/app/services/learning_progress/data_preparation.py
```

### Step 3: Evaluate Prediction Models

```bash
python backend/app/services/learning_progress/progress_prediction.py
```

This command prints validation results and final held-out test metrics. It also saves the evaluation summary to `datasets/component4/libs_prediction_results.csv`.

### Step 4: Run Stagnation Screening

```bash
python backend/app/services/learning_progress/stagnation_detection.py
```

### Step 5: Generate Support Plans

```bash
python backend/app/services/learning_progress/intervention_planning.py
```

### Step 6: Generate the Combined Report

```bash
python backend/app/services/learning_progress/progress_reporting.py
```

`data_preparation.py` loads and validates existing CSVs; it does not generate
the BAS CSV from the workbook.

### Run Component 4 Tests

```bash
python -m pytest tests/component4 -q -p no:cacheprovider
```

### Separate Older Multi-Assessment Workflow

The current prediction, screening, and reporting services use the BAS-only
`libs_reading_assessments_long.csv`, with `assessment` and `reading_score`
columns. The older exploratory workflow uses a different schema and now has
its own output path:

| Script | Input | Output |
|---|---|---|
| `prepare_libs_dataset.py` | Original workbook | `libs_all_assessments_long.csv` |
| `analyze_libs_progress.py` | `libs_all_assessments_long.csv` | `libs_progress_changes.csv` |
| `prepare_bas_prediction_data.py` | `libs_all_assessments_long.csv` | `libs_bas_prediction.csv` |

All CSV paths in this table are under `datasets/component4/`. The older format
contains `assessment_name`, `assessment_score`, and `source_variable`. Generate
it with `prepare_libs_dataset.py` only when using those exploratory scripts.
It must not replace the current BAS CSV. Existing generated results are not
automatically migrated or refreshed by this path change. The older scripts
overwrite their own output files when explicitly run.

## 11. Current Limitations

- The model predicts the next observed assessment score, not a score at a fixed future date.
- The combined report includes experimental next-score forecasts, but their real-world future accuracy has not been independently validated.
- The reported model results depend on the available LIBS data and the current participant split.
- The dataset does not directly measure the effectiveness of dyslexia interventions.
- Stagnation categories use preliminary, unvalidated rules.
- Intervention suggestions are rule-based and have not been demonstrated to improve reading outcomes.
- Further evaluation is required before using the component in real educational or clinical decision-making.
- The current implementation does not establish clinical validity.

## 12. Ethical Considerations

- Handle assessment records responsibly and follow the dataset's access and usage conditions.
- Avoid exposing participant identifiers in public demonstrations or reports.
- Use synthetic data only when clearly labelled as synthetic.
- Do not interpret model estimates as diagnoses or as replacements for educator assessment.
- Require educator review of support suggestions.
- Evaluate model performance and potential bias before considering real-world deployment.

## 13. Future Improvements

Potential future improvements include:

- Validate the current simple models on new data before adding complexity.
- Evaluating model performance across additional datasets and populations.
- Improving uncertainty estimates for predicted scores.
- Validating stagnation thresholds against suitable expert-reviewed criteria.
- Evaluating support recommendations using appropriate intervention-outcome data.
- Developing a user-facing dashboard when integrated with the wider system.
- Evaluating fairness, robustness, and generalization before deployment.

These are future directions, not claims about features already implemented.

## 14. Branch Information

Component branch:

`component4-learning-progress-prediction`

Base commit for the current local work:

`f7e46b9` — Add experimental prediction intervals

The corrections described here are subsequent local changes, not a new commit.

The current branch contains the data-preparation, prediction, stagnation-screening, intervention-planning, and combined-reporting updates described in this document.

## 15. Repeated Participant-Level Validation

Five repeated participant-level validation splits were used to compare the
Persistence Baseline, Linear Regression, and Random Forest models. Participants
were kept separate between the fitting and validation groups within each split.
These updated results were measured on 2026-10-21 using only the 567 development
participants. Each round uses 453 fitting and 114 validation participants;
the 142 test participants are excluded. The `test_size` argument in
`evaluate_repeated_group_validation()` means the validation fraction within
the development group, not the outer holdout fraction.

| Model | Mean MAE | MAE Std. Dev. | Mean RMSE | RMSE Std. Dev. | Mean R² | R² Std. Dev. |
|---|---:|---:|---:|---:|---:|---:|
| Persistence Baseline | 26.443 | 0.284 | 28.872 | 0.471 | -0.933 | 0.105 |
| Linear Regression | 8.247 | 0.424 | 10.368 | 0.513 | 0.751 | 0.019 |
| Random Forest | 6.979 | 0.325 | 9.343 | 0.282 | 0.798 | 0.008 |

Random Forest achieved the lowest mean MAE and RMSE and the highest mean R²
across these five splits. These results complement, but do not replace, the
separate held-out test evaluation.

The splits can overlap across rounds; their standard deviations describe
variation across these splits and are not confidence intervals. Earlier
generated repeated-validation CSVs were preserved and describe the old
whole-dataset experiment, so they do not match this updated table.

The results apply to the prepared LIBS dataset and current features. They do
not establish clinical effectiveness or guarantee generalization to children
diagnosed with dyslexia.


## 16. Experimental Prediction Intervals

An experimental split conformal prediction-interval evaluation was added
for the Random Forest reading-progress model. Participants were separated
into fitting, calibration, and test groups to avoid participant overlap.

The evaluation used nominal 90% coverage and produced these results on
2026-10-21 for the current prepared LIBS dataset, using the shared seed-42
test group. The fitting group contains 425 participants. These replace the
earlier interval results from a different test group (seed 62):

| Metric | Result |
|---|---:|
| Calibration participants | 142 |
| Test participants | 142 |
| Test records | 254 |
| Record-level coverage | 91.3% |
| Participant-level coverage | 85.2% |
| Mean interval width | 35.373 score points |
| Calibration threshold | 18.037 score points |

Record-level coverage measures the proportion of individual test records
inside their intervals. Participant-level coverage measures the proportion
of test participants for whom all evaluated records were covered.

Observed participant coverage is below the nominal 90% target on this split.
Record coverage above 90% does not establish that the participant target was
met. Do not select a different split or tune the threshold against these test
results to improve the reported coverage.

For an unavailable finite calibration rank, the implementation uses an infinite
threshold, producing the full valid score range [0, 90]. Such an interval is
uninformative, even though it covers all valid scores.

**Limitations:** These are experimental results from the current LIBS dataset.
The intervals are clipped to the documented 0–90 score range. For genuine
scores within this range, clipping does not remove covered scores.
The observed coverage does not guarantee nominal coverage for future
participants. The available local documentation does not establish a dyslexia-specific population,
and these results do not establish clinical validity or intervention
effectiveness.

To reproduce the updated evaluations without overwriting result CSVs, run
from the repository root:

```bash
python - <<'PY'
from backend.app.services.learning_progress.progress_prediction import (
    train_libs_progress_model,
    evaluate_repeated_group_validation,
    evaluate_prediction_intervals,
)
train_libs_progress_model(save_results=False)
evaluate_repeated_group_validation()
evaluate_prediction_intervals()
PY
```


## 17. Reproducible Evaluation and Research Materials

Run the complete aggregate evaluation without writing files:

```bash
python -m backend.app.services.learning_progress.evaluation_report
```

To save a **new** result file (an existing path is refused):

```bash
python -m backend.app.services.learning_progress.evaluation_report --output documentation/component4/evaluation_new_run.json
```

This evaluates the existing models, repeated validation, intervals, and reporting.
It includes package versions, input/source SHA-256 hashes, a test-participant
membership digest, missing-data counts, participant-weighted MAE, and transition
breakdowns. It does not save learner IDs or individual scores. A membership
digest detects changes but is not a persisted split manifest. Existing generated
CSVs are not overwritten. Keep the exact data and source version with a report.

Measured output: [evaluation_2026-10-21.json](documentation/component4/evaluation_2026-10-21.json).
Methodology, presentation outline, examples, and limitations:
[research_notes.md](documentation/component4/research_notes.md).

### Additional diagnostic results

| Comparator | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Current-time-point mean | 11.733 | 11.526 | 14.878 | 0.508 |

This baseline learns a mean target score for each current assessment time point
from fitting participants only. It does not use the future time point as a
predictor. It is a diagnostic comparator and does not change model selection.
Random Forest's test MAE with equal weight per participant is **8.066**; the
record-weighted MAE remains **7.822**.

| Observed transition | Test records | Approximate study years | Random Forest MAE |
|---|---:|---:|---:|
| T2 → T3 | 140 | 1 | 8.476 |
| T2 → T4 | 2 | 4 | 20.163 |
| T3 → T4 | 112 | 3 | 6.784 |

The two-record T2 → T4 subgroup cannot support a robust conclusion. The pooled
R² also reflects differences between assessment stages; it is not a percentage
accuracy or evidence of within-child causal learning improvement.

### Verification

Final local verification on 2026-10-21: **89 tests passed**. Python compilation
and `git diff --check` also completed successfully.

The implementation has unit tests for data validation, chronology, participant
separation, interval rank calculations, forecast support, trend screening,
and recommendation rules, plus temporary-file preparation and combined-report
integration tests. These verify software behavior, not educational effectiveness.
Run the test command in Section 10 after edits. The source workbook and existing
dataset CSVs were preserved during this implementation.
