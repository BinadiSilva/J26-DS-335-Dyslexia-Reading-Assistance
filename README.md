
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
- Handling unavailable values and duplicate participant/time-point records according to the preparation rules.

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

Model selection uses validation MAE. Final performance is evaluated on a separate group of held-out participants.

### 3.3 Stagnation Screening

The stagnation-detection service analyzes observed score changes for each participant.

It calculates recent score-change trends and assigns a preliminary category:

- **Low:** Recent scores show progress according to the current rules.
- **Medium:** Recent average change is small and positive.
- **High:** Recent average change is zero or negative.
- **Insufficient Data:** There are not enough observed score changes for screening.

These categories are based on preliminary rules. They are not clinically validated diagnoses or reliable standalone measures of a child's reading needs.

### 3.4 Intervention Planning

The intervention-planning service generates preliminary reading-support suggestions based on the screening category and available assessment history.

Suggestions can include continuing suitable reading practice, monitoring progress, or seeking educator review when information is insufficient or progress may be limited.

The generated plans are rule-based suggestions. They do not demonstrate that an intervention is effective and must be reviewed by a qualified educator before being used in practice.

### 3.5 Combined Progress Reporting

The reporting service combines:

- Latest observed reading scores and assessment time points
- Historical next-score estimates
- The model used for those estimates
- Recent score-change summaries
- Preliminary stagnation categories
- Suggested reading-support activities
- Recommendation frequency and session-duration guidance

The report is printed in the terminal and returned as a pandas DataFrame.

**Important:** The reporting service generates an experimental next-score forecast from each eligible participant's latest assessment. It predicts the next observed score, which may not correspond to a fixed future date. This forecast has not been independently validated as a real-world future prediction.

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

- Python installed
- Pandas, NumPy, and Scikit-learn installed
- The prepared dataset available at `datasets/component4/libs_reading_assessments_long.csv`

Install the required Python packages if they are not already available:

```bash
python -m pip install pandas numpy scikit-learn
```

### Step 1: Validate Python Syntax

```bash
python -m compileall backend/app/services/learning_progress
```

### Step 2: Prepare Reading Data

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

Run the preparation step before the other services if the prepared longitudinal dataset has not yet been generated or needs to be refreshed.

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

- Testing additional time-series and regression models.
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

Latest verified implementation commit:

`9de45ee` — Test repeated participant-level validation

The current branch contains the data-preparation, prediction, stagnation-screening, intervention-planning, and combined-reporting updates described in this document.

## 15. Repeated Participant-Level Validation

Five repeated participant-level validation splits were used to compare the
Persistence Baseline, Linear Regression, and Random Forest models. Participants
were kept separate between the fitting and validation groups within each split.

| Model | Mean MAE | MAE Std. Dev. | Mean RMSE | RMSE Std. Dev. | Mean R² | R² Std. Dev. |
|---|---:|---:|---:|---:|---:|---:|
| Persistence Baseline | 26.849 | 0.276 | 29.248 | 0.237 | -1.011 | 0.129 |
| Linear Regression | 8.218 | 0.238 | 10.404 | 0.292 | 0.746 | 0.018 |
| Random Forest | 7.069 | 0.124 | 9.487 | 0.281 | 0.789 | 0.013 |

Random Forest achieved the lowest mean MAE and RMSE and the highest mean R²
across these five splits. These results complement, but do not replace, the
separate held-out test evaluation.

The results apply to the prepared LIBS dataset and current features. They do
not establish clinical effectiveness or guarantee generalization to children
diagnosed with dyslexia.


## 16. Experimental Prediction Intervals

An experimental split conformal prediction-interval evaluation was added
for the Random Forest reading-progress model. Participants were separated
into fitting, calibration, and test groups to avoid participant overlap.

The evaluation used a nominal 90% confidence level and produced these results
for the current prepared LIBS dataset:

| Metric | Result |
|---|---:|
| Calibration participants | 142 |
| Test participants | 142 |
| Test records | 248 |
| Record-level coverage | 94.8% |
| Participant-level coverage | 90.8% |
| Mean interval width | 39.176 score points |
| Calibration threshold | 20.326 score points |

Record-level coverage measures the proportion of individual test records
inside their intervals. Participant-level coverage measures the proportion
of test participants for whom all evaluated records were covered.

**Limitations:** These are experimental results from the current LIBS dataset.
The intervals are clipped to the 0–90 score range, which can affect coverage.
The observed coverage does not guarantee nominal coverage for future
participants. The dataset is not specific to children diagnosed with dyslexia,
and these results do not establish clinical validity or intervention
effectiveness.
