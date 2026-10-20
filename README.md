# AI-Based Adaptive Reading Assistance System for Children with Dyslexia

## 1. Project Overview

This project aims to develop an AI-based adaptive reading assistance system to support children with dyslexia. The system uses reading assessment information to help identify reading difficulties, personalize reading support, and monitor learning progress over time.

The research project is divided into four components, with each component addressing a different part of the reading assistance process.

## 2. Research Project Components

| Component | Description |
|---|---|
| Component 1 (C1) | Reading Speech Analysis and Feature Extraction |
| Component 2 (C2) | Reading Difficulty Prediction, Cognitive Profiling and Explainable Risk Assessment |
| Component 3 (C3) | Adaptive Personalized Recommendation Engine |
| Component 4 (C4) | Learning Progress Prediction and AI-Driven Intervention Planning |

## 3. Component 4: Learning Progress Prediction and AI-Driven Intervention Planning

Component 4 focuses on analyzing longitudinal reading assessment data to understand learners' progress across assessment sessions.

### Main Objectives

- Analyze previous and current reading assessment scores.
- Predict a learner's score at a later assessment.
- Compare machine learning models for reading progress prediction.
- Identify patterns that may indicate limited or stalled progress.
- Support the planning of personalized reading interventions.
- Generate progress reports that can support the wider adaptive reading assistance system.

### Planned Functionalities

1. Reading progress data preparation.
2. Longitudinal assessment analysis.
3. Learning progress prediction.
4. Model performance evaluation.
5. Learning stagnation detection.
6. AI-driven intervention planning.
7. Learning progress reporting.

## 4. Technology Stack

- **Programming language:** Python
- **Data processing:** Pandas
- **Machine learning:** Scikit-learn
- **Prediction models:** Linear Regression and Random Forest (baseline models)
- **Model evaluation:** MAE, RMSE and R²
- **Dataset processing:** Excel and CSV files

Advanced models may be evaluated as the research progresses, depending on the available data and experimental results.

## 5. Dataset

### LIBS Secondary Dataset

Component 4 currently uses the Literacy in Birmingham and Solihull (LIBS) dataset, specifically the repeated British Ability Scales (BAS) word-reading assessment scores.

The dataset contains assessment observations collected at different time points. It allows the project to investigate changes in recorded word-reading scores and predict a later recorded score.

The original workbook is:

`LIBS_DataSet_UnLinked_July2024.xlsx`

The project uses an unlinked participant identifier to organize assessment records longitudinally. The participant identifiers are not treated as personal identities.

**Dataset source:** Aston University Research Data Repository  
https://researchdata.aston.ac.uk/id/eprint/644/

### Dataset Preparation

The current preparation workflow:

1. Loads the LIBS Excel workbook.
2. Identifies repeated assessment columns.
3. Converts the repeated assessment columns into a long-form dataset.
4. Converts time points and assessment scores into numeric values.
5. Treats dot markers as unavailable scores rather than zero.
6. Preserves assessment names and source-variable references.
7. Selects the BAS word-reading assessment for the initial prediction experiment.
8. Creates previous-score, current-score and later-target-score fields.
9. Records the time points associated with the observations.
10. Saves the prepared datasets separately from the original workbook.

Zero scores are currently retained because the available variable descriptions permit scores from 0 to 90 but do not explain the specific meaning of zero. They should not be removed or reclassified without supporting evidence.

### Synthetic Data

The project also contains a synthetic reading dataset generator for development and testing.

Synthetic records are generated for testing data-processing workflows. They must be clearly distinguished from the real LIBS secondary dataset and must not be presented as actual records from children.

## 6. Project Directory Structure

```text
J26-DS-335-Dyslexia-Reading-Assistance/
│
├── backend/
│   └── app/
│       └── services/
│           └── learning_progress/
│               ├── analyze_libs_progress.py
│               ├── data_preparation.py
│               ├── inspect_bas_zero_scores.py
│               ├── inspect_libs.py
│               ├── intervention_planning.py
│               ├── prepare_bas_prediction_data.py
│               ├── prepare_libs_dataset.py
│               ├── progress_prediction.py
│               ├── progress_reporting.py
│               ├── stagnation_detection.py
│               ├── train_libs_bas_model.py
│               ├── validate_bas_prediction_data.py
│               └── validate_libs_progress.py
│
├── datasets/
│   └── component4/
│       ├── generate_sample_dataset.py
│       ├── libs_assessment_validation.csv
│       ├── libs_bas_prediction.csv
│       ├── libs_progress_changes.csv
│       ├── libs_reading_assessments_long.csv
│       └── reading_progress.csv
│
├── documentation/
│   └── component4/
│       └── VariableDescriptions_LIBS_July2024.txt
│
├── models/
│   └── component4/
│
├── tests/
│   └── component4/
│
├── LIBS_DataSet_UnLinked_July2024.xlsx
└── README.md
```

The directory structure describes the current working layout. Files may be added, removed or reorganized as implementation continues.

## 7. Setup and Execution

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- The required Python libraries
- The LIBS workbook, stored at the project root

### Install Dependencies

Install the required packages if they are not already available in your Python environment:

```bash
# Install the libraries used by the current data-processing and baseline-model scripts
python -m pip install pandas openpyxl scikit-learn
```

### Prepare the LIBS Dataset

Run these commands from the project root.

```bash
# Convert the original LIBS assessment data into long-form records
python backend/app/services/learning_progress/prepare_libs_dataset.py

# Analyze changes between consecutive assessment scores
python backend/app/services/learning_progress/analyze_libs_progress.py

# Prepare the BAS word-reading prediction dataset
python backend/app/services/learning_progress/prepare_bas_prediction_data.py

# Validate the prepared prediction dataset
python backend/app/services/learning_progress/validate_bas_prediction_data.py
```

### Inspect and Evaluate the Data

```bash
# Inspect the original LIBS dataset
python backend/app/services/learning_progress/inspect_libs.py

# Investigate zero scores in the BAS prediction dataset
python backend/app/services/learning_progress/inspect_bas_zero_scores.py

# Validate the longitudinal LIBS progress data
python backend/app/services/learning_progress/validate_libs_progress.py
```

### Train the Baseline Prediction Models

```bash
# Train and compare the baseline models using the prepared LIBS data
python backend/app/services/learning_progress/train_libs_bas_model.py
```

Run the preparation steps before training if the required output CSV files do not exist or need to be regenerated.

## 8. Prediction Approach

The current baseline experiment uses previous and current BAS word-reading scores and their assessment time points to predict the next available recorded BAS word-reading score.

The prediction target is the next available observed assessment score, not a guaranteed score at a fixed future date.

### Baseline Models

**Linear Regression**

Used as a simple baseline to estimate the relationship between the input assessment information and the later score.

**Random Forest Regressor**

Used as a non-linear baseline model to capture patterns that may not be represented by a simple linear relationship.

**Persistence Baseline**

Uses the current score as the prediction for the next score. This provides a reference point for evaluating whether the machine learning models improve on a simple prediction strategy.

### Evaluation Metrics

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual scores.
- **Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors.
- **R²:** Measures how well the predictions explain variation in the target scores relative to a mean-based reference.

The experiment uses participant-grouped training and testing so that a participant's records are not shared between the training and test sets.

## 9. Initial Experimental Results

In the current baseline experiment, the following results were obtained:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence Baseline | 25.55 | 27.71 | -0.811 |
| Linear Regression | 7.46 | 9.53 | 0.786 |
| Random Forest | 6.40 | 8.61 | 0.825 |

These results are preliminary and depend on the current dataset preparation, target definition and held-out participant split. They must be reproduced after relevant code or data changes.

The results do not demonstrate clinical validity, diagnostic ability or suitability for making decisions about individual children with dyslexia.

## 10. Current Limitations

- The LIBS dataset is a general literacy dataset and is not, by itself, proof of performance specifically for children diagnosed with dyslexia.
- The available BAS scores do not directly provide every feature originally planned for Component 4, such as reading fluency, word error rate or intervention effectiveness.
- The meaning of zero scores requires further clarification from authoritative dataset documentation.
- The current experiment predicts the next available observed assessment score; assessment intervals may vary.
- Model performance needs further validation and comparison with appropriate baselines.
- Stagnation detection and intervention planning require separate evaluation.
- The current baseline results do not establish clinical validity or real-world effectiveness.

## 11. Future Work

Planned work includes:

1. Investigating the meaning and distribution of zero scores.
2. Reviewing data quality and documenting the limitations of the LIBS variables.
3. Improving prediction features and evaluation methods where supported by the dataset.
4. Testing additional models if justified by the research design and available data.
5. Evaluating learning stagnation detection.
6. Developing and evaluating personalized intervention-planning rules.
7. Integrating prediction, stagnation detection, intervention planning and progress reporting.
8. Documenting experiments, results, limitations and reproducibility.
9. Evaluating the system with suitable additional data where access and ethical approval permit.

## 12. Ethical Considerations

The project must protect participant privacy and follow the applicable dataset licence and research requirements.

- Use participant identifiers only for appropriate longitudinal analysis.
- Do not attempt to identify individual participants.
- Do not claim