# Component 4 — Learning Progress Prediction and Intervention Planning

Standalone PP1 research prototype for **J26-DS-335: AI-Based Adaptive Reading Assistance System for Children with Dyslexia**. The application includes a teacher/parent frontend, persistent session records, one-week reading forecasts, stagnation screening, intervention planning and follow-up monitoring. Components 1–3 are not required to run it.

The session application is in `backend/app/component4/`. The earlier LIBS/BAS experiment remains in `backend/app/services/learning_progress/` as a separate research benchmark. BAS scores are **not** converted into fluency percentages or weekly session observations.

## Start the existing installation

From the repository root on the development Mac:

```bash
cd /Users/binadisilva/J26-DS-335-Dyslexia-Reading-Assistance
.venv-component4/bin/python -m backend.app.component4.web --demo --model-dir models/component4/session_demo_v2
```

Open **http://127.0.0.1:5054**. `--demo` creates six fictional learners only when they do not already exist; it does not erase existing records. The first launch creates a `teacher` account and prints its random password once. Existing passwords are retained on later launches. Stop the server with **Ctrl+C**.

If the teacher password was not saved, stop the server and use the interactive recovery command:

```bash
.venv-component4/bin/python -m backend.app.component4.web --reset-password teacher
```

Enter a new password of at least 12 characters twice, then restart the server. Recovery changes the signing key, so previously issued browser sessions become invalid after restart. Passwords are hashed in SQLite; no credentials belong in Git or the research report.

## Install on a new checkout

Tested with Python 3.13.9 on macOS Apple Silicon. The existing development environment does not need to be recreated just to launch the app.

```bash
/opt/anaconda3/bin/python -m venv .venv-component4
.venv-component4/bin/python -m pip install -r requirements-component4.txt
.venv-component4/bin/python -m backend.app.component4.train --demo --output models/component4/session_demo_v2
.venv-component4/bin/python -m backend.app.component4.web --demo --model-dir models/component4/session_demo_v2
```

Training compares **Persistence, Linear Regression, Random Forest, XGBoost, LSTM and GRU**. It is a separate command; opening a learner page does not train a model. Training refuses to overwrite an experiment directory. For another experiment, choose a new directory and pass the same directory to the server with `--model-dir`.

The dependency file pins directly used libraries, including Flask, Werkzeug, Joblib, PyTorch and XGBoost. It is not a transitive dependency lockfile. A clean installation on other operating systems has not been verified.

## What the application does

- Register a learner using a pseudonymous alias, age, grade and educator-entered goals.
- Record dated assessments and calculate correct words per minute, reading accuracy and word error rate. Fluency uses a consistent educator-assigned 0–100 rubric.
- Display chronological history, charts, recent trends and an experimental forecast for seven days after the latest assessment.
- Explain provisional stagnation categories and identify recent declines. Changed difficulty, short histories and scale ceilings receive separate explanations.
- Propose a target skill, activity, support priority, practice duration/frequency and review period. An educator reviews or edits the proposal before saving it.
- Save each intervention's baseline, follow-up assessment, adherence and observed outcome; consider the previous outcome when proposing future support.
- Update learner goals without changing previously approved plan snapshots.
- Save educator-only professional notes, export learner reports as JSON and print reports from the browser.
- Give parents read-only access to their assigned learner. Teachers can access model evaluation and export pseudonymous training sessions.

Forms retain entered values when server validation fails. Original assessments and completed intervention outcomes are not silently overwritten. An active intervention must be reviewed before another is started.

## A short PP1 demonstration

1. Sign in as `teacher` and open **Demo learner 1**. Identify the synthetic-data banner.
2. Show the historical charts, the dated one-week forecast, its model name and the risk explanation.
3. Open a learner with a High screening category and compare the explanation with the underlying observations.
4. Review the active intervention using one of its later assessments. Record adherence and outcome notes, then complete the plan.
5. Show the calculated changes and the next proposed plan's use of intervention history. Review and approve a new plan.
6. Add a new assessment. Use a date seven days after the preceding assessment if demonstrating weekly forecasting; dates must not be in the future. Irregular observations are stored, but forecasts may be unavailable.
7. Show an invalid assessment being rejected, edit goals, save a professional note, and export the report.
8. Open **Model performance** to compare all models and explain how validation selected the model. The evaluation JSON can be downloaded from that page.
9. Demonstrate a parent account and confirm it cannot edit data or view another learner.

Create a parent account from a separate terminal, using a learner ID shown in the dashboard:

```bash
.venv-component4/bin/python -m backend.app.component4.web --add-parent demo-parent --learner DEMO0001
```

The command prompts for a password. It does not change or replace existing accounts.

## Data and prediction contract

See [the data contract](documentation/component4/data_contract.md) for fields, units, bounds, and CSV examples.

A forecast requires **four assessments exactly seven days apart**, at unchanged difficulty, with inputs inside the model's development-data ranges. Training examples require those four inputs plus the following weekly target. There must be at least 15 eligible learners for participant-separated training, validation and testing.

The forecast is one week ahead, not a recursive three-week forecast or a probability of improvement. Source and input checks are screening safeguards; they do not prove validity for a new population. No unsupported confidence percentage or expected intervention benefit is displayed.

Synthetic and collected records are kept distinct. A model trained on synthetic data refuses to forecast for a collected-data learner. Collected records can still be stored, charted and reviewed.

To train with collected observations, export **collected sessions** from the teacher dashboard, retain the raw field contract and run:

```bash
.venv-component4/bin/python -m backend.app.component4.train --csv /path/to/collected-sessions.csv --output models/component4/session_collected_v1
.venv-component4/bin/python -m backend.app.component4.web --model-dir models/component4/session_collected_v1
```

The CSV export omits aliases, professional notes and account details. An optional independent stagnation reference can be recorded on an assessment. It is used only in evaluation, never as a prediction feature. Do not copy the detector's output into the reference label.

## Research evidence and limits

The reproducible demonstration cohort has 180 fictional learners with 20 weekly observations each. The split has 115 fitting, 29 validation and 36 test learners. The generator uses simple noisy improvement, plateau and decline patterns; high scores on this dataset are not evidence of performance with real children.

Each model receives identical historical inputs. Validation selection uses equal-learner mean absolute error standardized by training-target standard deviations. Scalers are fitted on training records only. The selected model is then refitted on training + validation learners and evaluated separately on the untouched test group. Saved artifacts include the split manifest, example dates, dataset hash, source hashes, model hash and dependency versions.

The initial demonstration comparison selected **Linear Regression**. Read the saved JSON or the evaluation page for the measured MAE, RMSE and R² for speed, accuracy, fluency and WER. Stagnation evaluation includes accuracy, precision, recall, F1, confusion matrix, abstentions and a majority baseline. Synthetic references describe generator scenarios, not independent judgments about real children.

Intervention intensity is an explainable rule-based proposal informed by observations, forecast context and history. It is not a learned treatment policy. The outcome monitor records before/after differences and flags changed difficulty; it cannot attribute improvement to an intervention. Real-user prediction validation, educator review of thresholds/schedules, and intervention efficacy research still require suitable collected evidence.

## Verification

```bash
.venv-component4/bin/python -m pytest tests/component4 -q -p no:cacheprovider
.venv-component4/bin/python -m backend.app.component4.verify --model-dir models/component4/session_demo_v2
```

The verification command uses a temporary fictional database, checks authenticated routes with the saved model, and measures report latency against the proposal's five-second target. This is a local functional measurement, not a concurrent-user load test. Optional `--output NEW_FILE.json` saves evidence without overwriting previous results.

Tests cover existing BAS behaviour, participant separation, actual model families, model reloads, chronology, data validation, persistence, outcome matching, permissions, CSRF, escaped notes, source separation, goal updates and invalid-form recovery. Browser checks and the proposal mapping are recorded in [the PP1 checklist](documentation/component4/pp1_checklist.md).

## Files and operation

| Location | Purpose |
|---|---|
| `backend/app/component4/schema.py` | Raw observation validation and derived metrics |
| `backend/app/component4/store.py` | SQLite records, accounts, goals and intervention lifecycle |
| `backend/app/component4/models.py` | Model comparison, evaluation and guarded inference |
| `backend/app/component4/analytics.py` | Trend screening, planning and outcome reports |
| `backend/app/component4/web.py` | Local routes, authentication and teacher/parent permissions |
| `backend/app/component4/templates/` and `static/` | Dashboard frontend and print styles |
| `runtime/` | Ignored local database and signing key; preserve both when backing up |
| `models/component4/session_*/` | Ignored trusted training artifacts; reproducible with the training CLI |
| `documentation/component4/` | Evaluation evidence, methodology and PP1 checklist |

The server binds to `127.0.0.1` only and uses Flask's development server for the local evaluation. Database permissions are restricted to the local OS user; the database is not application-encrypted. Password hashing, role checks, session-cookie protections and CSRF checks support this local prototype. Internet deployment, institution-wide access control, concurrent deployment testing and integration with other components are outside this PP1 build.

If port 5054 is already occupied by this application, use its running terminal or stop it with Ctrl+C before restarting. Alternatively supply `--port 5055` and open the corresponding local URL. A missing model produces an explicit unavailable forecast; a corrupt artifact fails its checksum check.

## Earlier LIBS benchmark

[LIBS methodology and results](documentation/component4/libs_benchmark.md) and [research notes](documentation/component4/research_notes.md) document the separate BAS 0–90 experiment. Existing source datasets and previous result files are preserved. They do not establish a dyslexia-specific cohort or intervention effectiveness.
