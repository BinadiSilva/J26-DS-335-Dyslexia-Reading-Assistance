# Component 4 — implementation and demonstration checklist

Reviewed against `IT23269934.docx`, particularly sections 3.2–3.5, 5.1–5.2 and the Component 4 workflow/prototypes. Review date: **2026-10-21**.

The standalone software workflow is implemented, including the frontend. It runs without Components 1–3. This document records software completion and its evidence; it does not claim that real-user research validation or intervention efficacy has been completed. The interface uses normal product wording and does not display the evaluation stage.

## Proposal mapping

| Proposal requirement | Implementation | Verification / demonstration |
|---|---|---|
| Store longitudinal reading observations | SQLite learners and immutable dated sessions; raw inputs validated, speed/accuracy/WER calculated | Persistence, duplicate-date, bounds and chronology tests; browser registration and assessment submission |
| Speed, accuracy, fluency, WER, difficulty and intervention information | Explicit units and raw CSV contract; practice minutes, approved plans, adherence and follow-up records | `schema.py`, `store.py`; [data contract](data_contract.md) |
| Analyse historical/current progress | Chronological session table, dated charts, recent weekly slopes and latest-decline handling | Monitoring tests; observed and forecast chart segments inspected in the browser |
| Predict future progress | Four weekly observations predict the next week; all four targets; recompute from saved sessions | Forecast/reload tests; six seeded learners produced available forecasts using the saved model |
| Compare LR, RF, LSTM, GRU and XGBoost | Actual implementations of all five, plus Persistence; identical inputs and learner-separated splits | Full comparison run in `session_demo_v2`; individual-family fitting/reload tests |
| MAE, RMSE and R² | Four-target comparison, equal-learner MAE, validation-based selection and separate refitted-model test results | [Saved comparison](session_evaluation_pp1_2026-10-21.json); teacher “Model performance” page |
| Detect possible stagnation | Explainable provisional screening, insufficient-data/comparability checks, recent decline and ceiling handling | Independent-label evaluation; synthetic confusion matrix and majority baseline; chronology tests |
| Tailored intervention plan and intensity | Goal shortfalls select skill; trends and supported forecasts inform priority; schedule and activity require educator approval | Plan/forecast tests; browser approval and saved baseline |
| Continuous updates and history-aware planning | Assessments and goal edits refresh proposals; previous adherence and outcomes inform future suggestions | Browser goal change changed the proposed skill and displayed the previous matching intervention outcome |
| Monitor intervention outcomes | Baseline/follow-up matching, raw metric changes, comparability, adherence and outcome notes | Browser completed a fictional plan; observed improvement shown with association-only interpretation |
| Teacher and parent progress reports | Learner dashboard, charts, goals, forecasts, intervention history, JSON export and print control | Browser teacher workflow; automated parent assignment, read-only access and private-note exclusion tests |
| Authorized access and privacy | Hashed accounts, teacher/assigned-parent checks, CSRF, escaped notes, protected cookies, local-only server | Authentication/authorization/CSRF tests; login verified in browser |
| Consistent outputs | Saved selected model, deterministic inference, checksum verification and experiment metadata | Repeat-prediction and round-trip tests; corrupt artifact rejected |
| Preferably under five seconds | Model loads once; report generation does not retrain | [Local measurements](verification_2026-10-21.json): 18 report samples, maximum 0.00134 seconds; authenticated learner route 0.01832 seconds |
| Multiple learners and sessions | Independent learner histories and parent assignments; 180-learner training cohort, six seeded demonstration histories | Isolation tests and local six-learner verification; this is not concurrent-user load testing |
| Understandable reports | Plain-language labels, explanations, units, empty states, form-error recovery and responsive layout | Browser inspection at narrow pane widths (approximately 615–722 px); numerical history accompanies charts |

## Verification completed

Command:

```bash
.venv-component4/bin/python -m pytest tests/component4 -q -p no:cacheprovider
```

Result: **135 passed in 10.61 seconds**, with no skips in the configured environment. This includes the retained LIBS/BAS benchmark tests and the standalone session application.

The separate verification command checks the real saved model using a temporary database. All six checked authenticated routes returned HTTP 200: dashboard, learner page, model performance page, performance JSON, learner JSON and training CSV. Its latency measurements exclude network, browser rendering and concurrent load.

Browser workflow completed on the local application:

1. Signed in and viewed the learner dashboard.
2. Created a fictional record, now named **Sample learner**.
3. Submitted an assessment with more correct words than passage words; the error was explained and entered values were preserved.
4. Corrected and saved the assessment; progress refreshed.
5. Approved a support plan with the saved baseline.
6. Saved a later assessment, selected it for follow-up and recorded adherence; the plan became completed and displayed metric changes.
7. Saved an educator note and changed a goal; the next proposal reflected both the new goal and relevant intervention history.
8. Opened the model performance screen with all six comparators and the selected-model results.
9. Checked the seeded learner's dated forecast and rendered history charts; no warning/error console messages were captured.
10. Confirmed the updated navigation and screens contain no PP1 stage labels.

Only fictional records were used. The extra **Sample learner** remains available locally for inspecting the workflow; it is not training or validation evidence. Parent permissions and exports were checked automatically, not through a separate manual parent-browser session. The print control and print stylesheet are implemented; an exported PDF has not been visually verified. Desktop-wide layouts, other browsers, screen readers and concurrent deployment have not been fully tested.

## Decisions to explain during the demonstration

- **Forecast horizon:** one week from the latest assessment, requiring four observations exactly seven days apart at unchanged difficulty. A three-week trajectory or individual confidence percentage is not inferred from the mockup.
- **Neural-network library:** LSTM and GRU use PyTorch instead of the proposal's planned TensorFlow/Keras stack. They are actual recurrent models, trained and compared using the same input windows as the baselines.
- **Selected model:** Linear Regression won the synthetic validation comparison. Advanced models are not assumed to outperform a baseline. The selected model was refitted using development learners and evaluated on untouched test learners.
- **Source separation:** synthetic and collected records are exported/trained separately. A synthetic model refuses forecasts for collected-data learners. Input-range checks also reject unsupported histories.
- **Intervention logic:** explainable educational rules use observations, forecast context, goals and prior outcomes. The system does not learn a treatment effect or promise an improvement percentage.
- **Preserved benchmark:** existing LIBS/BAS work remains a separate literacy-score experiment. BAS 0–90 is not treated as fluency or converted into fictional weekly assessments.

## Research evidence still to collect

These activities require real observations and domain input; implementing more code cannot truthfully substitute for them:

- Obtain suitable consent/approval and repeated collected observations for the intended population.
- Review the fluency rubric, goals, stagnation thresholds and support schedules with an educator.
- Train/validate using the collected-data contract and independently annotated stagnation references.
- Evaluate intervention efficacy with an appropriate research design. Before/after differences alone do not establish causation.
- Conduct teacher/parent usability review and any institution-specific security or scale testing before wider use.

See the [README](../../README.md) for launch, password recovery, parent account creation and the demonstration sequence. No integration or public deployment is required for this standalone build.
