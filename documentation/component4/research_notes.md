# Component 4: methodology and research presentation notes

> Scope: this document records the earlier LIBS/BAS benchmark. The standalone session application and current PP1 workflow are documented in the root README and `pp1_checklist.md`. Historical statements about missing dashboards or session storage apply to this benchmark only.

Project: J26-DS-335 — AI-Based Adaptive Reading Assistance System for Children with Dyslexia

Individual component: Learning Progress Prediction & AI-driven Intervention Planning

Status: executable research prototype. Results below were reproduced on 21 October 2026. They do not establish clinical validity, deployment readiness, or intervention effectiveness. The implementation uses machine learning for score prediction and transparent rules for educational support planning.

## 1. Research question and scope

Can previous and current BAS word-reading scores, together with their assessment time points, estimate the next observed score for participants held out from fitting? Can those estimates be presented alongside historical changes and understandable support suggestions?

This component estimates a next observed **score**, then describes the difference from the current score. It does not predict a learning rate at a known future date. It does not diagnose dyslexia, identify the cause of a difficulty, or predict the effect of an intervention.

Treat the available LIBS data as general literacy data. The local workbook and dictionary do not establish a dyslexia-specific cohort. Verify the dataset's original source, citation, access conditions, consent scope, and redistribution permissions with the supervisor before publishing data. The supplied dictionary is evidence for variable names, score ranges, and study ages; it is not evidence of clinical representativeness.

## 2. Data preparation and missingness

The source workbook has 977 rows. For the four BAS variables used by this protocol:

| Valid observed BAS scores | Source rows |
|---|---:|
| 0 | 145 |
| 1 | 46 |
| 2 | 77 |
| 3 | 156 |
| 4 | 553 |

The BAS range is 0–90 according to `VariableDescriptions_LIBS_July2024.txt`. Zero is retained. Dot markers and invalid numeric values become unavailable observations; no missing score is replaced by zero. Workbook preparation keeps participants with at least two valid scores: 786 participants and 2,834 assessments. Of these, 709 have enough history to produce 1,262 prediction examples; 77 with only two scores can be reported but provide no supervised next-score examples.

The current prepared CSV has no invalid rows or identical duplicates removed by the updated loader. The loader checks assessment/time-point consistency and rejects conflicting duplicate scores. Identical duplicates may be collapsed with counts recorded in `DataFrame.attrs['quality_report']`. Source preparation is import-safe and refuses to replace an existing output without `--overwrite`.

These eligibility restrictions can create selection bias. Results describe participants with sufficient observations, not all 977 source rows. Reasons for missing assessments and missingness mechanisms are not established here. The protocol currently uses T1–T4; later variables mentioned in the dictionary are outside this experiment and require a separate, pre-specified compatibility review before inclusion.

## 3. Features and experimental design

For each participant, assessments are sorted chronologically. Each example contains:

- Previous observed BAS score and its time point.
- Current observed BAS score and its time point.
- Target: the next available observed BAS score.

Participant IDs and target time points are not model inputs. A participant with T1, T2, T4 observations contributes a T2 → T4 target; no T3 observation is invented.

The default outer split reserves 20% of the 709 eligible participants using seed 42: 567 development participants and 142 test participants. The development group is split into 453 fitting and 114 validation participants (seed 43). Linear Regression's scaler is fitted inside its pipeline. Random Forest uses 200 trees, minimum leaf size 2, seed 42. Validation MAE selects between those two models; both are refitted on the 567 development participants for test comparisons. Persistence is a comparator, not a selection candidate.

Five repeated splits (seed 52, 20% validation) operate only within the 567 development participants. Repeated splits overlap across rounds; their standard deviation is variability across those splits, not a confidence interval or five independent experiments.

The same default test participant group is reserved for interval evaluation. Calibration uses 142 development participants and fitting uses 425 (calibration seed 63). The interval model is a separately fitted Random Forest; it is not the model refitted on all development participants for point predictions.

The data population and random seed reproduce membership. A SHA-256 membership digest is recorded to detect changes; no persistent ID manifest is implemented. Adding/removing eligible participants can change membership. Preserve the original dataset with the experiment. Custom interval split fractions define a different experiment.

Earlier whole-dataset repeated validation and test-result inspection have already occurred. The corrected separation prevents further use of test participants for fitting or tuning, but cannot undo that history. Present these results as exploratory. A confirmatory study needs a pre-specified protocol and new/unseen evaluation data.

## 4. Measured results

All values below are from the aggregate execution saved in `evaluation_2026-10-21.json`, not hypothetical improvements.

| Model/comparator | Test MAE | Test RMSE | Test R² | Equal-participant MAE |
|---|---:|---:|---:|---:|
| Persistence | 27.055 | 29.641 | −0.953 | 26.658 |
| Linear Regression | 8.694 | 10.821 | 0.740 | 8.930 |
| Random Forest | 7.822 | 10.360 | 0.761 | 8.066 |
| Current-time-point mean | 11.526 | 14.878 | 0.508 | 12.153 |

Random Forest was selected with validation MAE 6.495. Its mean repeated-validation MAE is 6.979, with split standard deviation 0.325.

The current-time-point mean is an additional diagnostic baseline: it learns average targets for each current time point using only fitting participants. Unseen stages use the fitting-group mean. Its validation MAE is 11.733. It was added to assess whether prediction gains exceed a simple stage-based reference, without tuning or changing model selection after inspecting the test results.

Record MAE weights every prediction example equally. Equal-participant MAE first averages absolute errors within each participant, then averages across participants. This avoids giving participants with more examples extra weight. R² is not percentage accuracy. Pooled scores can benefit from separating assessment stages, so transition-specific errors also matter.

| Current → target assessment | Approximate study-year gap | Test records | Random Forest MAE |
|---|---:|---:|---:|
| T2 → T3 | 1 | 140 | 8.476 |
| T2 → T4 | 4 | 2 | 20.163 |
| T3 → T4 | 3 | 112 | 6.784 |

The T2 → T4 subgroup is too small for a reliable comparison. No significance test, generalization guarantee, or causal interpretation follows from these numbers.

## 5. Prediction intervals

For calibration participant i, take the maximum absolute error across that participant's evaluated records. Sort those participant maxima. With m calibration participants and nominal coverage c, use rank `ceil((m + 1) * c)` (one-based). If the rank exceeds m, use an infinite threshold. Construct prediction ± threshold and restrict it to the known 0–90 score range.

This use of a participant maximum targets simultaneous coverage of a new participant's comparable evaluated records under participant-level exchangeability and a fixed fitting/calibration procedure. It is not a probability that one specific child will improve. The assumptions are not established for a different population, different observation schedule, or an unseen future horizon. Background theory: [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).

| Interval measure | Measured result |
|---|---:|
| Nominal coverage | 90% |
| Calibration participants | 142 |
| Test participants | 142 |
| Test records | 254 |
| Record coverage | 91.34% |
| Participant coverage (all records covered) | 85.21% |
| Mean interval width | 35.373 points |
| Calibration threshold | 18.037 points |

The observed participant coverage is below the nominal target. A finite test sample can deviate from marginal nominal coverage; this single result neither establishes a guarantee nor justifies choosing a more favorable split. Do not tune the threshold on these test outcomes.

Clipping to 0–90 does not remove covered genuine BAS scores within that valid range. An infinite threshold becomes [0, 90], which is uninformative even though all valid scores are covered. Intervals are evaluated separately; they are not attached to latest-history forecasts because the fitting models and future input distributions differ.

## 6. Monitoring and recommendations

The provisional category uses the average of up to three recent observed changes, requiring at least two changes. High means <= 0, Medium means > 0 and < 2, and Low means >= 2. These are development rules, not validated cutoffs. Categories are labels for a heuristic, not measured probabilities of dyslexia or clinical risk.

The latest score change is also reported separately. A recent decline prompts educator review even when earlier gains leave the average positive. The rate per study year uses approximate dictionary ages T1=4, T2=5, T3=6, T4=9; it is not a rate derived from individual dates. Missing intermediate points and scale ceilings are identified.

Measured screening categories remain Low 706, Medium 3, High 0, Insufficient Data 77. There are zero latest-score declines and seven participants with missing intermediate time points. Therefore the real data cannot establish detection sensitivity for declines or High-category cases. Fictional cases test code behavior only. These categories are not supervised class labels; classifier accuracy or a confusion matrix would be unjustified without reference annotations.

The combined report includes 786 numerical forecasts. Of these, 579 use T4 inputs unseen in training and are explicitly flagged. The other 207 use T2/T3. A seen time point is only one support check; it does not establish future accuracy or similarity of every feature.

Planning order:

1. Insufficient history → collect further evidence; any recent decline is still mentioned for review.
2. Observed decline or High/Medium category → educator review.
3. Negative experimental forecast at a seen training time point → request corroborating observations before changing support.
4. Otherwise → continue the existing plan and monitor.

An unsupported forecast cannot change priority. A positive forecast cannot cancel an observed decline. Every plan requires educator review and states the evidence used. Activities include supported oral reading with feedback and recording word errors for educator-selected practice. Their general literacy rationale comes from the [IES/WWC foundational reading practice guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/21), intended for K–3; an educator must determine suitability for the learner. This does not validate the project's activity mapping or supply intervention-outcome evidence.

BAS totals do not distinguish decoding, fluency, and comprehension deficits. There are no intervention assignments, adherence measurements, or post-intervention outcomes in this implementation. Consequently, the system cannot rank activities by predicted benefit, learn an optimal treatment, or claim that recommendations improve reading.

### Fictional demonstration (not a measured participant)

Observed scores 10 → 50 → 40 produce changes +40 and −10: mean +15, category Low, latest-change flag Decline. The plan therefore requests educator review rather than allowing the positive mean to hide the latest decline. This scenario is covered by automated tests.

## 7. What to say in a presentation

Suggested six-slide outline:

1. **Problem and role:** educators need readable summaries of assessment history; this component combines an experimental score estimate with observable trends.
2. **Data and exclusions:** show the eligibility table and identify LIBS as general literacy data; explain retained zeros and missing observations.
3. **Method:** four historical inputs → grouped development/validation/test splits → simple model comparison → separate participant-calibrated intervals.
4. **Results:** show MAE/RMSE/R², the stage baseline, and transition counts; distinguish record and participant metrics.
5. **Prototype demonstration:** use the fictional rise-then-decline case and show forecast-support warnings and recommendation evidence.
6. **Limitations and next study:** participant interval coverage below target, unsupported T4 forecasting, class imbalance in heuristic outputs, missing outcome labels, and need for new evaluation data.

Suggested explanation: “The selected Random Forest estimated the next observed BAS score with test MAE 7.82 on the current participant split. This is a score error in BAS points, not diagnostic accuracy. Recommendations are transparent educational rules requiring educator review; their effectiveness has not been tested.”

### Likely supervisor questions

- **Why not a neural network?** The current data offer at most four BAS observations per included participant. Simple baselines and validation answer the present question without extra model complexity; additional complexity needs evidence of benefit.
- **Is this dyslexia prediction?** No. The project concerns assistance, but this component's available experiment uses general literacy scores and does not validate dyslexia-specific performance.
- **Why participant-level splitting?** Multiple records from the same participant are related. Splitting individual rows could let a learner's data occur on both sides of evaluation.
- **Why are future forecasts uncertain?** The next observed date varies, and most latest-history inputs are T4, whereas fitting examples use T2/T3. No later outcome is available in this protocol to validate those estimates.
- **Is planning AI-driven?** Prediction uses ML; recommendation rules use historical evidence and cautiously include supported forecast context. The planner is not a learned intervention-effect model.
- **Can you claim novelty?** Claim only the implemented integration and evaluated behavior. A novelty claim requires a systematic literature comparison, which has not been performed here.

## 8. Reproduction and remaining external work

From the repository root, using the verified Mac environment:

```bash
/opt/anaconda3/bin/python -m pytest tests/component4 -q -p no:cacheprovider
/opt/anaconda3/bin/python -m backend.app.services.learning_progress.evaluation_report
```

Use `--output NEW_PATH.json` to save a new aggregate report. Existing output paths are refused. The saved report includes environment versions, dataset/workbook/source hashes, test membership digest, repeated split results, and diagnostics. It omits individual participant records. Existing generated CSVs may represent older experiments; use the dated JSON and accompanying code for these results.

Software checks cover finite/infinite interval ranks, participant grouping, validation-only selection, chronological features, bad data, duplicate handling, missing history, unsupported forecasts, recent declines, and report joins. Passing tests establishes the tested behavior, not clinical or educational validity. A fresh installation in a new environment has not been verified.

Final local verification: 89 tests passed, Python compilation succeeded, and
the Git whitespace check passed. Checksums confirm the original workbook and
all nine existing CSVs were preserved.

Work requiring additional evidence or project decisions:

- Confirm dataset provenance, access terms, and the appropriate formal dataset citation with the supervisor.
- Obtain educator-reviewed reference judgments for monitoring thresholds and recommendation suitability.
- Define a fixed prediction horizon and evaluate on new participants or a compatible later wave under a pre-specified protocol.
- Assess demographic subgroup performance only when valid attributes, consent scope, and sample sizes support it; no fairness claim is made here.
- Collect consented intervention/outcome data before evaluating effectiveness or learning an adaptive policy.
- Agree an API/data contract with the other project components before integrating real sessions or a dashboard. Those systems are outside this local service implementation.
