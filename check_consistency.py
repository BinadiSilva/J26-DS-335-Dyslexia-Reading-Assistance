"""Do the batch evaluation path and the deployed predict_session path give the SAME predictions?

Run from the repo root (inside Docker):  python check_consistency.py
Compares, for every hold-out session, the difficulty score, intensity and weak skill predicted
(a) in batch (what evaluate_baselines.py uses for its tables) and (b) by predict_session (what the UI uses).
"""
import numpy as np
import pandas as pd
import joblib

from backend.app.services.reading_difficulty.features import (
    apply_grade_norms, clean, engineer_features, fit_grade_norms, load_raw, stratified_learner_level_split)
from backend.app.services.reading_difficulty.pipeline import predict_session

RAW, ART = "datasets/comp02_raw_reading_sessions.csv", "outputs/artifacts.joblib"
art = joblib.load(ART)
df = engineer_features(clean(load_raw(RAW)))
train, test = stratified_learner_level_split(df, stratify_col="intervention_intensity")
test = apply_grade_norms(test, fit_grade_norms(train))
raw = clean(load_raw(RAW)).set_index("session_id", drop=False)

def batch(target):
    t = art[target]
    X = test.reindex(columns=t["feature_columns"], fill_value=0)
    return t["model"].predict(X)

def label(target, v):
    labels = art[target]["class_labels"]
    return v if isinstance(v, str) else labels[int(v)]

b_score = np.asarray(batch("difficulty_score"), dtype=float)
b_int = [label("intervention_intensity", v) for v in batch("intervention_intensity")]
b_ws = [label("weak_skill", v) for v in batch("weak_skill")]
rows = []
for sid in test["session_id"]:
    o = predict_session(art, raw.loc[sid].to_dict())
    rows.append((o["difficulty_score"], o["intervention_intensity"], o["weak_skill"]))
s_score, s_int, s_ws = (np.array([r[i] for r in rows], dtype=object) for i in range(3))

diff = np.abs(b_score - s_score.astype(float))
print(f"sessions: {len(test)}")
print(f"difficulty score: max |batch - predict_session| = {diff.max():.6f}; sessions differing > 1e-6: {(diff > 1e-6).sum()}")
print(f"intensity differs in {sum(x != y for x, y in zip(b_int, s_int))} sessions")
print(f"weak skill differs in {sum(x != y for x, y in zip(b_ws, s_ws))} sessions")
refit = fit_grade_norms(train)
def _k(d):  # compare grade keys as ints, so "3" (saved) and 3 (fitted) count as the same grade
    return {c: {int(g): v for g, v in m.items()} for c, m in d.items()}
same_norms = (refit["global"] == art["norms"]["global"] and _k(refit["by_grade"]) == _k(art["norms"]["by_grade"]))
print(f"grade norms saved in the artifacts equal norms re-fitted on the train split: {same_norms}")
if not same_norms:
    print("  saved   :", {k: art['norms'][k] for k in ('global', 'by_grade')})
    print("  refitted:", {k: refit[k] for k in ('global', 'by_grade')})

if (diff > 1e-6).any():
    cols = art["difficulty_score"]["feature_columns"]
    X = test.reindex(columns=cols, fill_value=0)
    i = int(diff.argmax()); sid = test["session_id"].iloc[i]
    # replicate exactly what predict_session does to one session
    one = engineer_features(clean(pd.DataFrame([raw.loc[sid].to_dict()])), drop_first=False)
    one = apply_grade_norms(one, art["norms"]).reindex(columns=cols, fill_value=0)
    bad = [c for c in cols if not np.isclose(float(X.iloc[i][c]), float(one.iloc[0][c]), equal_nan=True)]
    print(f"worst session {sid}: batch {b_score[i]:.4f} vs predict_session {float(s_score[i]):.4f}")
    if bad:
        print("feature columns that differ for that session (batch vs single-row):")
        for c in bad:
            print(f"  {c}: batch={X.iloc[i][c]}  single-row={one.iloc[0][c]}")
    else:
        print("all model input columns are identical for that session -> the difference is inside the model call "
              "(e.g. column order / dtypes), not in feature engineering.")
else:
    print("OK: both paths agree, so the report tables and the UI show the same predictions.")
