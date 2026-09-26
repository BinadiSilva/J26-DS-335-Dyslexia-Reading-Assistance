from pathlib import Path

from backend.app.services.reading_difficulty.features import (
    load_raw, clean, engineer_features, stratified_learner_level_split,
    fit_grade_norms, apply_grade_norms,
    get_xy_difficulty, get_xy_weak_skill, get_xy_intervention,
)
from backend.app.services.reading_difficulty.models import train_and_evaluate_all
from backend.app.services.reading_difficulty.explain import explain_prediction

candidates = [
    Path("datasets/comp02_raw_reading_sessions.csv"),
    Path("datasets/comp02_synthetic_reading_sessions.csv"),
]
raw_path = next((p for p in candidates if p.exists()), candidates[0])

df = load_raw(str(raw_path))
df = clean(df)
df = engineer_features(df)

train_df, test_df = stratified_learner_level_split(df, stratify_col="intervention_intensity")
norms = fit_grade_norms(train_df)
train_df = apply_grade_norms(train_df, norms)
test_df = apply_grade_norms(test_df, norms)

results = train_and_evaluate_all(train_df, test_df)

# --- difficulty_score: regressor, RF and EBM ---
X_diff_test = get_xy_difficulty(test_df)[0]
row = X_diff_test.iloc[[0]]
for name in ["RandomForest", "EBM"]:
    model = results["difficulty_score"]["models"][name]
    out = explain_prediction(model, row, list(X_diff_test.columns))
    print(f"difficulty_score / {name}:", out)
    assert out["confidence"] is None, f"{name} regressor should have confidence=None"
    assert len(out["top_features"]) <= 5

# --- weak_skill: classifier, RF and EBM ---
X_ws_test = get_xy_weak_skill(test_df)[0]
row = X_ws_test.iloc[[0]]
for name in ["RandomForest", "EBM"]:
    model = results["weak_skill"]["models"][name]
    out = explain_prediction(model, row, list(X_ws_test.columns))
    print(f"weak_skill / {name}:", out)
    assert isinstance(out["confidence"], float)
    assert isinstance(out["low_confidence"], bool)
    assert len(out["top_features"]) <= 5

# --- intervention_intensity: classifier, XGBoost (tests SHAP multiclass path) ---
X_int_test = get_xy_intervention(test_df)[0]
row = X_int_test.iloc[[0]]
model = results["intervention_intensity"]["models"]["XGBoost"]
out = explain_prediction(model, row, list(X_int_test.columns))
print("intervention_intensity / XGBoost:", out)
assert isinstance(out["confidence"], float)

print("\nAll explainability checks passed.")
