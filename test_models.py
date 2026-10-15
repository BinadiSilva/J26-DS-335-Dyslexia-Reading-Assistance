from pathlib import Path

from backend.app.services.reading_difficulty.features import (
    load_raw, clean, engineer_features, stratified_learner_level_split,
    fit_grade_norms, apply_grade_norms,
)
from backend.app.services.reading_difficulty.models import train_and_evaluate_all

candidates = [
    Path('datasets/comp02_raw_reading_sessions.csv'),
    Path('datasets/comp02_synthetic_reading_sessions.csv'),
]
raw_path = next((p for p in candidates if p.exists()), candidates[0])

df = load_raw(str(raw_path))
df = clean(df)
df = engineer_features(df)  # stateless derived features only (safe pre-split)

# Stratified by intervention_intensity (its 'Moderate' class has only 2 children
# total in this dataset -- a plain random child split can easily miss it entirely
# in the test set, silently turning a 3-class evaluation into an easier 2-class one).
train_df, test_df = stratified_learner_level_split(df, stratify_col="intervention_intensity")

# Cohort norms MUST be fit on train only, then applied to both -- fitting on the
# full/combined dataset before splitting would leak test-set distribution into
# training, the same category of leakage the child_id split already guards against.
norms = fit_grade_norms(train_df)
train_df = apply_grade_norms(train_df, norms)
test_df = apply_grade_norms(test_df, norms)

print("Starting training for all 3 targets...")
results = train_and_evaluate_all(train_df, test_df)

for target in ['difficulty_score', 'weak_skill', 'intervention_intensity']:
    r = results[target]
    print(f"\n=== {target} ===")
    print("CV report:")
    print(r['cv_report'])
    print("Holdout report:")
    print(r['holdout_report'])
    print("Best model:", r['best_model_name'])

print("\nDone.")