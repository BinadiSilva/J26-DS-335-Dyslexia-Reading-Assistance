import sys
sys.path.insert(0, "backend/app/services/reading_difficulty")
import pandas as pd
from features import load_raw, clean, engineer_features
from profiling import (build_cognitive_profile, prosody_engagement_index,
                       load_calibration, CALIBRATION_PATH)

print("Calibration file used:", CALIBRATION_PATH, "| exists:", CALIBRATION_PATH.exists())

df = engineer_features(clean(load_raw("datasets/comp02_raw_reading_sessions.csv")))
cal = load_calibration()
assert "pitch_variability" in cal and "p5" in cal["pitch_variability"]

# Prosody index
idx = pd.Series([prosody_engagement_index(r, cal)["prosody_engagement_index"]
                 for _, r in df.iterrows()])
print("\nProsody index:", idx.describe().round(1).to_dict())
assert idx.between(0, 100).all(), "index out of range"
assert idx.std() > 5, "index barely varies"
print("corr(index, difficulty_score):", round(idx.corr(df["difficulty_score"], method="spearman"), 2))

# Cognitive profile (dataset's weak_skill label stands in for the ML prediction here)
profs = [build_cognitive_profile(r, r["weak_skill"], r["difficulty_level"])
         for _, r in df.iterrows()]
ns = pd.Series([len(p["strengths"]) for p in profs])
nw = pd.Series([len(p["weaknesses"]) for p in profs])
print("\nStrengths per row  :", ns.value_counts().sort_index().to_dict())
print("Weaknesses per row :", nw.value_counts().sort_index().to_dict())
print("Rows with no strengths:", (ns == 0).sum(), "| no weaknesses:", (nw == 0).sum())

rule = pd.Series([p["rule_based_weakest_area"] for p in profs])
print("\nRule-based weakest vs weak_skill label:")
print(pd.crosstab(df["weak_skill"].values, rule.values))
print("Agreement:", round(pd.Series([p["model_rule_agreement"] for p in profs]).mean(), 2))
print("\nSample profile:", profs[0])
print("\nProfiling checks passed.")