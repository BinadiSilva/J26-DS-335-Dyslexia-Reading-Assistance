import json
from pathlib import Path

from backend.app.services.reading_difficulty.features import load_raw
from backend.app.services.reading_difficulty.intervention import build_intervention
from backend.app.services.reading_difficulty.pipeline import (
    append_session_log,
    load_artifacts,
    load_history,
    longitudinal_summary,
    predict_session,
    train_pipeline,
)

raw_path = Path("datasets/comp02_raw_reading_sessions.csv")
artifacts_path = Path("outputs/artifacts.joblib")

# 1) training pipeline works once, then reuses the saved artifact bundle.
if artifacts_path.exists():
    artifacts = load_artifacts(artifacts_path)
else:
    artifacts = train_pipeline(str(raw_path))
    artifacts = load_artifacts(artifacts_path)

assert isinstance(artifacts, dict)
assert "difficulty_score" in artifacts
assert "weak_skill" in artifacts
assert "intervention_intensity" in artifacts
assert "norms" in artifacts

# 2) predict_session for held-out rows returns complete JSON-serialisable schema
raw_df = load_raw(str(raw_path))
weak_skill_labels = ["phonological", "decoding", "fluency", "pronunciation"]
intervention_labels = ["Low", "Moderate", "High"]
print("weak_skill label set:", weak_skill_labels)
print("intervention label set:", intervention_labels)

results = [predict_session(artifacts, raw_df.iloc[[i]]) for i in range(min(5, len(raw_df)))]
for idx, result in enumerate(results):
    assert result["weak_skill"] in weak_skill_labels, result["weak_skill"]
    assert result["intervention_intensity"] in intervention_labels, result["intervention_intensity"]

    true_row = raw_df.iloc[idx]
    print(
        "match check",
        idx,
        {
            "weak_skill": result["weak_skill"] == true_row["weak_skill"],
            "intervention_intensity": result["intervention_intensity"] == true_row["intervention_intensity"],
        },
    )

    schema_keys = {
        "child_id",
        "session_id",
        "timestamp",
        "difficulty_score",
        "difficulty_level",
        "weak_skill",
        "weak_skill_confidence",
        "intervention_intensity",
        "intervention_confidence",
        "level_consistency",
        "low_confidence_flags",
        "shap_top_features",
        "cognitive_profile",
        "prosody_engagement_index",
        "intervention",
        "model_versions",
    }
    assert set(result.keys()) == schema_keys, result.keys()
    assert "recommended_activities" not in result["intervention"]
    json.dumps(result)

# 3) low-confidence intervention review reasons are triggered
low_confidence = build_intervention(
    intensity="High",
    confidence=0.3,
    difficulty_level="High",
    profile={
        "primary_focus_area": "phonological",
        "weaknesses": ["phonological", "decoding", "fluency", "pronunciation"],
    },
    prosody={"prosody_engagement_index": 55.0},
    weak_skill_confidence=0.5,
)
assert "recommended_activities" not in low_confidence
assert low_confidence["requires_professional_review"] is True
assert any("low confidence in weak-skill prediction" in reason.lower() for reason in low_confidence["review_reasons"])
assert any("low confidence" in reason.lower() for reason in low_confidence["review_reasons"])
assert any("high intensity" in reason.lower() for reason in low_confidence["review_reasons"])

weak_skill_low_confidence_session = {
    "primary_focus_area": "phonological",
    "weaknesses": ["phonological"],
}
low_skill_review = build_intervention(
    intensity="Low",
    confidence=0.95,
    difficulty_level="Low",
    profile=weak_skill_low_confidence_session,
    prosody={"prosody_engagement_index": 55.0},
    weak_skill_confidence=0.59,
)
assert "recommended_activities" not in low_skill_review
assert low_skill_review["requires_professional_review"] is True
assert any("low confidence in weak-skill prediction" in reason.lower() for reason in low_skill_review["review_reasons"]) 


# 4) history logging and summary
log_path = Path("outputs/test_session_log.jsonl")
if log_path.exists():
    log_path.unlink()

for i, row in enumerate(raw_df.iloc[:3].itertuples(index=False), start=1):
    result = predict_session(artifacts, row._asdict())
    result["child_id"] = "TEST_CHILD"
    result["session_id"] = f"TEST_CHILD-S{i:02d}"
    result["timestamp"] = f"2026-08-0{i}"
    append_session_log(result, path=str(log_path))

history = load_history("TEST_CHILD", str(log_path))
assert len(history) == 3
assert [e["timestamp"] for e in history] == sorted(e["timestamp"] for e in history)
summary = longitudinal_summary(history)
assert summary["n_sessions"] == 3
assert summary["trend_label"] in {"improving", "stable", "worsening"}

# 5) print one full example JSON result
print(json.dumps(results[0], indent=2, default=str))
print("Pipeline checks passed.")
