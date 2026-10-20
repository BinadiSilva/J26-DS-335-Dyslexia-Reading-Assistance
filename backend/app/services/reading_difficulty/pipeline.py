from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.services.reading_difficulty.explain import LOW_CONFIDENCE_THRESHOLD, explain_prediction
from backend.app.services.reading_difficulty.features import (
    apply_grade_norms,
    clean,
    engineer_features,
    fit_grade_norms,
    get_feature_cols,
    get_xy_difficulty,
    get_xy_intervention,
    get_xy_weak_skill,
    learner_level_split,
    load_raw,
    stratified_learner_level_split,
)
from backend.app.services.reading_difficulty.intervention import build_intervention
from backend.app.services.reading_difficulty.models import (
    select_best_classification,
    select_best_regression,
    train_and_evaluate_all,
)
from backend.app.services.reading_difficulty.profiling import build_cognitive_profile, prosody_engagement_index


def _as_python(value: Any) -> Any:
    """Recursively convert numpy scalar/array values to native Python objects."""
    if isinstance(value, dict):
        return {str(k): _as_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_as_python(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _difficulty_level_from_score(score: float) -> str:
    if score < 0.26:
        return "Low"
    if score < 0.50:
        return "Moderate"
    return "High"


def _feature_columns_for_target(target: str, df: pd.DataFrame) -> list[str]:
    target = str(target)
    if target == "difficulty_score":
        return get_feature_cols(df, exclude_subskills=False)
    if target == "weak_skill":
        cols = [c for c in [
            "hesitation_count", "repetition_count", "omission_count",
            "substitution_count", "fluency_score", "pronunciation_accuracy",
            "word_error_rate",
        ] if c in df.columns]
        return cols
    if target == "intervention_intensity":
        return get_feature_cols(df, exclude_subskills=False)
    raise ValueError(f"Unsupported target: {target}")


class PipelineArtifacts(dict):
    def save_artifacts(self, path: str | Path) -> None:
        save_artifacts(self, path)

    @classmethod
    def load_artifacts(cls, path: str | Path) -> "PipelineArtifacts":
        return load_artifacts(path)


def save_artifacts(artifacts: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_as_python(artifacts), str(out_path))


def load_artifacts(path: str | Path) -> PipelineArtifacts:
    loaded = joblib.load(str(path))
    if not isinstance(loaded, dict):
        raise TypeError("Loaded artifact is not a dictionary-like object.")
    return PipelineArtifacts(loaded)


def _decode_class_label(raw_prediction: Any, class_order: list[str]) -> str:
    """Convert integer-encoded classes back to the original label names."""
    if raw_prediction is None:
        raise ValueError("raw_prediction is None; cannot decode label.")

    if isinstance(raw_prediction, str):
        return raw_prediction

    try:
        idx = int(raw_prediction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not decode class prediction {raw_prediction!r} using class_order={class_order!r}.") from exc

    if not 0 <= idx < len(class_order):
        raise ValueError(
            f"Decoded class index {idx} is out of range for class_order={class_order!r}. "
            f"Expected an index in [0, {len(class_order)-1}]."
        )

    return str(class_order[idx])


def train_pipeline(raw_csv: str | Path) -> PipelineArtifacts:
    seed = 42
    raw_df = load_raw(str(raw_csv))
    cleaned = clean(raw_df)
    engineered = engineer_features(cleaned)
    train_df, test_df = stratified_learner_level_split(engineered, stratify_col="intervention_intensity")
    norms = fit_grade_norms(train_df)
    train_df = apply_grade_norms(train_df, norms)
    test_df = apply_grade_norms(test_df, norms)

    trained = train_and_evaluate_all(train_df, test_df, seed=seed)

    artifacts = PipelineArtifacts()
    artifacts["norms"] = norms
    artifacts["feature_columns"] = {}
    artifacts["class_labels"] = {}
    artifacts["label_maps"] = {}
    artifacts["train_rows"] = int(len(train_df))
    artifacts["test_rows"] = int(len(test_df))
    artifacts["model_versions"] = {}
    artifacts["training_metadata"] = {
        "training_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "train_rows": int(len(train_df)),
        "library_versions": {
            "python": __import__("platform").python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "interpret": version("interpret-core"),
        },
    }

    for target in ["difficulty_score", "weak_skill", "intervention_intensity"]:
        report = trained[target]
        if target == "difficulty_score":
            best_name = select_best_regression(report["cv_report"])
            model = report["models"][best_name]
            labels = []
            label_map = {}
        else:
            best_name = select_best_classification(report["cv_report"])
            model = report["models"][best_name]
            labels = list(report.get("class_order", []))
            label_map = {idx: label for idx, label in enumerate(labels)}

        feature_cols = _feature_columns_for_target(target, train_df)
        artifacts["feature_columns"][target] = feature_cols
        artifacts["class_labels"][target] = labels
        artifacts["label_maps"][target] = label_map
        artifacts["model_versions"][target] = best_name
        artifacts[target] = {
            "model": model,
            "best_model_name": best_name,
            "feature_columns": feature_cols,
            "class_labels": labels,
            "label_map": label_map,
            "cv_report": _as_python(report["cv_report"]),
            "holdout_report": _as_python(report["holdout_report"]),
        }

    save_artifacts(artifacts, "outputs/artifacts.joblib")
    return artifacts


def _prepare_session_frame(raw_session_df: pd.DataFrame | pd.Series | dict[str, Any]) -> pd.DataFrame:
    if isinstance(raw_session_df, dict):
        frame = pd.DataFrame([raw_session_df])
    elif isinstance(raw_session_df, pd.Series):
        frame = raw_session_df.to_frame().T
    elif isinstance(raw_session_df, pd.DataFrame):
        frame = raw_session_df.copy()
    else:
        raise TypeError("raw_session_df must be a DataFrame, Series, or dict row.")

    if frame.empty:
        raise ValueError("Session input is empty.")
    if len(frame) > 1:
        raise ValueError("predict_session expects exactly one session row.")
    return frame


def _align_to_feature_columns(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[feature_columns]


def predict_session(artifacts: dict[str, Any], raw_session_df: pd.DataFrame | pd.Series | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifacts, dict):
        raise TypeError("artifacts must be a dictionary-like training artifact set.")

    row = _prepare_session_frame(raw_session_df)
    cleaned = clean(row)
    engineered = engineer_features(cleaned, drop_first=False)
    engineered = apply_grade_norms(engineered, artifacts["norms"])

    difficulty_target = artifacts["difficulty_score"]
    difficulty_model = difficulty_target["model"]
    difficulty_features = difficulty_target["feature_columns"]
    difficulty_X = _align_to_feature_columns(engineered, difficulty_features)
    difficulty_score = float(difficulty_model.predict(difficulty_X)[0])
    difficulty_level = _difficulty_level_from_score(difficulty_score)

    weak_skill_target = artifacts["weak_skill"]
    weak_skill_model = weak_skill_target["model"]
    weak_skill_features = weak_skill_target["feature_columns"]
    weak_skill_X = _align_to_feature_columns(engineered, weak_skill_features)
    weak_skill_raw_pred = weak_skill_model.predict(weak_skill_X)[0]
    weak_skill_pred = _decode_class_label(weak_skill_raw_pred, artifacts["weak_skill"]["class_labels"])
    weak_skill_confidence = float(np.max(weak_skill_model.predict_proba(weak_skill_X)[0]))

    intervention_target = artifacts["intervention_intensity"]
    intervention_model = intervention_target["model"]
    intervention_features = intervention_target["feature_columns"]
    intervention_X = _align_to_feature_columns(engineered, intervention_features)
    intervention_raw_pred = intervention_model.predict(intervention_X)[0]
    intervention_pred = _decode_class_label(intervention_raw_pred, artifacts["intervention_intensity"]["class_labels"])
    intervention_confidence = float(np.max(intervention_model.predict_proba(intervention_X)[0]))

    cognitive_profile = build_cognitive_profile(engineered.iloc[0], str(weak_skill_pred), difficulty_level)
    prosody = prosody_engagement_index(engineered.iloc[0], artifacts["norms"].get("calibration"))

    intervention = build_intervention(
        intensity=str(intervention_pred),
        confidence=intervention_confidence,
        difficulty_level=difficulty_level,
        profile=cognitive_profile,
        prosody=prosody,
        weak_skill_confidence=weak_skill_confidence,
    )

    shap_top_features = {}
    for target, model_name in {
        "difficulty_score": difficulty_model,
        "weak_skill": weak_skill_model,
        "intervention_intensity": intervention_model,
    }.items():
        feature_col_list = {
            "difficulty_score": difficulty_features,
            "weak_skill": weak_skill_features,
            "intervention_intensity": intervention_features,
        }[target]
        aligned_row = _align_to_feature_columns(engineered, feature_col_list).iloc[[0]]
        explanation = explain_prediction(model_name, aligned_row, feature_col_list, top_n=5)
        shap_top_features[target] = [[k, float(v)] for k, v in explanation["top_features"]]

    level_consistency = {
        "difficulty_level": difficulty_level,
        "support_level": str(intervention_pred),
        "consistent": str(intervention_pred) == difficulty_level,
    }
    if not level_consistency["consistent"]:
        intervention = dict(intervention)
        if not intervention.get("review_reasons"):
            intervention["review_reasons"] = []
        if "difficulty level and support intensity disagree" not in intervention["review_reasons"]:
            intervention["review_reasons"].append("difficulty level and support intensity disagree")
        intervention["requires_professional_review"] = True

    result = {
        "child_id": str(engineered.iloc[0].get("child_id", "unknown")),
        "session_id": str(engineered.iloc[0].get("session_id", "unknown_session")),
        "timestamp": str(engineered.iloc[0].get("session_date", engineered.iloc[0].get("timestamp", ""))),
        "difficulty_score": float(difficulty_score),
        "difficulty_level": difficulty_level,
        "weak_skill": str(weak_skill_pred),
        "weak_skill_confidence": float(weak_skill_confidence),
        "intervention_intensity": str(intervention_pred),
        "intervention_confidence": float(intervention_confidence),
        "level_consistency": level_consistency,
        "low_confidence_flags": {
            "weak_skill": bool(weak_skill_confidence < LOW_CONFIDENCE_THRESHOLD),
            "intervention_intensity": bool(intervention_confidence < LOW_CONFIDENCE_THRESHOLD),
        },
        "shap_top_features": shap_top_features,
        "cognitive_profile": cognitive_profile,
        "prosody_engagement_index": prosody,
        "intervention": intervention,
        "model_versions": {
            "difficulty_score": difficulty_target["best_model_name"],
            "weak_skill": weak_skill_target["best_model_name"],
            "intervention_intensity": intervention_target["best_model_name"],
        },
    }
    return _as_python(result)


def append_session_log(result: dict[str, Any], path: str | Path = "outputs/session_log.jsonl") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_as_python(result), default=str) + "\n")


def load_history(child_id: str, path: str | Path = "outputs/session_log.jsonl") -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("child_id")) == str(child_id):
                records.append(item)

    return sorted(records, key=lambda r: str(r.get("timestamp", "1970-01-01")))


def longitudinal_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "n_sessions": 0,
            "difficulty_score_series": [],
            "delta_vs_previous": 0.0,
            "trend_label": "stable",
            "most_frequent_weak_skill": None,
            "intensity_history": [],
        }

    scores = [float(item.get("difficulty_score", 0.0)) for item in history]
    recent = scores[-3:]
    delta_vs_previous = float(scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0

    if len(recent) >= 2:
        change = float(recent[-1] - recent[0])
        if change <= -0.05:
            trend_label = "improving"
        elif change >= 0.05:
            trend_label = "worsening"
        else:
            trend_label = "stable"
    else:
        trend_label = "stable"

    weak_skill_counts = Counter(str(item.get("weak_skill", "unknown")) for item in history)
    most_frequent_weak_skill = weak_skill_counts.most_common(1)[0][0] if weak_skill_counts else None

    return {
        "n_sessions": len(history),
        "difficulty_score_series": scores,
        "delta_vs_previous": delta_vs_previous,
        "trend_label": trend_label,
        "most_frequent_weak_skill": most_frequent_weak_skill,
        "intensity_history": [str(item.get("intervention_intensity", "Unknown")) for item in history],
    }


__all__ = [
    "PipelineArtifacts",
    "append_session_log",
    "load_artifacts",
    "load_history",
    "longitudinal_summary",
    "predict_session",
    "save_artifacts",
    "train_pipeline",
]
