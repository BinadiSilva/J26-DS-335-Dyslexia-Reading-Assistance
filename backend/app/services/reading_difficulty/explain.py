"""Explainability layer for the reading-difficulty service.

Two separate explanation mechanisms are used on purpose:
- RandomForest and XGBoost (tree ensembles, not inherently interpretable) are
  explained via SHAP's TreeExplainer, which approximates each feature's
  contribution to a specific prediction.
- EBM (Explainable Boosting Machine) is a glass-box model by design -- it already
  knows its own per-feature contributions exactly, with no approximation needed.
  Wrapping it in SHAP as well would be redundant, slower, and would throw away the
  exact contributions EBM already computes in favor of an approximation of them.
  This is a deliberate methodology choice, not an oversight: state it as such if
  asked why SHAP isn't used uniformly across all three model families.

Both paths return the same shape -- a list of (feature_name, contribution) tuples,
sorted by absolute contribution descending -- so callers (pipeline.py, the
dashboard) never need to know which mechanism produced a given explanation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

LOW_CONFIDENCE_THRESHOLD = 0.6  # named constant, not a magic number inline

_TREE_MODEL_TYPES = (RandomForestClassifier, RandomForestRegressor, XGBClassifier, XGBRegressor)
_EBM_MODEL_TYPES = (ExplainableBoostingClassifier, ExplainableBoostingRegressor)
_CLASSIFIER_TYPES = (RandomForestClassifier, XGBClassifier, ExplainableBoostingClassifier)


def shap_top_features_classifier(
    model: Any, X_row: pd.DataFrame, feature_names: list[str], top_n: int = 5
) -> list[tuple[str, float]]:
    """SHAP explanation for a single row from a RandomForestClassifier or
    XGBClassifier. Explains the PREDICTED class specifically -- defaulting to
    class 0 (a realistic mistake here) would silently explain the wrong outcome
    whenever the model didn't predict class 0 for this row.
    """
    predicted_class_idx = int(np.argmax(model.predict_proba(X_row)[0]))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    # SHAP's multiclass tree output shape is inconsistent across versions:
    # sometimes a list of per-class arrays, sometimes a single 3D array
    # (n_samples, n_features, n_classes). Handle both explicitly.
    if isinstance(shap_values, list):
        row_values = shap_values[predicted_class_idx][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            row_values = arr[0, :, predicted_class_idx]
        else:
            row_values = arr[0]

    pairs = list(zip(feature_names, row_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [(name, float(value)) for name, value in pairs[:top_n]]


def shap_top_features_regressor(
    model: Any, X_row: pd.DataFrame, feature_names: list[str], top_n: int = 5
) -> list[tuple[str, float]]:
    """SHAP explanation for a single row from a RandomForestRegressor or
    XGBRegressor. Regressors have no class dimension, so this is simpler than
    the classifier path -- shap_values() returns one array directly.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    row_values = np.asarray(shap_values)[0]

    pairs = list(zip(feature_names, row_values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [(name, float(value)) for name, value in pairs[:top_n]]


def ebm_top_features(ebm_model: Any, X_row: pd.DataFrame, top_n: int = 5) -> list[tuple[str, float]]:
    """Native EBM explanation for a single row -- NOT SHAP. EBM's own
    explain_local() gives exact per-feature contributions (it's glass-box by
    design), so no approximation step is needed or used here.

    For a MULTICLASS classifier (weak_skill: 4 classes, intervention_intensity:
    3 classes), each feature's score from explain_local() is itself a per-class
    array, not a single scalar -- EBM warns about this at fit time ("Detected
    multiclass problem... Multiclass interactions only have local explanations").
    This resolves each feature down to the PREDICTED class's specific
    contribution, mirroring the SHAP classifier path's same rule (explain the
    class actually predicted, not class 0 by default and not some blend across
    classes, either of which would misrepresent which outcome the contribution
    actually supports).
    """
    local_explanation = ebm_model.explain_local(X_row)
    data = local_explanation.data(0)  # index 0: the only row, since X_row is single-row
    names = data["names"]
    scores = data["scores"]

    predicted_class_idx = None
    if hasattr(ebm_model, "predict_proba"):
        predicted_class_idx = int(np.argmax(ebm_model.predict_proba(X_row)[0]))

    resolved_scores: list[float] = []
    for score in scores:
        arr = np.asarray(score)
        if arr.ndim > 0 and arr.size > 1:
            idx = predicted_class_idx if predicted_class_idx is not None else 0
            resolved_scores.append(float(arr[idx]))
        else:
            resolved_scores.append(float(arr))

    pairs = list(zip(names, resolved_scores))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [(name, value) for name, value in pairs[:top_n]]


def get_top_features(
    model: Any, X_row: pd.DataFrame, feature_names: list[str], top_n: int = 5
) -> list[tuple[str, float]]:
    """Router: picks SHAP (tree models) or EBM's native explanation, based on the
    model's actual type. Raises rather than silently guessing for any model type
    not covered by this project (RandomForest, XGBoost, EBM).
    """
    if isinstance(model, _EBM_MODEL_TYPES):
        return ebm_top_features(model, X_row, top_n=top_n)

    if isinstance(model, _TREE_MODEL_TYPES):
        if isinstance(model, _CLASSIFIER_TYPES):
            return shap_top_features_classifier(model, X_row, feature_names, top_n=top_n)
        return shap_top_features_regressor(model, X_row, feature_names, top_n=top_n)

    raise ValueError(
        f"get_top_features: unsupported model type {type(model)!r}. "
        "Expected RandomForest/XGBoost (SHAP path) or EBM (native path)."
    )


def classification_confidence(model: Any, X_row: pd.DataFrame) -> float:
    """Max class probability for a classifier's prediction on a single row.
    Works uniformly across RF/XGBoost/EBM classifiers, since all three expose
    predict_proba(). Meaningless for the difficulty_score regressor -- there is
    no equivalent notion of "confidence" for a continuous prediction, and this
    function must never be called on a regressor (see explain_prediction, which
    routes around it for models with no predict_proba).
    """
    proba = model.predict_proba(X_row)[0]
    return float(np.max(proba))


def is_low_confidence(confidence: float, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
    """Simple threshold check, factored out on its own so the threshold is
    configurable and this logic is testable without a real model."""
    return confidence < threshold


def explain_prediction(
    model: Any, X_row: pd.DataFrame, feature_names: list[str], top_n: int = 5
) -> dict[str, Any]:
    """Single entry point: top contributing features plus confidence/low-confidence
    flag (for classifiers only). Regressors (difficulty_score) have no
    predict_proba, so confidence/low_confidence are explicitly None rather than a
    fabricated stand-in -- regression uncertainty (e.g. prediction interval width)
    is a different concept from classification confidence and conflating them
    would misrepresent the output.
    """
    top_features = get_top_features(model, X_row, feature_names, top_n=top_n)

    if hasattr(model, "predict_proba"):
        confidence = classification_confidence(model, X_row)
        low_confidence = is_low_confidence(confidence)
    else:
        confidence = None
        low_confidence = None

    return {
        "top_features": top_features,
        "confidence": confidence,
        "low_confidence": low_confidence,
    }