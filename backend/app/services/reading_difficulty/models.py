"""Reusable modeling layer for the reading-difficulty service.

The project trains three distinct targets, each with a random forest, XGBoost, and
EBM model. Regression and classification are intentionally separated because the
metrics and cross-validation splitters differ: the continuous difficulty score uses
KFold regression scoring, whereas the 3- and 4-class targets require stratified
folds to preserve class balance during evaluation.
"""

from __future__ import annotations
from sklearn.preprocessing import LabelEncoder

from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from backend.app.services.reading_difficulty.features import (
    get_xy_difficulty,
    get_xy_intervention,
    get_xy_weak_skill,
)


def get_regression_models(seed: int = 42) -> dict[str, Any]:
    """Return fresh, unfitted regressor instances for the difficulty-score target.

    Each call should produce new model objects so target-specific training does not
    leak state between runs. The RF and XGBoost variants use a fixed random seed for
    reproducibility, while EBM gets a deterministic seed without any extra tuning.
    """
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            random_state=seed,
            objective="reg:squarederror",
        ),
        "EBM": ExplainableBoostingRegressor(random_state=seed),
    }


def get_classification_models(seed: int = 42) -> dict[str, Any]:
    """Return fresh, unfitted classifiers for the weak-skill and intervention targets.

    These models are reused across both classification tasks by calling this function
    once per target. That ensures each target receives its own freshly initialized
    model objects rather than shared fitted instances carrying state across training.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="mlogloss",
            random_state=seed,
            objective="multi:softprob",
            use_label_encoder=False,
        ),
        "EBM": ExplainableBoostingClassifier(random_state=seed),
    }


def get_weak_skill_models(seed: int = 42) -> dict[str, Any]:
    """Fresh, unfitted classifiers specifically for the weak_skill target, with
    tighter regularization than get_classification_models(). weak_skill is a
    4-class problem on ~96 training rows — the default hyperparameters (tuned
    against the better-signal intervention_intensity target) overfit here.
    Do not reuse this factory for intervention_intensity or vice versa.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            objective="multi:softprob",
        ),
        "EBM": ExplainableBoostingClassifier(
            random_state=seed,
            max_bins=128,
        ),
    }


def train_all(models: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Fit every model in the provided dictionary in place and return the same dict.

    All supported model classes in this project expose a uniform .fit(X, y) API, so
    this training helper can serve both regression and classification tasks without
    logic branches.
    """
    for model_name, model in models.items():
        models[model_name] = model.fit(X_train, y_train)
    return models


def cross_validate_regression(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Cross-validate regression models using KFold and RMSE/R2 scoring.

    Continuous targets are not stratified because ordering is irrelevant; the split
    should preserve overall data distribution without imposing class-like constraints.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        rmse_scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        rows.append(
            {
                "model": model_name,
                "cv_rmse_mean": float(rmse_scores.mean()),
                "cv_rmse_std": float(rmse_scores.std()),
                "cv_r2_mean": float(r2_scores.mean()),
                "cv_r2_std": float(r2_scores.std()),
            }
        )

    return pd.DataFrame(rows, columns=[
        "model",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
    ])


def cross_validate_classification(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Cross-validate classification models with stratified folds to preserve class balance.

    The synthetic dataset is small enough that class imbalance or fold imbalance can
    distort the metric, so StratifiedKFold keeps each fold representative of the
    target distribution.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        rows.append(
            {
                "model": model_name,
                "cv_f1_macro_mean": float(f1_scores.mean()),
                "cv_f1_macro_std": float(f1_scores.std()),
            }
        )

    return pd.DataFrame(rows, columns=["model", "cv_f1_macro_mean", "cv_f1_macro_std"])


def evaluate_holdout_regression(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate regression models on the held-out learner split.

    This focuses on calibration and accuracy of the numerical difficulty prediction,
    reporting RMSE, MAE, and R2 for direct comparison across model families.
    """
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        preds = model.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "mae": float(mean_absolute_error(y_test, preds)),
                "r2": float(r2_score(y_test, preds)),
            }
        )

    return pd.DataFrame(rows, columns=["model", "rmse", "mae", "r2"])


def evaluate_holdout_classification(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate classification models on the held-out learner split.

    Macro-averaged precision, recall, and F1 are used because the class counts are small
    and this makes the evaluation less sensitive to a single class dominating the dataset.
    """
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1_macro),
            }
        )

    return pd.DataFrame(rows, columns=["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"])


def confusion(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, model_name: str, labels: list[str]):
    """Return the confusion matrix for one named model using the caller's label order.

    The label order must be explicit because the intervention classes have a natural
    ranking of Low < Moderate < High, and passing that order preserves interpretability
    for reporting.
    """
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in provided dict.")

    preds = models[model_name].predict(X_test)
    return confusion_matrix(y_test, preds, labels=labels)


def select_best_regression(cv_report: pd.DataFrame) -> str:
    """Return the model name with the lowest cross-validated RMSE."""
    if cv_report.empty:
        raise ValueError("cv_report is empty; no model comparison available.")
    return str(cv_report.loc[cv_report["cv_rmse_mean"].idxmin(), "model"])


def select_best_classification(cv_report: pd.DataFrame) -> str:
    """Return the model name with the highest cross-validated macro F1."""
    if cv_report.empty:
        raise ValueError("cv_report is empty; no model comparison available.")
    return str(cv_report.loc[cv_report["cv_f1_macro_mean"].idxmax(), "model"])


def train_and_evaluate_all(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42) -> dict[str, dict[str, Any]]:
    """Train and evaluate all three targets with their matching model families.

    The function is intentionally orchestration-heavy: it resolves the correct target
    features and labels via the feature module, trains fresh models, then reports
    both CV and hold-out metrics. This keeps the target logic centralized in one place
    instead of duplicating selection rules across the project.
    """
    results: dict[str, dict[str, Any]] = {}

    difficulty_X_train, difficulty_y_train = get_xy_difficulty(train_df)
    difficulty_X_test, difficulty_y_test = get_xy_difficulty(test_df)
    difficulty_models = get_regression_models(seed=seed)
    difficulty_models = train_all(difficulty_models, difficulty_X_train, difficulty_y_train)
    difficulty_cv = cross_validate_regression(difficulty_models, difficulty_X_train, difficulty_y_train, seed=seed)
    difficulty_holdout = evaluate_holdout_regression(difficulty_models, difficulty_X_test, difficulty_y_test)
    results["difficulty_score"] = {
        "models": difficulty_models,
        "cv_report": difficulty_cv,
        "holdout_report": difficulty_holdout,
        "best_model_name": select_best_regression(difficulty_cv),
    }

    WEAK_SKILL_CLASSES = ["phonological", "decoding", "fluency", "pronunciation"]
    weak_skill_class_map = {label: idx for idx, label in enumerate(WEAK_SKILL_CLASSES)}
    weak_skill_X_train, weak_skill_y_train_raw = get_xy_weak_skill(train_df)
    weak_skill_X_test, weak_skill_y_test_raw = get_xy_weak_skill(test_df)
    weak_skill_y_train = weak_skill_y_train_raw.map(weak_skill_class_map)
    weak_skill_y_test = weak_skill_y_test_raw.map(weak_skill_class_map)
    weak_skill_models = get_weak_skill_models(seed=seed)
    weak_skill_models = train_all(weak_skill_models, weak_skill_X_train, weak_skill_y_train)
    weak_skill_cv = cross_validate_classification(weak_skill_models, weak_skill_X_train, weak_skill_y_train, seed=seed)
    weak_skill_holdout = evaluate_holdout_classification(weak_skill_models, weak_skill_X_test, weak_skill_y_test)
    results["weak_skill"] = {
        "models": weak_skill_models,
        "cv_report": weak_skill_cv,
        "holdout_report": weak_skill_holdout,
        "best_model_name": select_best_classification(weak_skill_cv),
        "class_order": WEAK_SKILL_CLASSES,
    }

    INTERVENTION_CLASSES = ["Low", "Moderate", "High"]
    intervention_class_map = {label: idx for idx, label in enumerate(INTERVENTION_CLASSES)}
    intervention_X_train, intervention_y_train_raw = get_xy_intervention(train_df)
    intervention_X_test, intervention_y_test_raw = get_xy_intervention(test_df)
    intervention_y_train = intervention_y_train_raw.map(intervention_class_map)
    intervention_y_test = intervention_y_test_raw.map(intervention_class_map)
    intervention_models = get_classification_models(seed=seed)
    intervention_models = train_all(intervention_models, intervention_X_train, intervention_y_train)
    intervention_cv = cross_validate_classification(intervention_models, intervention_X_train, intervention_y_train, seed=seed)
    intervention_holdout = evaluate_holdout_classification(intervention_models, intervention_X_test, intervention_y_test)
    results["intervention_intensity"] = {
        "models": intervention_models,
        "cv_report": intervention_cv,
        "holdout_report": intervention_holdout,
        "best_model_name": select_best_classification(intervention_cv),
        "class_order": INTERVENTION_CLASSES,
    }

    return results
