"""Preprocessing and feature engineering for the reading-difficulty service.

This module keeps all data cleaning in one place so the downstream models can reuse
exactly the same input schema and leakage rules. The main design choice is that the
weak-skill classifier must not see the subskill summary columns because those scores
were used to derive the target label in the synthetic data generator; they are valid
features for the difficulty and intervention models, but they would leak the answer
if included in the weak-skill model.
"""

from __future__ import annotations

import random
import warnings
from typing import Any

import pandas as pd

METADATA_COLS = [
    "child_id", "session_id", "session_date",
    "transcribed_text", "error_pattern_features",
]

SUBSKILL_COLS = [
    "phonological_score", "decoding_score",
    "fluency_subscore", "pronunciation_subscore",
]

TARGET_COLS = [
    "difficulty_score", "difficulty_level",
    "weak_skill", "intervention_intensity",
]

CATEGORICAL_COLS = ["gender", "passage_level", "passage_id"]

WEAK_SKILL_FEATURE_COLS = [
    "hesitation_count", "repetition_count",
    "omission_count", "substitution_count",
    "fluency_score", "pronunciation_accuracy",
    "word_error_rate",
]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as delivered, without any transformation.

    Keeping this function intentionally minimal ensures the rest of the pipeline can
    be built from a single source of truth and makes debugging easier when source-data
    drift occurs.
    """
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the session data while preserving the original schema.

    Numeric feature columns are imputed with their median so later model code sees a
    complete matrix without mutating the input data frame. Missing target values or a
    missing child id are treated as rows that should be dropped, because they cannot
    produce a valid supervised example.
    """
    cleaned = df.copy()

    feature_cols = [
        col for col in cleaned.columns
        if col not in set(METADATA_COLS) | set(TARGET_COLS)
    ]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            median_value = cleaned[col].median()
            if pd.notna(median_value):
                cleaned[col] = cleaned[col].fillna(median_value)

    rows_to_drop = cleaned[["child_id", *TARGET_COLS]].isna().any(axis=1)
    cleaned = cleaned.loc[~rows_to_drop].copy()

    if {"words_correct", "words_incorrect", "words_total"}.issubset(cleaned.columns):
        mismatch = (cleaned["words_correct"] + cleaned["words_incorrect"]) - cleaned["words_total"]
        bad_rows = mismatch.abs() > 1
        if bad_rows.any():
            warnings.warn(
                "Rows with inconsistent word counts were detected; "
                f"count={int(bad_rows.sum())} rows flagged.",
                UserWarning,
            )

    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded categorical columns without altering the underlying dataset.

    The dataset already contains normalized and variability features, so this function
    stays intentionally lightweight and only expands the categorical variables needed
    by the models. The goal is to make the later feature selectors trivial while
    preserving all original columns for traceability.
    """
    engineered = df.copy()

    for col in CATEGORICAL_COLS:
        if col in engineered.columns:
            dummies = pd.get_dummies(engineered[col], prefix=col, drop_first=True, dtype=int)
            engineered = pd.concat([engineered.drop(columns=[col]), dummies], axis=1)

    return engineered


def learner_level_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data at the learner level so all sessions for a child stay together.

    A row-level split would leak information from one session into another and
    artificially boost model performance. Grouping by child_id makes the evaluation
    reflect real-world generalization to unseen learners.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1.")

    children = df["child_id"].drop_duplicates().tolist()
    random.Random(seed).shuffle(children)

    test_count = max(1, round(len(children) * test_frac)) if len(children) > 1 else 1
    test_children = set(children[:test_count])
    train_children = set(children[test_count:])

    train_df = df[df["child_id"].isin(train_children)].copy()
    test_df = df[df["child_id"].isin(test_children)].copy()
    return train_df, test_df


def get_feature_cols(df: pd.DataFrame, exclude_subskills: bool = False) -> list[str]:
    """Return the model-input feature columns while keeping leakage rules centralized.

    Metadata and target columns are never learned from, and the weak-skill model
    explicitly excludes the four subskill summary scores because those scores were
    used to derive the target label in the synthetic data generator.
    """
    excluded = set(METADATA_COLS) | set(TARGET_COLS)
    feature_cols = [col for col in df.columns if col not in excluded]
    if exclude_subskills:
        feature_cols = [col for col in feature_cols if col not in SUBSKILL_COLS]
    return feature_cols


def get_xy_difficulty(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the difficulty regression dataset using all engineered feature columns.

    The difficulty model can legitimately use the subskill summaries because they are
    part of the feature set for predicting difficulty, not a leaked target proxy.
    """
    X = df[get_feature_cols(df, exclude_subskills=False)]
    y = df["difficulty_score"]
    return X, y


def get_xy_weak_skill(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the weak-skill classification dataset using only the raw features
    that actually drive the four sub-skill scores, rather than all engineered
    columns. This is a deliberate feature-selection choice (not the general
    leakage-avoidance rule from get_feature_cols) made specifically because
    weak_skill underperformed with the full feature set on this dataset size —
    see project notes. Still leakage-safe: none of these are the aggregated
    subskill_score columns themselves.
    """
    available_cols = [c for c in WEAK_SKILL_FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df["weak_skill"]
    return X, y


def get_xy_intervention(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the intervention classification dataset using the full engineered feature set.

    Intervention intensity is a separate target from weak_skill and is allowed to use
    the subskill columns because they are meaningful diagnostic features for planning
    support rather than a direct leakage of the weak-skill target.
    """
    X = df[get_feature_cols(df, exclude_subskills=False)]
    y = df["intervention_intensity"]
    return X, y
