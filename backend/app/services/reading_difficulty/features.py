"""Preprocessing and feature engineering for the reading-difficulty service.

Design notes:
- Cohort-normalized features (normalized_speed/fluency/accuracy) are fit-then-applied,
  not computed in one pass over the whole dataset. Norms must be fit on the TRAINING
  split only and then applied to both train and test -- computing them from the full
  dataset (train+test combined) would leak test-set distribution into training, the
  same category of problem the child_id-level split already guards against.
- The weak-skill classifier's feature list is deliberately restricted to the raw
  counts/scores that actually drive the four sub-skill scores in the data generator,
  not the full engineered feature set -- see WEAK_SKILL_FEATURE_COLS below.
"""

from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np
import pandas as pd

METADATA_COLS = [
    "child_id", "session_id", "session_date",
    "transcribed_text", "pitch_samples", "energy_samples", "pause_durations",
    "error_pattern_features",  # human-readable tag string -- description only,
                                 # never a model feature (see error_micro_profile
                                 # for the actual low-cardinality model feature)
]

SUBSKILL_COLS = [
    "phonological_score", "decoding_score",
    "fluency_subscore", "pronunciation_subscore",
]

TARGET_COLS = [
    "difficulty_score", "difficulty_level",
    "weak_skill", "intervention_intensity",
]

CATEGORICAL_COLS = ["gender", "passage_level", "passage_id", "error_micro_profile"]

# Restricted to the raw features that actually drive the four sub-skill scores in
# the generator (phonological <- hesitation/repetition; decoding <- omission/
# substitution; fluency <- fluency_score; pronunciation <- pronunciation_accuracy),
# plus word_error_rate as a general severity signal. Deliberately excludes prosody
# (pitch/energy/pause) and normalized_speed -- those have no real relationship to
# which subskill is weakest and were shown to dilute this target's signal.
WEAK_SKILL_FEATURE_COLS = [
    "hesitation_count", "repetition_count",
    "omission_count", "substitution_count",
    "fluency_score", "pronunciation_accuracy",
    "word_error_rate",
]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as delivered, without any transformation."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the session data while preserving the original schema."""
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


def _parse_sample_string(raw_value: Any) -> list[float]:
    """Return a cleaned numeric list from a semicolon-delimited sample string."""
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, str):
        return [float(v.strip()) for v in raw_value.split(";") if v.strip()]
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        return [float(v) for v in raw_value]
    return []


def _safe_cv(values: list[float]) -> float:
    """Coefficient of variation (std/mean) for a sample list; 0 if too short or mean=0."""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean_val = float(arr.mean())
    if mean_val == 0:
        return 0.0
    return float(arr.std(ddof=1) / abs(mean_val))


def _error_micro_profile(row) -> str:
    """Low-cardinality error-pattern grouping (proposal: 'grouped ... into micro-
    profiles to reveal unique error combinations'). Deliberately coarse -- a
    combinatorial per-count high/low/none tagging (as an earlier version did) can
    produce 100+ distinct strings on a 120-row dataset, meaning one-hot encoding
    would create near-unique dummy columns per row and add noise, not signal.
    This buckets by (a) overall severity tier and (b) which error family
    (phonological-type: hesitation+repetition, vs decoding-type: omission+
    substitution) dominates, giving at most ~9 categories regardless of dataset
    size."""
    total = (row.hesitation_count + row.repetition_count
             + row.omission_count + row.substitution_count + row.insertion_count)
    if total <= 3:
        severity = "low"
    elif total <= 7:
        severity = "moderate"
    else:
        severity = "high"

    phon_component = row.hesitation_count + row.repetition_count
    dec_component = row.omission_count + row.substitution_count
    if severity == "low":
        dominant = "balanced"
    elif phon_component > dec_component:
        dominant = "phonological_leaning"
    elif dec_component > phon_component:
        dominant = "decoding_leaning"
    else:
        dominant = "balanced"

    return f"{severity}_{dominant}"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the STATELESS derived features -- ones that depend only on each row's
    own raw values, not on any cross-row statistic. Safe to call on the full dataset
    before splitting, since nothing here is fit against a distribution that could leak
    between train and test (categorical one-hot encoding and per-row prosodic-sample
    summaries only reference that row's own data).

    Cohort-normalized features (normalized_speed/fluency/accuracy) are NOT computed
    here -- they require a train-only fitted norm and must be added via
    fit_grade_norms()/apply_grade_norms() AFTER learner_level_split(), never before.
    """
    engineered = df.copy()

    # --- genuine prosodic variability/pattern signal from the raw per-segment
    # samples (not a repeat of the existing raw aggregate columns) ---
    engineered["pitch_variability"] = engineered["pitch_samples"].map(
        lambda v: _safe_cv(_parse_sample_string(v))
    )
    engineered["energy_variability"] = engineered["energy_samples"].map(
        lambda v: _safe_cv(_parse_sample_string(v))
    )

    def _pause_pattern(row) -> float:
        pauses = _parse_sample_string(row["pause_durations"])
        if not pauses or row.get("reading_duration_sec", 0) in (0, None):
            return 0.0
        pause_rate_per_min = len(pauses) / (row["reading_duration_sec"] / 60)
        pause_cv = _safe_cv(pauses)
        # combines HOW OFTEN a child pauses with HOW ERRATIC those pauses are --
        # genuinely new information vs. avg_pause_duration (which only captures
        # mean length and already exists as a raw column)
        return round(pause_rate_per_min * (1 + pause_cv), 3)

    engineered["pause_pattern_score"] = engineered.apply(_pause_pattern, axis=1)

    # --- low-cardinality error micro-profile (see _error_micro_profile docstring) ---
    engineered["error_pattern_features"] = engineered.apply(
        lambda row: (
            f"hesitation:{row.hesitation_count};repetition:{row.repetition_count};"
            f"omission:{row.omission_count};substitution:{row.substitution_count};"
            f"insertion:{row.insertion_count}"
        ),
        axis=1,
    )  # kept as a human-readable description (METADATA_COLS), not a model feature
    engineered["error_micro_profile"] = engineered.apply(_error_micro_profile, axis=1)

    # --- categorical encoding (stateless: each row's own category membership) ---
    for col in CATEGORICAL_COLS:
        if col in engineered.columns:
            dummies = pd.get_dummies(engineered[col], prefix=col, drop_first=True, dtype=int)
            engineered = pd.concat([engineered.drop(columns=[col]), dummies], axis=1)

    return engineered


def fit_grade_norms(train_df: pd.DataFrame, min_group_size: int = 3) -> dict[str, Any]:
    """Fit grade-level cohort norms from the TRAINING split only. Returns per-grade
    mean for reading_speed_wpm/fluency_score/pronunciation_accuracy, with a global
    fallback for any grade with fewer than min_group_size training rows (small
    grade groups produce unstable norms otherwise). Must be called AFTER
    learner_level_split(), passing only train_df -- never the full/combined dataset,
    or test-set values leak into the norms used to normalize training rows too."""
    norms: dict[str, Any] = {"by_grade": {}, "global": {}}
    for col in ["reading_speed_wpm", "fluency_score", "pronunciation_accuracy"]:
        norms["global"][col] = float(train_df[col].mean()) if col in train_df.columns else 1.0
        norms["by_grade"][col] = {}
        if "grade" in train_df.columns and col in train_df.columns:
            for grade_val, group in train_df.groupby("grade"):
                if len(group) >= min_group_size:
                    norms["by_grade"][col][grade_val] = float(group[col].mean())
    return norms


def apply_grade_norms(df: pd.DataFrame, norms: dict[str, Any]) -> pd.DataFrame:
    """Apply previously-fit grade norms to add normalized_speed/fluency/accuracy.
    Works on train OR test using the SAME norms dict (fit once on train, applied to
    both) -- this is what keeps normalization leakage-free."""
    applied = df.copy()

    mapping = {
        "reading_speed_wpm": "normalized_speed",
        "fluency_score": "normalized_fluency",
        "pronunciation_accuracy": "normalized_accuracy",
    }
    for raw_col, new_col in mapping.items():
        if raw_col not in applied.columns:
            applied[new_col] = 1.0
            continue

        by_grade = norms["by_grade"].get(raw_col, {})
        global_norm = norms["global"].get(raw_col, 1.0) or 1.0

        def _normalize(row, raw_col=raw_col, by_grade=by_grade, global_norm=global_norm):
            grade_val = row.get("grade")
            grade_norm = by_grade.get(grade_val, global_norm) or global_norm
            return row[raw_col] / grade_norm if grade_norm else 1.0

        applied[new_col] = applied.apply(_normalize, axis=1)

    return applied


def learner_level_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data at the learner level so all sessions for a child stay together.
    Plain random child-level split -- does NOT guarantee every class of every target
    appears in both train and test. With few children per class (see
    stratified_learner_level_split), prefer that function instead for any target
    with a rare class, or this split can silently produce a test set missing a
    whole class entirely."""
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


def stratified_learner_level_split(
    df: pd.DataFrame, stratify_col: str = "intervention_intensity",
    test_frac: float = 0.2, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Child-level split (no session leakage, same as learner_level_split) but
    stratified by each child's MODAL label for stratify_col, so every class is
    guaranteed to appear in both train and test. This matters here specifically
    because some classes (e.g. intervention_intensity='Moderate') have as few as
    2 children total in the whole dataset -- a plain random child split can easily
    miss such a class entirely in the test set, which silently degrades a 3-class
    evaluation into an easier 2-class one and can produce misleadingly perfect
    holdout scores that don't reflect real 3-class performance (this happened in
    practice on this dataset before this function was added).

    A class with only 1 child total cannot be split at all -- that child is kept
    in train (a class absent from test is safer than a class absent from train,
    since a model with zero training examples for a class cannot learn it at all).
    """
    child_labels = df.groupby("child_id")[stratify_col].agg(lambda x: x.mode()[0])
    rng = random.Random(seed)

    train_children: list[str] = []
    test_children: list[str] = []
    for _, group in child_labels.groupby(child_labels):
        children_in_group = group.index.tolist()
        rng.shuffle(children_in_group)
        if len(children_in_group) <= 1:
            n_test = 0
        else:
            n_test = max(1, round(len(children_in_group) * test_frac))
        test_children.extend(children_in_group[:n_test])
        train_children.extend(children_in_group[n_test:])

    train_df = df[df["child_id"].isin(train_children)].copy()
    test_df = df[df["child_id"].isin(test_children)].copy()
    return train_df, test_df


def get_feature_cols(df: pd.DataFrame, exclude_subskills: bool = False) -> list[str]:
    """Return the model-input feature columns while keeping leakage rules centralized."""
    excluded = set(METADATA_COLS) | set(TARGET_COLS)
    feature_cols = [col for col in df.columns if col not in excluded]
    if exclude_subskills:
        feature_cols = [col for col in feature_cols if col not in SUBSKILL_COLS]
    return feature_cols


def get_xy_difficulty(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = all engineered features, y = difficulty_score (continuous, regression)."""
    X = df[get_feature_cols(df, exclude_subskills=False)]
    y = df["difficulty_score"]
    return X, y


def get_xy_weak_skill(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = ONLY the raw features that drive the four sub-skill scores (see
    WEAK_SKILL_FEATURE_COLS docstring), y = weak_skill (categorical, classification)."""
    available_cols = [c for c in WEAK_SKILL_FEATURE_COLS if c in df.columns]
    X = df[available_cols]
    y = df["weak_skill"]
    return X, y


def get_xy_intervention(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X = all engineered features, y = intervention_intensity (categorical, classification)."""
    X = df[get_feature_cols(df, exclude_subskills=False)]
    y = df["intervention_intensity"]
    return X, y