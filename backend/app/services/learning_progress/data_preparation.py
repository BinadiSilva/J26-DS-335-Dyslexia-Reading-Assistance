
"""Prepare synthetic and LIBS reading data for Component 4."""

from pathlib import Path

import pandas as pd
import numpy as np


# Locate the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Paths to the synthetic and real secondary-assessment datasets.
SYNTHETIC_DATA_PATH = (
    PROJECT_ROOT / "datasets" / "component4" / "reading_progress.csv"
)

LIBS_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_reading_assessments_long.csv"
)

# Required fields for the existing synthetic workflow.
SYNTHETIC_REQUIRED_COLUMNS = [
    "child_id",
    "session_number",
    "reading_speed_wpm",
    "reading_accuracy",
    "fluency_score",
    "word_error_rate",
    "progress_score",
]

SYNTHETIC_NUMERIC_COLUMNS = [
    "session_number",
    "reading_speed_wpm",
    "reading_accuracy",
    "fluency_score",
    "word_error_rate",
    "progress_score",
]

# Required fields for the LIBS longitudinal workflow.
LIBS_REQUIRED_COLUMNS = [
    "UnlinkedParticipantNumber",
    "assessment",
    "reading_score",
    "time_point",
]


def _check_columns(df, required_columns, dataset_name):
    """Raise an informative error if required columns are missing."""
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_reading_data():
    """Load and validate the existing synthetic reading dataset.

    This function is retained for compatibility with the original
    Component 4 services until they are migrated to the LIBS workflow.
    """

    if not SYNTHETIC_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic reading dataset not found: {SYNTHETIC_DATA_PATH}"
        )

    df = pd.read_csv(SYNTHETIC_DATA_PATH)
    _check_columns(df, SYNTHETIC_REQUIRED_COLUMNS, "Synthetic dataset")

    # Clean participant identifiers.
    df["child_id"] = df["child_id"].astype("string").str.strip()
    df = df.dropna(subset=["child_id"])
    df = df[df["child_id"] != ""]

    # Convert numeric fields and remove invalid records.
    for column in SYNTHETIC_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=SYNTHETIC_NUMERIC_COLUMNS)

    # Keep one record per child and session.
    df = df.drop_duplicates(
        subset=["child_id", "session_number"],
        keep="last",
    )

    # Sort sessions chronologically.
    df = df.sort_values(
        ["child_id", "session_number"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid synthetic reading records remain.")

    # Calculate change from the previous session.
    df["previous_progress_score"] = (
        df.groupby("child_id")["progress_score"].shift(1)
    )
    df["progress_change"] = (
        df["progress_score"] - df["previous_progress_score"]
    )

    return df


def load_libs_reading_data():
    """Load and validate the prepared LIBS longitudinal dataset.

    LIBS participant IDs are unlinked identifiers, not child identities.
    Zero reading scores are retained because zero is within the documented
    BAS score range. Dot values should already have been converted to
    missing values during longitudinal dataset preparation.
    """

    if not LIBS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"LIBS dataset not found: {LIBS_DATA_PATH}"
        )

    df = pd.read_csv(LIBS_DATA_PATH)
    _check_columns(df, LIBS_REQUIRED_COLUMNS, "LIBS dataset")
    source_rows = len(df)

    # Standardize the assessment participant identifier.
    participant_column = "UnlinkedParticipantNumber"
    df[participant_column] = pd.to_numeric(
        df[participant_column], errors="coerce"
    )

    # Validate score and time-point fields.
    df["reading_score"] = pd.to_numeric(
        df["reading_score"], errors="coerce"
    )
    df["time_point"] = pd.to_numeric(
        df["time_point"], errors="coerce"
    )

    df = df.dropna(
        subset=[participant_column, "reading_score", "time_point"]
    )
    df = df[np.isfinite(df[participant_column])]

    # Check the documented BAS score range and expected time points.
    df = df[df["reading_score"].between(0, 90)]
    df = df[df["time_point"].isin([1, 2, 3, 4])]

    # Reject mixed tests or a time-point/name mismatch instead of silently
    # treating another assessment's scale as a BAS reading score.
    df["assessment"] = df["assessment"].astype("string").str.strip()
    expected_names = df["time_point"].map({i: f"T{i}BAS_word_reading" for i in range(1, 5)})
    if not df["assessment"].eq(expected_names).fillna(False).all():
        raise ValueError("Assessment names must match BAS word reading and time_point.")

    invalid_rows = source_rows - len(df)
    keys = [participant_column, "time_point"]
    if df.groupby(keys)["reading_score"].nunique().gt(1).any():
        raise ValueError("Conflicting reading scores for the same participant/time point.")
    duplicates = int(df.duplicated(keys).sum())
    df = df.drop_duplicates(subset=keys, keep="first")

    # Sort each participant's assessments in time-point order.
    df = df.sort_values(
        [participant_column, "time_point"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid LIBS reading records remain.")

    # Calculate the difference from the previous observed assessment.
    df["previous_reading_score"] = (
        df.groupby(participant_column)["reading_score"].shift(1)
    )
    df["reading_score_change"] = (
        df["reading_score"] - df["previous_reading_score"]
    )
    df.attrs["quality_report"] = {
        "source_rows": source_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_rows_removed": duplicates,
        "retained_rows": len(df),
    }

    return df


def get_dataset_summary(df):
    """Return a basic summary of a prepared dataset."""

    return {
        "total_records": len(df),
        "total_participants": (
            df["UnlinkedParticipantNumber"].nunique()
            if "UnlinkedParticipantNumber" in df.columns
            else df["child_id"].nunique()
        ),
        "fields": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
    }


if __name__ == "__main__":
    # Test both data sources without mixing their schemas.
    for dataset_name, loader in [
        ("Synthetic", load_reading_data),
        ("LIBS", load_libs_reading_data),
    ]:
        data = loader()
        summary = get_dataset_summary(data)

        print(f"\n{dataset_name} dataset loaded successfully.")
        print("-" * 45)
        print(f"Records: {summary['total_records']}")
        print(f"Participants: {summary['total_participants']}")
        print(f"Missing values: {summary['missing_values']}")
        print(f"Fields: {summary['fields']}")
