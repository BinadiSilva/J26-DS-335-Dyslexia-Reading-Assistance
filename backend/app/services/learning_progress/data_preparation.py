
from pathlib import Path

import pandas as pd


# Locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "reading_progress.csv"
)


# Required fields currently available in the dataset
REQUIRED_COLUMNS = [
    "child_id",
    "session_number",
    "reading_speed_wpm",
    "reading_accuracy",
    "fluency_score",
    "word_error_rate",
    "progress_score",
]

# Fields that must contain numeric values
NUMERIC_COLUMNS = [
    "session_number",
    "reading_speed_wpm",
    "reading_accuracy",
    "fluency_score",
    "word_error_rate",
    "progress_score",
]


def load_reading_data():
    """Load, validate, clean, and prepare reading-progress data."""

    # Check whether the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reading dataset not found: {DATA_PATH}"
        )

    # Load the CSV dataset
    df = pd.read_csv(DATA_PATH)

    # Check for required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Remove rows without a child ID
    df["child_id"] = df["child_id"].astype("string").str.strip()
    df = df.dropna(subset=["child_id"])
    df = df[df["child_id"] != ""]

    # Convert reading measurements to numeric values
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows with missing or invalid required values
    df = df.dropna(subset=NUMERIC_COLUMNS)

    # Remove duplicate child-session records
    df = df.drop_duplicates(
        subset=["child_id", "session_number"],
        keep="last",
    )

    # Sort sessions in chronological session order
    df = df.sort_values(
        ["child_id", "session_number"]
    ).reset_index(drop=True)

    # Check whether valid records remain
    if df.empty:
        raise ValueError(
            "No valid records remain after data cleaning."
        )

    # Calculate each child's previous progress score
    df["previous_progress_score"] = (
        df.groupby("child_id")["progress_score"].shift(1)
    )

    # Calculate progress change between consecutive sessions
    df["progress_change"] = (
        df["progress_score"] - df["previous_progress_score"]
    )

    # Keep the first session's progress change empty because
    # there is no previous session to compare against.
    # Do not treat this missing value as zero.

    return df


def get_dataset_summary(df):
    """Return basic information about the prepared dataset."""

    return {
        "total_records": len(df),
        "total_children": df["child_id"].nunique(),
        "total_sessions": df["session_number"].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "fields": df.columns.tolist(),
    }


if __name__ == "__main__":
    data = load_reading_data()
    summary = get_dataset_summary(data)

    print("\nDataset loaded and prepared successfully!")
    print("-" * 45)

    print(f"Dataset path: {DATA_PATH}")
    print(f"Total records: {summary['total_records']}")
    print(f"Total children: {summary['total_children']}")
    print(f"Distinct session numbers: {summary['total_sessions']}")

    print("\nAvailable fields:")
    for field in summary["fields"]:
        print(f"- {field}")

    print("\nFirst five prepared records:")
    print(data.head().to_string(index=False))

    print("\nMissing values by field:")
    print(data.isnull().sum().to_string())

    print("\nDataset preparation completed.")
