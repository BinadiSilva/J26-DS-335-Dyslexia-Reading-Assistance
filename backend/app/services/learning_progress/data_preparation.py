
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


def load_reading_data():
    """Load and prepare the sample reading-progress dataset."""
    df = pd.read_csv(DATA_PATH)

    # Check required columns
    required_columns = [
        "child_id",
        "session_number",
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
        "progress_score",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Convert numeric columns
    numeric_columns = required_columns[1:]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows with missing required values
    df = df.dropna(subset=required_columns)

    # Sort sessions for each child
    df = df.sort_values(
        ["child_id", "session_number"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = load_reading_data()

    print("\nDataset loaded successfully!")
    print(f"Number of records: {len(data)}")
    print(f"Number of children: {data['child_id'].nunique()}")

    print("\nFirst five records:")
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum())
