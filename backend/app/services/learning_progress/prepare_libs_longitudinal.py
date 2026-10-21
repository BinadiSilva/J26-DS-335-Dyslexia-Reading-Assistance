"""Prepare BAS-only longitudinal data without writing files on import."""

import argparse

from pathlib import Path

import pandas as pd
import numpy as np

# Locate the actual project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKBOOK_PATH = PROJECT_ROOT / "LIBS_DataSet_UnLinked_July2024.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "component4" / "libs_reading_assessments_long.csv"

# Define the four longitudinal reading-score columns.
SCORE_COLUMNS = [
    "T1BAS_word_reading",
    "T2BAS_word_reading",
    "T3BAS_word_reading",
    "T4BAS_word_reading",
]

def prepare_longitudinal_data(workbook_path=WORKBOOK_PATH):
    """Return observed BAS records for participants with at least two scores."""
    # Load the original dataset without changing the source workbook.
    df = pd.read_excel(workbook_path, sheet_name=0)

    # Check that the required columns exist before processing.
    required_columns = ["UnlinkedParticipantNumber", *SCORE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}")

    # Convert dots and other non-numeric entries into missing values.
    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column].astype("string").str.strip().replace(".", pd.NA),
            errors="coerce",
        )

    # Values outside the documented score range are unavailable, not zero.
    for column in SCORE_COLUMNS:
        df[column] = df[column].where(df[column].between(0, 90))
    df["UnlinkedParticipantNumber"] = pd.to_numeric(
        df["UnlinkedParticipantNumber"], errors="coerce"
    )
    df = df.dropna(subset=["UnlinkedParticipantNumber"])
    df = df[np.isfinite(df["UnlinkedParticipantNumber"])]
    if df["UnlinkedParticipantNumber"].duplicated().any():
        raise ValueError("Duplicate participant IDs in the source workbook.")

    # Keep participants with at least two observed assessment scores.
    df["observed_score_count"] = df[SCORE_COLUMNS].notna().sum(axis=1)
    eligible = df.loc[df["observed_score_count"] >= 2].copy()

    # Convert the four assessment columns into one row per participant and time point.
    long_df = eligible.melt(
        id_vars=["UnlinkedParticipantNumber"],
        value_vars=SCORE_COLUMNS,
        var_name="assessment",
        value_name="reading_score",
    )

    # Extract the assessment number to preserve the original time-point order.
    long_df["time_point"] = (
        long_df["assessment"].str.extract(r"^T(\d+)")[0].astype(int)
    )

    # Remove missing assessment scores, but retain genuine zero scores.
    long_df = long_df.dropna(subset=["reading_score"])

    # Sort observations chronologically within each participant.
    long_df = long_df.sort_values(
        ["UnlinkedParticipantNumber", "time_point"]
    ).reset_index(drop=True)

    return long_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow replacing an existing output file")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists. Use a new --output path or --overwrite intentionally.")
    data = prepare_longitudinal_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Eligible participants: {data['UnlinkedParticipantNumber'].nunique()}")
    print(f"Observed assessment records: {len(data)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
