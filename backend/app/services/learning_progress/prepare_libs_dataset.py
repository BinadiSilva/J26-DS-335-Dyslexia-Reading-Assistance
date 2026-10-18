# Import libraries for reading, transforming, and saving the dataset
from pathlib import Path
import re
import pandas as pd

# Locate the project root and source Excel workbook
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = PROJECT_ROOT / "LIBS_DataSet_UnLinked_July2024.xlsx"

# Define the output path without overwriting the existing dataset
OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_reading_assessments_long.csv"
)

# Identify columns that contain repeated assessment time points
TIME_PATTERN = re.compile(r"^T(\d+)_?(.+)$")


def main():
    # Stop safely if the source workbook is missing
    if not INPUT_FILE.exists():
        print(f"ERROR: Source dataset not found: {INPUT_FILE}")
        return

    # Load the first worksheet from the real LIBS workbook
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")

    # Find the participant identifier column
    participant_column = "UnlinkedParticipantNumber"

    if participant_column not in df.columns:
        print(f"ERROR: Missing column: {participant_column}")
        print("Available columns:", df.columns.tolist())
        return

    # Collect reading assessment columns and their time points
    assessment_columns = []

    for column in df.columns:
        # Match repeated assessment columns such as T1BAS_word_reading
        match = TIME_PATTERN.match(str(column))

        if match:
            assessment_columns.append(column)

    # Stop if no repeated assessment columns were identified
    if not assessment_columns:
        print("ERROR: No time-point assessment columns were found.")
        return

    # Convert the wide spreadsheet into one row per participant,
    # assessment time point, and reading measure
    long_df = df.melt(
        id_vars=[participant_column],
        value_vars=assessment_columns,
        var_name="source_variable",
        value_name="assessment_score",
    )

    # Extract the time point and original assessment name
    long_df["time_point"] = long_df["source_variable"].str.extract(
        r"^T(\d+)_?(.+)$"
    )[0]

    long_df["assessment_name"] = long_df["source_variable"].str.extract(
        r"^T\d+_?(.+)$"
    )[0]

    # Convert time points and scores to numeric values
    long_df["time_point"] = pd.to_numeric(
        long_df["time_point"], errors="coerce"
    )
    long_df["assessment_score"] = pd.to_numeric(
        long_df["assessment_score"], errors="coerce"
    )

    # Remove records where the reading assessment score is missing
    long_df = long_df.dropna(
        subset=["time_point", "assessment_score"]
    )

    # Keep the original score and assessment name; different tests
    # must not be assumed to use the same scoring scale
    long_df = long_df[
        [
            participant_column,
            "time_point",
            "assessment_name",
            "assessment_score",
            "source_variable",
        ]
    ]

    # Sort records for easier longitudinal analysis
    long_df = long_df.sort_values(
        [participant_column, "assessment_name", "time_point"]
    ).reset_index(drop=True)

    # Create the output directory if it does not exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save the prepared data separately from the original dataset
    long_df.to_csv(OUTPUT_FILE, index=False)

    # Print a summary so the output can be verified
    print("LIBS data preparation completed.")
    print(f"Source participants: {df[participant_column].nunique()}")
    print(f"Source assessment columns: {len(assessment_columns)}")
    print(f"Prepared records: {len(long_df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nRecords per time point:")
    print(long_df["time_point"].value_counts().sort_index().to_string())
    print("\nAssessment types:")
    print(long_df["assessment_name"].value_counts().to_string())
    print("\nFirst five prepared records:")
    print(long_df.head().to_string(index=False))


# Run the preparation only when this file is executed directly
if __name__ == "__main__":
    main()