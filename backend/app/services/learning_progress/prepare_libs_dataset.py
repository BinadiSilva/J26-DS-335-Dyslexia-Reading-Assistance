
# Import libraries for reading, transforming, and saving the dataset
from pathlib import Path
import re
import pandas as pd

# Locate the project root and source Excel workbook
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = PROJECT_ROOT / "LIBS_DataSet_UnLinked_July2024.xlsx"

# Define the output path without overwriting the original dataset
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

    # Collect assessment columns that include a time point
    assessment_columns = []

    for column in df.columns:
        # Match repeated columns such as T1BAS_word_reading
        match = TIME_PATTERN.match(str(column))

        if match:
            assessment_columns.append(column)

    # Stop if no repeated assessment columns were identified
    if not assessment_columns:
        print("ERROR: No time-point assessment columns were found.")
        return

    # Convert the wide spreadsheet into long-form assessment records
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

    # Count dot markers before converting them to missing values
    dot_mask = (
        long_df["assessment_score"]
        .astype("string")
        .str.strip()
        .eq(".")
        .fillna(False)
    )
    dot_count = int(dot_mask.sum())

    # Replace dot markers with missing values.
    # This treats dots as unavailable scores, not as zero scores.
    long_df.loc[dot_mask, "assessment_score"] = pd.NA

    # Convert time points and assessment scores to numeric values
    long_df["time_point"] = pd.to_numeric(
        long_df["time_point"], errors="coerce"
    )
    long_df["assessment_score"] = pd.to_numeric(
        long_df["assessment_score"], errors="coerce"
    )

    # Count records that will be excluded due to unavailable scores
    missing_score_count = int(long_df["assessment_score"].isna().sum())
    missing_time_count = int(long_df["time_point"].isna().sum())

    # Remove records without a usable time point or assessment score
    long_df = long_df.dropna(
        subset=["time_point", "assessment_score"]
    )

    # Keep the original score and assessment name.
    # Different tests must not be assumed to share the same scale.
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

    # Save the prepared data separately from the original workbook
    long_df.to_csv(OUTPUT_FILE, index=False)

    # Print a summary so preprocessing can be verified
    print("LIBS data preparation completed.")
    print(f"Source participants: {df[participant_column].nunique()}")
    print(f"Source assessment columns: {len(assessment_columns)}")
    print(f"Dot markers found: {dot_count}")
    print(f"Missing scores before removal: {missing_score_count}")
    print(f"Missing time points before removal: {missing_time_count}")
    print(f"Prepared records retained: {len(long_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    # Show records available at each time point
    print("\nRecords per time point:")
    print(long_df["time_point"].value_counts().sort_index().to_string())

    # Show the available assessment types
    print("\nAssessment types:")
    print(long_df["assessment_name"].value_counts().to_string())

    # Display a sample of the prepared records
    print("\nFirst five prepared records:")
    print(long_df.head().to_string(index=False))


# Run the preparation only when this file is executed directly
if __name__ == "__main__":
    main()
