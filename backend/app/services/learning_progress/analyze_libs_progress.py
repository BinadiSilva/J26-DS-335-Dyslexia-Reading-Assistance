# Import libraries for longitudinal progress analysis
from pathlib import Path
import pandas as pd

# Locate the project root and prepared LIBS dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_reading_assessments_long.csv"
)

# Define a separate output file for progress changes
OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_progress_changes.csv"
)


def main():
    # Check whether the prepared dataset exists
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared longitudinal records
    df = pd.read_csv(INPUT_FILE)

    # Sort each participant's scores within each assessment type
    df = df.sort_values(
        [
            "UnlinkedParticipantNumber",
            "assessment_name",
            "time_point",
        ]
    )

    # Calculate the previous score for the same participant
    # and the same assessment type
    grouping = df.groupby(
        ["UnlinkedParticipantNumber", "assessment_name"]
    )

    df["previous_score"] = grouping["assessment_score"].shift(1)

    # Calculate score change between consecutive available assessments
    df["score_change"] = (
        df["assessment_score"] - df["previous_score"]
    )

    # Create a separate output directory if necessary
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save the results without modifying the source CSV
    df.to_csv(OUTPUT_FILE, index=False)

    # Display a summary of the calculated changes
    valid_changes = df.dropna(subset=["score_change"])

    print("LIBS progress analysis completed.")
    print(f"Total records: {len(df)}")
    print(f"Records with a previous score: {len(valid_changes)}")
    print(f"Output file: {OUTPUT_FILE}")

    # Summarize score changes separately for each assessment type
    print("\nScore-change summary by assessment:")
    summary = valid_changes.groupby("assessment_name")["score_change"].agg(
        ["count", "mean", "median"]
    )
    print(summary.round(2).to_string())

    # Display sample records for verification
    print("\nFirst five records:")
    print(df.head().to_string(index=False))


# Run the analysis when this file is executed directly
if __name__ == "__main__":
    main()