
# Import libraries for investigating zero scores
from pathlib import Path
import pandas as pd

# Locate the project root and prepared BAS prediction dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_bas_prediction.csv"
)


def main():
    # Stop if the prepared dataset is missing
    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared prediction dataset
    df = pd.read_csv(INPUT_FILE)

    # Define the score columns to investigate
    score_columns = [
        "previous_score",
        "assessment_score",
        "target_score",
    ]

    # Display the overall score distributions
    print("ZERO-SCORE INVESTIGATION")
    print("========================")
    print(f"Total records: {len(df)}")
    print(
        "Unique participants:",
        df["UnlinkedParticipantNumber"].nunique(),
    )

    # Report missing values, zero values, and score ranges
    print("\nScore summaries:")
    for column in score_columns:
        # Count missing and zero scores separately
        missing_count = int(df[column].isna().sum())
        zero_count = int((df[column] == 0).sum())
        valid_count = int(df[column].notna().sum())

        # Avoid division by zero when no valid scores exist
        zero_percentage = (
            zero_count / valid_count * 100 if valid_count else 0
        )

        print(f"\n{column}")
        print(f"  Valid values: {valid_count}")
        print(f"  Missing values: {missing_count}")
        print(f"  Zero values: {zero_count}")
        print(f"  Zero percentage: {zero_percentage:.2f}%")

        # Display descriptive statistics for non-missing scores
        print(df[column].describe().to_string())

    # Show current score distributions at each time point
    print("\nCurrent scores by time point:")
    time_summary = df.groupby("time_point")["assessment_score"].agg(
        ["count", "min", "median", "max"]
    )
    print(time_summary.to_string())

    # Display examples of records with zero current scores
    print("\nExamples with a zero current score:")
    columns_to_show = [
        "UnlinkedParticipantNumber",
        "previous_time_point",
        "previous_score",
        "time_point",
        "assessment_score",
        "target_time_point",
        "target_score",
    ]

    zero_examples = df.loc[
        df["assessment_score"] == 0,
        columns_to_show,
    ]

    print(zero_examples.head(15).to_string(index=False))

    # Check whether zero scores occur at particular time points
    print("\nZero current scores by time point:")
    zero_by_time = (
        df.assign(is_zero=df["assessment_score"].eq(0))
        .groupby("time_point")["is_zero"]
        .agg(["sum", "count"])
    )
    zero_by_time["zero_percentage"] = (
        zero_by_time["sum"] / zero_by_time["count"] * 100
    )
    print(zero_by_time.to_string())

    # Summarize participants with repeated zero current scores
    zero_counts = (
        df.assign(is_zero=df["assessment_score"].eq(0))
        .groupby("UnlinkedParticipantNumber")["is_zero"]
        .sum()
    )
    repeated_zero_participants = zero_counts[zero_counts >= 2]

    print("\nParticipants with zero current scores in 2 or more records:")
    print(f"Count: {len(repeated_zero_participants)}")
    print(repeated_zero_participants.head(15).to_string())


# Run the investigation only when this script is executed directly
if __name__ == "__main__":
    main()
