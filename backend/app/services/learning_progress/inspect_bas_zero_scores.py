# Import libraries for investigating zero scores
from pathlib import Path
import pandas as pd

# Locate the project root and BAS prediction dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_bas_prediction.csv"
)


def main():
    # Check that the prediction dataset exists
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared BAS prediction data
    df = pd.read_csv(INPUT_FILE)

    # Identify the score columns that require inspection
    score_columns = [
        "previous_score",
        "assessment_score",
        "target_score",
    ]

    # Count zero values and calculate their proportions
    print("ZERO-SCORE INVESTIGATION")
    print("========================")

    for column in score_columns:
        # Count zero values in the selected score column
        zero_count = (df[column] == 0).sum()

        # Calculate the percentage of zero values
        zero_percentage = zero_count / len(df) * 100

        print(
            f"{column}: {zero_count} zero values "
            f"({zero_percentage:.2f}%)"
        )

    # Summarize current scores by assessment time point
    print("\nCurrent scores by time point:")
    time_summary = df.groupby("time_point")["assessment_score"].agg(
        ["count", "min", "median", "max"]
    )
    print(time_summary.to_string())

    # Display records containing a zero current score
    print("\nExamples with a zero current score:")
    zero_examples = df[df["assessment_score"] == 0][
        [
            "UnlinkedParticipantNumber",
            "previous_time_point",
            "previous_score",
            "time_point",
            "assessment_score",
            "target_time_point",
            "target_score",
        ]
    ]

    print(zero_examples.head(10).to_string(index=False))


# Run the investigation when executed directly
if __name__ == "__main__":
    main()