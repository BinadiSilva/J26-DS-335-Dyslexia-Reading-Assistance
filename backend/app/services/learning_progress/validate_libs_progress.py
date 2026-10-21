# Import libraries for validating longitudinal assessment records
from pathlib import Path
import pandas as pd

# Locate the project root and progress-change dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_progress_changes.csv"
)


def main():
    # Check that the progress dataset exists
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the calculated progress changes
    df = pd.read_csv(INPUT_FILE)

    # Count distinct participants and repeated observations per assessment
    summary = (
        df.groupby("assessment_name")
        .agg(
            participants=("UnlinkedParticipantNumber", "nunique"),
            observations=("assessment_score", "count"),
            repeated_changes=("score_change", "count"),
            mean_score=("assessment_score", "mean"),
            mean_change=("score_change", "mean"),
            positive_changes=("score_change", lambda values: (values > 0).sum()),
            negative_changes=("score_change", lambda values: (values < 0).sum()),
            unchanged=("score_change", lambda values: (values == 0).sum()),
        )
        .round(2)
        .sort_values("participants", ascending=False)
    )

    # Display the validation results
    print("LIBS LONGITUDINAL VALIDATION")
    print("============================")
    print(f"Total records: {len(df)}")
    print(
        "Distinct participants:",
        df["UnlinkedParticipantNumber"].nunique(),
    )

    print("\nSummary by assessment:")
    print(summary.to_string())

    # Save the summary separately for later model planning
    output_file = INPUT_FILE.parent / "libs_assessment_validation.csv"
    summary.to_csv(output_file)

    print(f"\nValidation summary saved to: {output_file}")


# Run validation when executed directly
if __name__ == "__main__":
    main()