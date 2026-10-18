# Import libraries for preparing longitudinal BAS prediction data
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

# Save the BAS prediction data to a separate file
OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_bas_prediction.csv"
)


def main():
    # Stop if the prepared LIBS dataset is missing
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared longitudinal assessment data
    df = pd.read_csv(INPUT_FILE)

    # Select only the BAS word-reading assessment
    bas = df[df["assessment_name"] == "BAS_word_reading"].copy()

    # Remove records without a participant, time point, or score
    bas = bas.dropna(
        subset=[
            "UnlinkedParticipantNumber",
            "time_point",
            "assessment_score",
        ]
    )

    # Sort observations chronologically for each participant
    bas = bas.sort_values(
        ["UnlinkedParticipantNumber", "time_point"]
    )

    # Keep one observation per participant and time point.
    # If duplicates exist, stop so they can be investigated.
    duplicate_mask = bas.duplicated(
        subset=["UnlinkedParticipantNumber", "time_point"],
        keep=False,
    )

    if duplicate_mask.any():
        print("Duplicate participant/time-point observations found.")
        print(
            bas.loc[
                duplicate_mask,
                [
                    "UnlinkedParticipantNumber",
                    "time_point",
                    "assessment_score",
                    "source_variable",
                ],
            ].to_string(index=False)
        )
        print("Please inspect duplicates before creating model data.")
        return

    # Create previous-score and next-observed-score columns
    grouped = bas.groupby("UnlinkedParticipantNumber")
    bas["previous_score"] = grouped["assessment_score"].shift(1)
    bas["previous_time_point"] = grouped["time_point"].shift(1)
    bas["target_score"] = grouped["assessment_score"].shift(-1)
    bas["target_time_point"] = grouped["time_point"].shift(-1)

    # Keep only observations with both a previous and a later score
    model_data = bas.dropna(
        subset=[
            "previous_score",
            "previous_time_point",
            "target_score",
            "target_time_point",
        ]
    ).copy()

    # Record the time gap so predictions are interpreted correctly
    model_data["time_gap"] = (
        model_data["target_time_point"]
        - model_data["time_point"]
    )

    # Keep only cases where the target occurs after the current time
    model_data = model_data[model_data["time_gap"] > 0]

    # Select clearly named model fields and source references
    model_data = model_data[
        [
            "UnlinkedParticipantNumber",
            "previous_time_point",
            "previous_score",
            "time_point",
            "assessment_score",
            "target_time_point",
            "target_score",
            "time_gap",
        ]
    ]

    # Create the output folder if it does not exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save the new dataset without changing any source files
    model_data.to_csv(OUTPUT_FILE, index=False)

    # Print a summary to verify the output
    print("BAS prediction dataset prepared.")
    print(f"Model records: {len(model_data)}")
    print(
        "Unique participants:",
        model_data["UnlinkedParticipantNumber"].nunique(),
    )
    print(f"Output file: {OUTPUT_FILE}")
    print("\nFirst five records:")
    print(model_data.head().to_string(index=False))


# Run preparation only when this script is executed directly
if __name__ == "__main__":
    main()