
# Import libraries for preparing longitudinal BAS prediction data
from pathlib import Path
import pandas as pd

# Locate the project root and prepared LIBS dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_all_assessments_long.csv"
)

# Save the prepared prediction data separately
OUTPUT_FILE = (
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

    # Load the prepared longitudinal assessment data
    df = pd.read_csv(INPUT_FILE)

    # Select the genuine BAS word-reading scores
    bas = df[df["assessment_name"] == "BAS_word_reading"].copy()

    # Remove records without required identifiers or scores
    bas = bas.dropna(
        subset=[
            "UnlinkedParticipantNumber",
            "time_point",
            "assessment_score",
        ]
    )

    # Sort scores chronologically within each participant
    bas = bas.sort_values(
        ["UnlinkedParticipantNumber", "time_point"]
    )

    # Stop if a participant has duplicate observations at one time point
    duplicate_mask = bas.duplicated(
        subset=["UnlinkedParticipantNumber", "time_point"],
        keep=False,
    )

    if duplicate_mask.any():
        print("ERROR: Duplicate participant/time-point records found.")
        print(
            bas.loc[
                duplicate_mask,
                [
                    "UnlinkedParticipantNumber",
                    "time_point",
                    "assessment_score",
                ],
            ].to_string(index=False)
        )
        return

    # Store each participant's previous, current, and next observations
    grouped = bas.groupby("UnlinkedParticipantNumber")
    bas["previous_score"] = grouped["assessment_score"].shift(1)
    bas["previous_time_point"] = grouped["time_point"].shift(1)
    bas["target_score"] = grouped["assessment_score"].shift(-1)
    bas["target_time_point"] = grouped["time_point"].shift(-1)

    # Keep rows with genuine previous and future observations
    model_data = bas.dropna(
        subset=[
            "previous_score",
            "previous_time_point",
            "target_score",
            "target_time_point",
        ]
    ).copy()

    # Calculate the gap between the current and target assessments
    model_data["time_gap"] = (
        model_data["target_time_point"] - model_data["time_point"]
    )

    # Keep only consecutive target periods to maintain a consistent horizon
    model_data = model_data[
        model_data["time_gap"] == 1
    ].copy()

    # Preserve the source values and assessment time points
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

    # Create the output directory if necessary
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save the prediction dataset without modifying the original data
    model_data.to_csv(OUTPUT_FILE, index=False)

    # Display summary information for verification
    print("BAS prediction dataset prepared.")
    print(f"Model records: {len(model_data)}")
    print(
        "Unique participants:",
        model_data["UnlinkedParticipantNumber"].nunique(),
    )
    print("Missing values:")
    print(model_data.isna().sum().to_string())
    print("Time-gap distribution:")
    print(model_data["time_gap"].value_counts().to_string())
    print(f"Output file: {OUTPUT_FILE}")

    # Show sample records for manual verification
    print("\nFirst five records:")
    print(model_data.head().to_string(index=False))


# Run preparation only when this script is executed directly
if __name__ == "__main__":
    main()
