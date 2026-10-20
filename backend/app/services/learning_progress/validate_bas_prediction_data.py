# Import libraries for checking BAS prediction data quality
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
    # Stop if the prepared dataset cannot be found
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared prediction dataset
    df = pd.read_csv(INPUT_FILE)

    # Display dataset dimensions and column names
    print("BAS PREDICTION DATA QUALITY CHECK")
    print("=================================")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())

    # Check missing values in every column
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # Check duplicate complete records
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Check whether participant and target time points are valid
    print(
        "\nUnique participants:",
        df["UnlinkedParticipantNumber"].nunique(),
    )
    print(
        "Invalid time gaps:",
        (df["time_gap"] <= 0).sum(),
    )

    # Display ranges for numeric score and time columns
    numeric_columns = [
        "previous_score",
        "assessment_score",
        "target_score",
        "previous_time_point",
        "time_point",
        "target_time_point",
        "time_gap",
    ]

    print("\nNumeric value ranges:")
    print(df[numeric_columns].describe().round(2).to_string())

    # Check whether a participant appears more than once
    # for the same current time point
    duplicate_keys = df.duplicated(
        subset=["UnlinkedParticipantNumber", "time_point"],
        keep=False,
    )
    print(
        "\nRows sharing a participant/current time point:",
        duplicate_keys.sum(),
    )

    # Show a few rows so the values can be inspected
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


# Run validation only when this script is executed directly
if __name__ == "__main__":
    main()