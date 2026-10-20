
# Import libraries for reading and preparing the LIBS dataset.
from pathlib import Path

import pandas as pd

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

# Load the original dataset without changing the source workbook.
df = pd.read_excel(WORKBOOK_PATH, sheet_name=0)

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

# Save the prepared data separately from the original workbook.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUTPUT_PATH, index=False)

# Print a summary so the preparation can be checked.
print("LIBS longitudinal dataset prepared.")
print(f"Eligible participants: {eligible['UnlinkedParticipantNumber'].nunique()}")
print(f"Observed assessment records: {len(long_df)}")
print(f"Output file: {OUTPUT_PATH}")
print("\nRecords per time point:")
print(long_df["time_point"].value_counts().sort_index().to_string())
print("\nFirst 10 prepared records:")
print(long_df.head(10).to_string(index=False))
print("\nNote: This is real secondary dataset content, not synthetic data.")
