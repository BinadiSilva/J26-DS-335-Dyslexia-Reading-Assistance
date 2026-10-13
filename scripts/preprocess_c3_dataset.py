
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Define dataset paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT / "datasets" / "c3" / "raw"
    / "C3_Learner_Recommendation_Raw_Dataset.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "c3" / "processed"
PROCESSED_PATH = (
    PROCESSED_DIR / "C3_Learner_Recommendation_Processed_Dataset.csv"
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------
df = pd.read_csv(RAW_PATH)

original_rows = len(df)
original_missing = int(df.isna().sum().sum())

print("Starting preprocessing...")
print(f"Original shape: {df.shape}")
print(f"Original missing cells: {original_missing}")

# --------------------------------------------------
# 3. Standardize categorical values
# --------------------------------------------------
categorical_columns = [
    "preferred_topic",
    "preferred_activity",
    "difficulty_level",
    "weak_skill",
    "previous_activity",
    "passage_topic",
    "passage_difficulty",
    "recommended_activity",
    "recommendation_level",
]

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.title()
    )

# Clean identifier whitespace without changing ID format.
for column in ["child_id", "session_id"]:
    df[column] = df[column].astype("string").str.strip()

# --------------------------------------------------
# 4. Convert invalid numeric values to missing values
# --------------------------------------------------
# Define valid ranges based on the meaning of each field.
valid_ranges = {
    "age": (5, 18),
    "grade": (1, 12),
    "preferred_duration_min": (1, 120),
    "reading_score": (0, 100),
    "reading_speed_wpm": (0, None),
    "pronunciation_accuracy": (0, 100),
    "fluency_score": (0, 100),
    "difficulty_score": (0, 1),
    "previous_score": (0, 100),
    "activity_completion_rate": (0, 1),
    "recommended_duration_min": (1, 120),
}

for column, (minimum, maximum) in valid_ranges.items():
    df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[column] < minimum
    if maximum is not None:
        invalid = invalid | (df[column] > maximum)

    df.loc[invalid, column] = float("nan")

# --------------------------------------------------
# 5. Handle missing values
# --------------------------------------------------
# Do not invent missing recommendation target labels.
target_columns = [
    "recommended_activity",
    "recommendation_level",
    "recommended_duration_min",
]

# Remove rows only if a required target is missing.
df = df.dropna(subset=target_columns).copy()

# Fill missing numeric input values with their medians.
numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

for column in numeric_columns:
    median_value = df[column].median()

    if pd.notna(median_value):
        df[column] = df[column].fillna(median_value)

# Fill missing categorical values with "Unknown".
for column in df.select_dtypes(
    include=["object", "string"]
).columns:
    df[column] = df[column].fillna("Unknown")

# --------------------------------------------------
# 6. Check the cleaned dataset
# --------------------------------------------------
print("\nPreprocessing results")
print("-" * 40)
print(f"Original rows: {original_rows}")
print(f"Processed rows: {len(df)}")
print(f"Rows removed: {original_rows - len(df)}")
print(f"Remaining missing cells: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\nProcessed dataset shape:")
print(df.shape)

print("\nRemaining missing values by column:")
print(df.isna().sum()[df.isna().sum() > 0])

# --------------------------------------------------
# 7. Save the processed dataset
# --------------------------------------------------
df.to_csv(PROCESSED_PATH, index=False)

print("\nProcessed dataset saved successfully:")
print(PROCESSED_PATH)
