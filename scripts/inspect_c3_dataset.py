
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "c3"
    / "raw"
    / "C3_Learner_Recommendation_Raw_Dataset.csv"
)

# Load the raw-like dataset
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("C3 DATASET INSPECTION")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. Dataset shape")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Column names and data types
print("\n2. Column names and data types")
print(df.dtypes)

# 3. Missing values
print("\n3. Missing values per column")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print(f"\nTotal missing cells: {df.isnull().sum().sum()}")

# 4. Duplicate rows
print("\n4. Duplicate rows")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 5. Numeric statistics
print("\n5. Numeric column statistics")
print(df.describe(include="number").T)

# 6. Unique values in categorical columns
print("\n6. Categorical column values")

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False).head(15))

# 7. Preview
print("\n7. First five rows")
print(df.head().to_string(index=False))

print("\nInspection completed.")
