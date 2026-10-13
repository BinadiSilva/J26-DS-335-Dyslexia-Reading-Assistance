
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CATALOGUE_PATH = (
    ROOT / "datasets" / "c3"
    / "learning_content_catalogue.csv"
)

df = pd.read_csv(CATALOGUE_PATH)

required_columns = [
    "item_id",
    "title",
    "activity_type",
    "target_skill",
    "difficulty_level",
    "duration_minutes",
    "topic",
    "content",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df["item_id"].duplicated().any():
    raise ValueError("Duplicate item IDs found.")

print("Catalogue loaded successfully!")
print(f"Total content items: {len(df)}")
print("\nItems by activity type:")
print(df["activity_type"].value_counts())
print("\nCatalogue preview:")
print(df[required_columns].head(12).to_string(index=False))
