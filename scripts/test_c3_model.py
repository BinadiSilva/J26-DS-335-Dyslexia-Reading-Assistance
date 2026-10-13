
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "backend" / "app" / "models" / "trained"
    / "c3_activity_model.joblib"
)

DATA_PATH = (
    ROOT / "datasets" / "c3" / "processed"
    / "C3_Learner_Recommendation_Processed_Dataset.csv"
)

# Load the saved pipeline and processed dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

excluded_columns = [
    "child_id",
    "session_id",
    "recommended_activity",
    "recommendation_level",
    "recommended_duration_min",
    "passage_topic",
    "passage_difficulty",
]

X = df.drop(columns=excluded_columns)
y = df["recommended_activity"]

# Test a few example profiles
sample = X.iloc[[0, 1, 2, 3, 4]]
actual = y.iloc[[0, 1, 2, 3, 4]]
predicted = model.predict(sample)

results = pd.DataFrame({
    "Actual Activity": actual.to_numpy(),
    "Predicted Activity": predicted,
})

print("Saved Model - Example Predictions")
print("=" * 50)
print(results.to_string(index=False))
