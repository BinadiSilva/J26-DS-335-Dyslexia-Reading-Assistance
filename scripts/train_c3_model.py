
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "datasets" / "c3" / "processed"
    / "C3_Learner_Recommendation_Processed_Dataset.csv"
)

MODEL_DIR = ROOT / "backend" / "app" / "models" / "trained"
RESULTS_DIR = ROOT / "datasets" / "c3" / "model_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset and define target
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

target = "recommended_activity"

# IDs are identifiers, not useful learner features.
# Exclude other recommendation outputs to avoid leakage.
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
y = df[target]

print("=" * 60)
print("C3 MACHINE-LEARNING EXPERIMENT")
print("=" * 60)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 4. Split into training and test sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# --------------------------------------------------
# 5. Define candidate models
# --------------------------------------------------
models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------
for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
    })

    trained_models[name] = pipeline

    print(f"\n{'=' * 50}")
    print(name)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    if name == "Random Forest":
        labels = pipeline.named_steps[
            "classifier"
        ].classes_

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        ).to_csv(RESULTS_DIR / "random_forest_confusion_matrix.csv")

# --------------------------------------------------
# 7. Compare models
# --------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    by="macro_f1",
    ascending=False,
)

print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False,
)

# Save the candidate with the highest macro F1.
best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

model_path = MODEL_DIR / "c3_activity_model.joblib"
joblib.dump(best_model, model_path)

metadata = {
    "selected_model": best_name,
    "target": target,
    "feature_columns": X.columns.tolist(),
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "selection_metric": "macro_f1",
    "note": (
        "Initial experiment using synthetic data. "
        "Requires further validation before real-world use."
    ),
}

with open(RESULTS_DIR / "training_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print(f"\nSelected model: {best_name}")
print(f"Saved model: {model_path}")
print(f"Saved results: {RESULTS_DIR}")
print("\nTraining experiment completed.")
