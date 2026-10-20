# Import libraries for training and evaluating prediction models
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Locate the project root and BAS prediction dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_bas_prediction.csv"
)

# Use only information available at the current assessment
FEATURES = [
    "previous_score",
    "assessment_score",
    "previous_time_point",
    "time_point",
    "time_gap",
]
TARGET = "target_score"
GROUP = "UnlinkedParticipantNumber"


def main():
    # Check that the prepared dataset exists
    if not INPUT_FILE.exists():
        print(f"Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared BAS records
    df = pd.read_csv(INPUT_FILE)

    # Remove rows with missing targets, features, or participant IDs
    df = df.dropna(subset=FEATURES + [TARGET, GROUP]).copy()

    # Separate features, target scores, and participant groups
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    # Split by participant to reduce leakage between train and test sets
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Define two baseline models with preprocessing
    models = {
        "Linear Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    }

    # Train and evaluate each model on the same held-out participants
    print("LIBS BAS WORD-READING MODEL EVALUATION")
    print("======================================")
    print(f"Total usable records: {len(df)}")
    print(f"Training records: {len(train_indices)}")
    print(f"Testing records: {len(test_indices)}")
    print(
        "Training participants:",
        groups.iloc[train_indices].nunique(),
    )
    print(
        "Testing participants:",
        groups.iloc[test_indices].nunique(),
    )

    for name, model in models.items():
        # Fit the model using training participants only
        model.fit(X_train, y_train)

        # Predict later BAS scores for held-out participants
        predictions = model.predict(X_test)

        # Calculate regression evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")

    # Explain the scope of this initial experiment
    print("\nNote: This is an initial baseline experiment.")
    print("Results do not establish clinical validity or suitability")
    print("for individual children without further validation.")


# Run model training when this script is executed directly
if __name__ == "__main__":
    main()