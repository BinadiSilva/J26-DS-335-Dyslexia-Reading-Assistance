
# Import libraries for training and evaluating prediction models
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Locate the project root and prepared BAS prediction dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_bas_prediction.csv"
)

# Use only information available before the target assessment
FEATURES = [
    "previous_score",
    "assessment_score",
    "previous_time_point",
    "time_point",
]
TARGET = "target_score"
GROUP = "UnlinkedParticipantNumber"


def main():
    # Stop if the prepared dataset is missing
    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return

    # Load the prepared longitudinal BAS data
    df = pd.read_csv(INPUT_FILE)

    # Check that all required columns exist
    required_columns = FEATURES + [TARGET, GROUP]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing required columns:", missing_columns)
        return

    # Remove rows with missing features, targets, or participant IDs
    df = df.dropna(subset=required_columns).copy()

    # Ensure there is enough data for a meaningful group split
    if df[GROUP].nunique() < 2:
        print("ERROR: At least two participants are required.")
        return

    # Separate features, target scores, and participant groups
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    # Keep each participant entirely in either train or test data
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    # Create the training and testing subsets
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Define baseline regression models
    models = {
        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression(),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    }

    # Display the experiment setup
    print("LIBS BAS WORD-READING MODEL EVALUATION")
    print("======================================")
    print(f"Total usable records: {len(df)}")
    print(f"Training records: {len(train_indices)}")
    print(f"Testing records: {len(test_indices)}")
    print("Training participants:", groups.iloc[train_indices].nunique())
    print("Testing participants:", groups.iloc[test_indices].nunique())
    print("Features:", ", ".join(FEATURES))
    print("Target:", TARGET)

    # Train and evaluate each model on held-out participants
    for name, model in models.items():
        # Fit only on the training participants
        model.fit(X_train, y_train)

        # Predict the target scores for unseen participants
        predictions = model.predict(X_test)

        # Calculate regression performance metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")

    # Clearly state the limits of this initial experiment
    print("\nExperiment limitations:")
    print("- This is a baseline prediction experiment using LIBS data.")
    print("- Evaluation uses participants held out from model training.")
    print("- Results do not establish clinical validity for dyslexia.")
    print("- Further validation is needed before real-world use.")


# Run model training only when this script is executed directly
if __name__ == "__main__":
    main()
