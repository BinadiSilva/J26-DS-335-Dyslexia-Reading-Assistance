# Import libraries for training and evaluating prediction models
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Locate the project root and prepared real LIBS longitudinal dataset
PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "component4"
    / "libs_reading_assessments_long.csv"
)

# Define features available before the next observed assessment
FEATURES = [
    "previous_score",
    "assessment_score",
    "previous_time_point",
    "time_point",
]

# Predict the next observed reading score
TARGET = "target_score"

# Keep each participant entirely within one data split
GROUP = "UnlinkedParticipantNumber"


def main():
    # Stop if the prepared dataset is missing
    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found: {INPUT_FILE}")
        return

    # Load the real longitudinal LIBS assessment data
    df = pd.read_csv(INPUT_FILE)

    # Check that the original dataset columns exist
    required_original_columns = [
        GROUP,
        "assessment",
        "reading_score",
        "time_point",
    ]
    missing_columns = [
        column
        for column in required_original_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing required columns:", missing_columns)
        return

    # Sort each participant's assessments chronologically
    df = df.sort_values(
        [GROUP, "time_point"]
    ).copy()

    # Create the previous observed score
    df["previous_score"] = (
        df.groupby(GROUP)["reading_score"].shift(1)
    )

    # Create the score from the current assessment
    df["assessment_score"] = df["reading_score"]

    # Record the previous assessment time point
    df["previous_time_point"] = (
        df.groupby(GROUP)["time_point"].shift(1)
    )

    # Create the next observed score as the prediction target
    df["target_score"] = (
        df.groupby(GROUP)["reading_score"].shift(-1)
    )

    # Record the time point associated with the target
    df["target_time_point"] = (
        df.groupby(GROUP)["time_point"].shift(-1)
    )

    # Remove rows without a previous score or next observed target
    df = df.dropna(
        subset=FEATURES + [TARGET, GROUP]
    ).copy()

    # Ensure time-point columns use integer values
    df["previous_time_point"] = (
        df["previous_time_point"].astype(int)
    )
    df["time_point"] = df["time_point"].astype(int)
    df["target_time_point"] = (
        df["target_time_point"].astype(int)
    )

    # Ensure enough participants are available for evaluation
    if df[GROUP].nunique() < 2:
        print("ERROR: At least two participants are required.")
        return

    # Separate model features, target scores, and participant groups
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    # Split by participant to prevent participant leakage
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    # Create training and testing subsets
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Define the baseline prediction models
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

    # Store evaluation metrics for model comparison
    results = []

    # Display the experiment setup
    print("LIBS LONGITUDINAL READING PROGRESS EVALUATION")
    print("============================================")
    print(f"Input dataset: {INPUT_FILE.name}")
    print(f"Usable prediction records: {len(df)}")
    print(f"Unique participants: {groups.nunique()}")
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
    print("Features:", ", ".join(FEATURES))
    print("Target:", TARGET)

    # Use the current observed score as a persistence prediction
    persistence_predictions = X_test["assessment_score"]

    # Evaluate the persistence baseline on the held-out participants
    results.append({
        "Model": "Persistence Baseline",
        "MAE": mean_absolute_error(
            y_test, persistence_predictions
        ),
        "RMSE": mean_squared_error(
            y_test, persistence_predictions
        ) ** 0.5,
        "R2": r2_score(
            y_test, persistence_predictions
        ),
    })

    # Train and evaluate each machine-learning model
    for name, model in models.items():
        # Fit using training participants only
        model.fit(X_train, y_train)

        # Predict scores for held-out participants
        predictions = model.predict(X_test)

        # Calculate regression metrics
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "R2": r2_score(y_test, predictions),
        })

    # Display the model comparison
    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print("================")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": "{:.2f}".format,
                "RMSE": "{:.2f}".format,
                "R2": "{:.3f}".format,
            },
        )
    )

    # Identify the model with the lowest mean absolute error
    best_model = results_df.loc[
        results_df["MAE"].idxmin(), "Model"
    ]
    print(f"\nLowest MAE: {best_model}")

    # Explain the limitations of the experiment
    print("\nExperiment limitations:")
    print("- Uses real secondary LIBS assessment data.")
    print("- Predicts the next observed score, not a fixed future date.")
    print("- Participants are held out from model training.")
    print("- The dataset does not directly measure dyslexia intervention effectiveness.")
    print("- Results do not establish clinical validity for dyslexia.")
    print("- Further validation is required before real-world use.")


# Run model training only when this script is executed directly
if __name__ == "__main__":
    main()