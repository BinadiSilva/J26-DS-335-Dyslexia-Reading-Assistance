
"""Train and evaluate reading-progress prediction models for Component 4."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Support imports when this module is run directly or imported as a package.
try:
    from .data_preparation import load_libs_reading_data, load_reading_data
except ImportError:
    from data_preparation import load_libs_reading_data, load_reading_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = (
    PROJECT_ROOT / "datasets" / "component4" / "libs_prediction_results.csv"
)

PARTICIPANT_COLUMN = "UnlinkedParticipantNumber"
RANDOM_STATE = 42


def prepare_libs_prediction_data():
    """Create examples that predict each participant's next observed score.

    The target is the next observed assessment score, not a score at a
    fixed future date. Participants with missing assessment time points
    may therefore have non-consecutive observations.
    """

    # Load and sort assessments so score shifts follow chronological order.
    df = load_libs_reading_data()
    df = df.sort_values(
        [PARTICIPANT_COLUMN, "time_point"]
    ).copy()

    grouped_scores = df.groupby(PARTICIPANT_COLUMN)["reading_score"]

    # Use the previous and current observed scores as predictors.
    df["previous_score"] = grouped_scores.shift(1)
    df["target_score"] = grouped_scores.shift(-1)

    # Keep assessment time points as additional predictors.
    df["previous_time_point"] = (
        df.groupby(PARTICIPANT_COLUMN)["time_point"].shift(1)
    )
    df["target_time_point"] = (
        df.groupby(PARTICIPANT_COLUMN)["time_point"].shift(-1)
    )

    features = [
        "previous_score",
        "reading_score",
        "previous_time_point",
        "time_point",
    ]

    # Remove rows that do not have all required predictors and target.
    usable = df.dropna(
        subset=features + ["target_score", PARTICIPANT_COLUMN]
    ).copy()

    usable = usable.rename(
        columns={"reading_score": "assessment_score"}
    )

    # Return only the columns needed for training, evaluation, and reporting.
    return usable[
        [
            PARTICIPANT_COLUMN,
            "previous_score",
            "assessment_score",
            "previous_time_point",
            "time_point",
            "target_time_point",
            "target_score",
        ]
    ].reset_index(drop=True)


def _evaluate_predictions(actual, predicted):
    """Calculate standard regression evaluation metrics."""

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _create_models():
    """Create fresh model instances for each training stage."""

    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
    }


def train_libs_progress_model(save_results=True):
    """Select models using validation participants and evaluate on test participants.

    Participants are separated into training, validation, and test groups.
    The validation set selects the model; the held-out test set is used only
    for final evaluation. The selected model is then refitted on all training
    participants before final test evaluation.
    """

    data = prepare_libs_prediction_data()

    # Check that the dataset is large enough for participant-level splitting.
    if data.empty or data[PARTICIPANT_COLUMN].nunique() < 3:
        raise ValueError(
            "At least three participants with usable prediction records "
            "are required for training, validation, and testing."
        )

    features = [
        "previous_score",
        "assessment_score",
        "previous_time_point",
        "time_point",
    ]
    target = "target_score"

    # First, reserve 20% of participants for the final test set.
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        outer_splitter.split(
            data[features],
            data[target],
            groups=data[PARTICIPANT_COLUMN],
        )
    )

    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Split the remaining training participants into model-training and
    # validation groups. Test participants are not used for model selection.
    if train_data[PARTICIPANT_COLUMN].nunique() < 2:
        raise ValueError(
            "Not enough training participants for a separate validation split."
        )

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE + 1,
    )

    inner_train_indices, validation_indices = next(
        validation_splitter.split(
            train_data[features],
            train_data[target],
            groups=train_data[PARTICIPANT_COLUMN],
        )
    )

    inner_train = train_data.iloc[inner_train_indices]
    validation_data = train_data.iloc[validation_indices]

    X_inner_train = inner_train[features]
    y_inner_train = inner_train[target]
    X_validation = validation_data[features]
    y_validation = validation_data[target]

    # Compare candidate models on validation participants.
    validation_results = {}
    validation_models = _create_models()

    # Include the persistence baseline in validation reporting.
    validation_baseline = X_validation["assessment_score"].to_numpy()
    validation_results["Persistence Baseline"] = _evaluate_predictions(
        y_validation,
        validation_baseline,
    )

    for model_name, model in validation_models.items():
        # Fit only on the inner training participants.
        model.fit(X_inner_train, y_inner_train)
        validation_predictions = model.predict(X_validation)

        validation_results[model_name] = _evaluate_predictions(
            y_validation,
            validation_predictions,
        )

    # Select the machine-learning model with the lowest validation MAE.
    # The final test-set results do not influence this choice.
    best_model_name = min(
        validation_models,
        key=lambda name: validation_results[name]["mae"],
    )

    # Refit fresh candidate models on all outer training participants.
    # This makes all training participants available after selection.
    final_models = _create_models()
    results = {}

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Evaluate the persistence baseline on the held-out test participants.
    baseline_predictions = X_test["assessment_score"].to_numpy()
    results["Persistence Baseline"] = {
        "model": None,
        **_evaluate_predictions(y_test, baseline_predictions),
    }

    for model_name, model in final_models.items():
        # Refit on all training participants, then evaluate once on test data.
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)

        results[model_name] = {
            "model": model,
            **_evaluate_predictions(y_test, test_predictions),
        }

    print("\nLIBS Longitudinal Reading Progress Evaluation")
    print("=" * 48)
    print(
        "Input dataset: "
        "datasets/component4/libs_reading_assessments_long.csv"
    )
    print(f"Usable prediction records: {len(data)}")
    print(f"Unique participants: {data[PARTICIPANT_COLUMN].nunique()}")
    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")
    print(
        "Training participants: "
        f"{train_data[PARTICIPANT_COLUMN].nunique()}"
    )
    print(
        "Testing participants: "
        f"{test_data[PARTICIPANT_COLUMN].nunique()}"
    )
    print(f"Validation records: {len(validation_data)}")
    print(
        "Validation participants: "
        f"{validation_data[PARTICIPANT_COLUMN].nunique()}"
    )

    # Display validation results to make the selection process transparent.
    validation_summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Validation MAE": metrics["mae"],
                "Validation RMSE": metrics["rmse"],
                "Validation R2": metrics["r2"],
            }
            for model_name, metrics in validation_results.items()
        ]
    )

    print("\nVALIDATION RESULTS (MODEL SELECTION)")
    print("===================================")
    print(
        validation_summary.to_string(
            index=False,
            formatters={
                "Validation MAE": "{:.2f}".format,
                "Validation RMSE": "{:.2f}".format,
                "Validation R2": "{:.3f}".format,
            },
        )
    )
    print(f"\nSelected model using validation MAE: {best_model_name}")

    # Report final metrics on the held-out test participants.
    summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
            }
            for model_name, metrics in results.items()
        ]
    )

    print("\nFINAL TEST RESULTS")
    print("==================")
    print(summary.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.3f}".format,
        },
    ))

    # Save only the final held-out test metrics.
    if save_results:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(RESULTS_PATH, index=False)
        print(f"\nTest metrics saved to: {RESULTS_PATH}")

    print("\nLimitations:")
    print("- Uses real secondary LIBS assessment data.")
    print("- Predicts the next observed score, not a fixed future date.")
    print("- Participants are separated across training, validation, and test.")
    print("- Model selection uses validation MAE, not test MAE.")
    print("- The data do not directly measure dyslexia intervention outcomes.")
    print("- Results do not establish clinical validity.")

    return {
        "results": results,
        "summary": summary,
        "validation_results": validation_results,
        "validation_summary": validation_summary,
        "data": data,
        "features": features,
        "target": target,
        "best_model_name": best_model_name,
        "models": final_models,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
    }


def train_progress_model():
    """Train and compare models on the original synthetic dataset.

    This function preserves the earlier API for services that still
    depend on the synthetic reading-progress schema.
    """

    df = load_reading_data()

    features = [
        "reading_speed_wpm",
        "reading_accuracy",
        "fluency_score",
        "word_error_rate",
    ]
    X = df[features]
    y = df["progress_score"]

    # This record-level split is for the legacy synthetic workflow only.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    for model_name, model in models.items():
        # Train and evaluate each model on the synthetic sample.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[model_name] = {
            "model": model,
            **_evaluate_predictions(y_test, predictions),
        }

        print(f"\n{model_name}")
        print("-" * len(model_name))
        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")
        print(f"MAE: {results[model_name]['mae']:.2f}")
        print(f"RMSE: {results[model_name]['rmse']:.2f}")
        print(f"R²: {results[model_name]['r2']:.2f}")

    return results


if __name__ == "__main__":
    train_libs_progress_model()
