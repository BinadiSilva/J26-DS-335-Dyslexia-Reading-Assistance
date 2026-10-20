
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

    # Use previous and current observed scores as predictors.
    df["previous_score"] = grouped_scores.shift(1)
    df["target_score"] = grouped_scores.shift(-1)

    # Preserve the assessment time points for prediction.
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

    # Remove records missing a required predictor, target, or participant ID.
    usable = df.dropna(
        subset=features + ["target_score", PARTICIPANT_COLUMN]
    ).copy()

    usable = usable.rename(
        columns={"reading_score": "assessment_score"}
    )

    # Return the columns required by training, evaluation, and reporting.
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


def evaluate_repeated_group_validation(n_splits=5, test_size=0.20):
    """Evaluate candidate models over repeated participant-level splits.

    This additional evaluation measures how model performance varies across
    different participant groups. It does not replace the separate final
    held-out test evaluation performed by train_libs_progress_model().
    """

    # Validate the requested evaluation settings.
    if not isinstance(n_splits, int) or isinstance(n_splits, bool):
        raise ValueError("n_splits must be an integer of at least 2.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Load the same longitudinal examples used by the main training workflow.
    data = prepare_libs_prediction_data()

    if data.empty or data[PARTICIPANT_COLUMN].nunique() < 2:
        raise ValueError(
            "At least two participants are required for repeated validation."
        )

    features = [
        "previous_score",
        "assessment_score",
        "previous_time_point",
        "time_point",
    ]
    target = "target_score"

    # Create reproducible splits that keep each participant in one group
    # within each split.
    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=RANDOM_STATE + 10,
    )

    records = []

    for split_number, (fit_indices, validation_indices) in enumerate(
        splitter.split(
            data[features],
            data[target],
            groups=data[PARTICIPANT_COLUMN],
        ),
        start=1,
    ):
        # Separate the fitting and validation records for this round.
        fit_data = data.iloc[fit_indices]
        validation_data = data.iloc[validation_indices]

        # Verify that the same participant does not occur in both groups.
        fit_ids = set(fit_data[PARTICIPANT_COLUMN])
        validation_ids = set(validation_data[PARTICIPANT_COLUMN])

        if not fit_ids.isdisjoint(validation_ids):
            raise RuntimeError(
                "Participant overlap detected in repeated validation."
            )

        X_fit = fit_data[features]
        y_fit = fit_data[target]
        X_validation = validation_data[features]
        y_validation = validation_data[target]

        # Evaluate the persistence baseline: predict no change from the
        # current assessment score.
        baseline_predictions = X_validation["assessment_score"].to_numpy()
        baseline_metrics = _evaluate_predictions(
            y_validation,
            baseline_predictions,
        )

        records.append(
            {
                "split": split_number,
                "model": "Persistence Baseline",
                **baseline_metrics,
                "fit_participants": len(fit_ids),
                "validation_participants": len(validation_ids),
            }
        )

        # Fit new instances of both candidate models for this split.
        for model_name, model in _create_models().items():
            model.fit(X_fit, y_fit)
            predictions = model.predict(X_validation)

            records.append(
                {
                    "split": split_number,
                    "model": model_name,
                    **_evaluate_predictions(y_validation, predictions),
                    "fit_participants": len(fit_ids),
                    "validation_participants": len(validation_ids),
                }
            )

    # Keep the per-split results for inspection and reproducibility.
    detailed_results = pd.DataFrame(records)

    # Summarize average performance and variation across the validation splits.
    summary = (
        detailed_results.groupby("model")
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
        )
        .reset_index()
    )

    print("\nREPEATED PARTICIPANT-LEVEL VALIDATION")
    print("=" * 44)
    print(f"Validation splits: {n_splits}")
    print("Mean values describe average performance; standard deviations")
    print("describe variation between splits.\n")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    return {
        "detailed_results": detailed_results,
        "summary": summary,
    }



def evaluate_prediction_intervals(
    calibration_size=0.20,
    test_size=0.20,
    confidence_level=0.90,
):
    """Evaluate experimental prediction intervals on separate participants.

    Calibration uses each participant's maximum absolute residual, so the
    threshold accounts for repeated records within calibration participants.
    This is an experimental evaluation, not clinical validation.
    """

    # Validate the requested proportions and confidence level.
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be between 0 and 1.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if calibration_size + test_size >= 1:
        raise ValueError(
            "calibration_size + test_size must be less than 1."
        )

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    # Load prediction examples and define the existing model features.
    data = prepare_libs_prediction_data()
    features = [
        "previous_score",
        "assessment_score",
        "previous_time_point",
        "time_point",
    ]
    target = "target_score"

    # Require enough participants to form three separate groups.
    participant_count = data[PARTICIPANT_COLUMN].nunique()
    if data.empty or participant_count < 10:
        raise ValueError(
            "At least 10 participants are required for interval evaluation."
        )

    # First reserve the test participants, which are not used for fitting
    # or calibration.
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE + 20,
    )
    development_indices, test_indices = next(
        outer_splitter.split(
            data[features],
            data[target],
            groups=data[PARTICIPANT_COLUMN],
        )
    )

    development_data = data.iloc[development_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Separate calibration participants from model-fitting participants.
    relative_calibration_size = calibration_size / (1 - test_size)
    calibration_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_calibration_size,
        random_state=RANDOM_STATE + 21,
    )
    fit_indices, calibration_indices = next(
        calibration_splitter.split(
            development_data[features],
            development_data[target],
            groups=development_data[PARTICIPANT_COLUMN],
        )
    )

    fit_data = development_data.iloc[fit_indices].copy()
    calibration_data = development_data.iloc[calibration_indices].copy()

    # Verify that no participant crosses a split boundary.
    fit_ids = set(fit_data[PARTICIPANT_COLUMN])
    calibration_ids = set(calibration_data[PARTICIPANT_COLUMN])
    test_ids = set(test_data[PARTICIPANT_COLUMN])

    if (
        not fit_ids.isdisjoint(calibration_ids)
        or not fit_ids.isdisjoint(test_ids)
        or not calibration_ids.isdisjoint(test_ids)
    ):
        raise RuntimeError(
            "Participant overlap detected across interval evaluation groups."
        )

    # Fit a fresh Random Forest using only the fitting participants.
    model = _create_models()["Random Forest"]
    model.fit(fit_data[features], fit_data[target])

    # Estimate calibration errors without using test participants.
    calibration_predictions = model.predict(calibration_data[features])
    calibration_errors = calibration_data[[PARTICIPANT_COLUMN]].copy()
    calibration_errors["absolute_error"] = np.abs(
        calibration_data[target].to_numpy() - calibration_predictions
    )

    # Use each participant's maximum residual to account for repeated
    # assessment records belonging to that participant.
    participant_errors = (
        calibration_errors.groupby(PARTICIPANT_COLUMN)["absolute_error"]
        .max()
        .to_numpy()
    )

    # Use the finite-sample conformal quantile rank.
    calibration_count = len(participant_errors)
    quantile_rank = int(
        np.ceil((calibration_count + 1) * confidence_level)
    )

    if quantile_rank > calibration_count:
        raise ValueError(
            "Too few calibration participants for this confidence level."
        )

    threshold = float(
        np.sort(participant_errors)[quantile_rank - 1]
    )

    # Evaluate the resulting intervals only on the held-out test group.
    test_predictions = model.predict(test_data[features])
    actual_scores = test_data[target].to_numpy()

    lower_bounds = np.clip(test_predictions - threshold, 0, 90)
    upper_bounds = np.clip(test_predictions + threshold, 0, 90)

    test_results = test_data[
        [PARTICIPANT_COLUMN, target, "target_time_point"]
    ].copy()
    test_results["predicted_score"] = test_predictions
    test_results["lower_bound"] = lower_bounds
    test_results["upper_bound"] = upper_bounds
    test_results["covered"] = (
        (actual_scores >= lower_bounds)
        & (actual_scores <= upper_bounds)
    )
    test_results["interval_width"] = upper_bounds - lower_bounds

    # Report record-level coverage and width, plus the stricter participant-
    # level measure: whether all tested records for a participant are covered.
    participant_coverage = (
        test_results.groupby(PARTICIPANT_COLUMN)["covered"].all()
    )

    summary = {
        "confidence_level": confidence_level,
        "calibration_participants": len(calibration_ids),
        "test_participants": len(test_ids),
        "calibration_threshold": threshold,
        "record_coverage": float(test_results["covered"].mean()),
        "participant_coverage": float(participant_coverage.mean()),
        "mean_interval_width": float(test_results["interval_width"].mean()),
        "test_records": len(test_results),
    }

    print("\nEXPERIMENTAL PREDICTION INTERVAL EVALUATION")
    print("=" * 46)
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")

    print(
        "\nLimitations: these intervals are experimental, use LIBS data, "
        "and do not establish clinical validity or guaranteed coverage "
        "for future participants."
    )

    return {
        "summary": summary,
        "test_results": test_results,
        "model": model,
        "features": features,
        "fit_participants": fit_ids,
        "calibration_participants": calibration_ids,
        "test_participants": test_ids,
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

    # Reserve 20% of participants for the final test set.
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

    # Split the remaining participants into model-fitting and validation groups.
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

    # Compare candidate models on the validation participants.
    validation_results = {}
    validation_models = _create_models()

    # Include the persistence baseline for comparison.
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

    # Select the candidate with the lowest validation MAE.
    best_model_name = min(
        validation_models,
        key=lambda name: validation_results[name]["mae"],
    )

    # Refit fresh candidate models on all outer training participants.
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
        # Refit on all outer training participants, then test once.
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

    # Display validation metrics used for model selection.
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
    print(
        summary.to_string(
            index=False,
            formatters={
                "MAE": "{:.2f}".format,
                "RMSE": "{:.2f}".format,
                "R2": "{:.3f}".format,
            },
        )
    )

    # Save only the final held-out test metrics when requested.
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


def forecast_future_progress(model_results):
    """Forecast the next observed reading score beyond each learner's history.

    Participants need at least two observed assessments to provide the
    previous-score features used during training.
    """

    # Load validated LIBS data and sort assessments chronologically.
    df = load_libs_reading_data()
    df = df.sort_values(
        [PARTICIPANT_COLUMN, "time_point"]
    ).copy()

    # Build features using previous and latest observed scores.
    grouped = df.groupby(PARTICIPANT_COLUMN, sort=False)
    df["previous_score"] = grouped["reading_score"].shift(1)
    df["previous_time_point"] = grouped["time_point"].shift(1)

    # Keep each participant's latest assessment.
    latest = (
        df.groupby(PARTICIPANT_COLUMN, sort=False)
        .tail(1)
        .copy()
    )

    # Exclude participants who lack the history required by the model.
    latest = latest.dropna(
        subset=["previous_score", "previous_time_point"]
    )

    features = model_results["features"]
    model_name = model_results["best_model_name"]
    model = model_results["results"][model_name]["model"]

    # Match the feature names used during model training.
    latest["assessment_score"] = latest["reading_score"]

    if latest.empty or model is None:
        return pd.DataFrame(
            columns=[
                PARTICIPANT_COLUMN,
                "latest_reading_score",
                "latest_time_point",
                "forecasted_next_score",
                "forecasted_score_change",
                "prediction_model",
            ]
        )

    # Predict the next observed score beyond each latest assessment.
    X_future = latest[features]
    predicted_scores = model.predict(X_future)

    # Keep forecasts within the documented BAS score range.
    latest["forecasted_next_score"] = np.clip(
        predicted_scores, 0, 90
    ).round(2)

    # Calculate the predicted change from the latest observed score.
    latest["forecasted_score_change"] = (
        latest["forecasted_next_score"] - latest["reading_score"]
    ).round(2)

    # Return forecast values separately from historical test metrics.
    forecasts = latest[
        [
            PARTICIPANT_COLUMN,
            "reading_score",
            "time_point",
            "forecasted_next_score",
            "forecasted_score_change",
        ]
    ].rename(
        columns={
            "reading_score": "latest_reading_score",
            "time_point": "latest_time_point",
        }
    )

    forecasts["prediction_model"] = model_name

    return forecasts.reset_index(drop=True)


if __name__ == "__main__":
    # Preserve the existing default behaviour when running this file directly.
    train_libs_progress_model()
