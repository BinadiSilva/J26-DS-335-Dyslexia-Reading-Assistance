
"""Unit tests for Component 4 future-progress forecasting."""

import numpy as np
import pandas as pd
import pytest

from backend.app.services.learning_progress import progress_prediction


class FakeModel:
    """Provide controlled predictions without training a real ML model."""

    def __init__(self, predictions):
        # Store predictable values for testing forecast behaviour.
        self.predictions = predictions
        self.received_features = None

    def predict(self, features):
        # Record model inputs and return the predetermined scores.
        self.received_features = features.copy()
        return np.array(self.predictions[:len(features)])


def make_model_results(model):
    """Build the result structure expected by forecast_future_progress()."""

    return {
        "features": [
            "previous_score",
            "assessment_score",
            "previous_time_point",
            "time_point",
        ],
        "best_model_name": "Test Model",
        "results": {
            "Test Model": {
                "model": model,
            }
        },
    }


def make_reading_data(rows):
    """Create fictional longitudinal assessment records."""

    return pd.DataFrame(
        rows,
        columns=[
            "UnlinkedParticipantNumber",
            "time_point",
            "reading_score",
        ],
    )


def test_forecast_uses_each_participants_latest_assessment(monkeypatch):
    """Forecasts should use the latest observed score for each participant."""

    # Participant 1 has two observations; participant 2 has three.
    data = make_reading_data(
        [
            (1, 1, 40),
            (1, 2, 50),
            (2, 1, 30),
            (2, 2, 35),
            (2, 3, 45),
        ]
    )
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: data.copy(),
    )

    model = FakeModel([55, 48])
    result = progress_prediction.forecast_future_progress(
        make_model_results(model)
    )

    # Both participants should have one forecast from their latest history.
    assert len(result) == 2
    assert result["UnlinkedParticipantNumber"].tolist() == [1, 2]
    assert result["latest_reading_score"].tolist() == [50, 45]
    assert result["forecasted_next_score"].tolist() == [55.0, 48.0]
    assert result["prediction_model"].eq("Test Model").all()


def test_single_assessment_participant_is_excluded(monkeypatch):
    """A participant without previous-score history should be excluded."""

    # Participant 1 has one score; participant 2 has two scores.
    data = make_reading_data(
        [
            (1, 1, 40),
            (2, 1, 30),
            (2, 2, 35),
        ]
    )
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: data.copy(),
    )

    model = FakeModel([42])
    result = progress_prediction.forecast_future_progress(
        make_model_results(model)
    )

    # Only participant 2 has enough history for a forecast.
    assert result["UnlinkedParticipantNumber"].tolist() == [2]


@pytest.mark.parametrize(
    ("model_prediction", "expected_score"),
    [
        (-10, 0.0),   # Predictions below the valid range are clipped.
        (45, 45.0),   # Predictions inside the valid range are retained.
        (100, 90.0),  # Predictions above the valid range are clipped.
    ],
)
def test_forecast_score_is_clipped_to_valid_range(
    monkeypatch,
    model_prediction,
    expected_score,
):
    """Forecast scores should remain within the documented 0–90 range."""

    data = make_reading_data(
        [
            (1, 1, 40),
            (1, 2, 50),
        ]
    )
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: data.copy(),
    )

    model = FakeModel([model_prediction])
    result = progress_prediction.forecast_future_progress(
        make_model_results(model)
    )

    assert result.iloc[0]["forecasted_next_score"] == expected_score
    assert 0 <= result.iloc[0]["forecasted_next_score"] <= 90


def test_forecast_change_is_calculated_correctly(monkeypatch):
    """The reported score change should equal forecast minus latest score."""

    data = make_reading_data(
        [
            (1, 1, 40),
            (1, 2, 50),
        ]
    )
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: data.copy(),
    )

    model = FakeModel([60])
    result = progress_prediction.forecast_future_progress(
        make_model_results(model)
    )

    assert result.iloc[0]["forecasted_score_change"] == 10.0


def test_empty_dataset_returns_empty_forecast(monkeypatch):
    """An empty dataset should return an empty result with expected columns."""

    empty_data = make_reading_data([])
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: empty_data.copy(),
    )

    model = FakeModel([])
    result = progress_prediction.forecast_future_progress(
        make_model_results(model)
    )

    # No participant history means no forecasts can be generated.
    assert result.empty
    assert "forecasted_next_score" in result.columns
    assert "forecasted_score_change" in result.columns


def test_missing_model_returns_empty_forecast(monkeypatch):
    """A missing model should produce an empty forecast result."""

    data = make_reading_data(
        [
            (1, 1, 40),
            (1, 2, 50),
        ]
    )
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: data.copy(),
    )

    result = progress_prediction.forecast_future_progress(
        make_model_results(None)
    )

    assert result.empty
    assert "forecasted_next_score" in result.columns


def test_prediction_split_has_no_participant_overlap():
    """Training, validation, and test participants must be disjoint."""

    # Recreate the same deterministic participant-level splits.
    from sklearn.model_selection import GroupShuffleSplit

    data = progress_prediction.prepare_libs_prediction_data()
    participant_column = progress_prediction.PARTICIPANT_COLUMN

    # Reserve the final test participants first.
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=progress_prediction.RANDOM_STATE,
    )
    train_indices, test_indices = next(
        outer_splitter.split(
            data,
            data["target_score"],
            groups=data[participant_column],
        )
    )

    outer_train = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    # Split the outer training participants into fitting and validation groups.
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=progress_prediction.RANDOM_STATE + 1,
    )
    fit_indices, validation_indices = next(
        inner_splitter.split(
            outer_train,
            outer_train["target_score"],
            groups=outer_train[participant_column],
        )
    )

    fit_data = outer_train.iloc[fit_indices]
    validation_data = outer_train.iloc[validation_indices]

    # Compare participant IDs, not record counts.
    fit_ids = set(fit_data[participant_column])
    validation_ids = set(validation_data[participant_column])
    test_ids = set(test_data[participant_column])

    assert fit_ids.isdisjoint(validation_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_best_model_is_selected_by_validation_mae():
    """The selected model should have the lowest candidate validation MAE."""

    # Train and evaluate without writing a results CSV.
    result = progress_prediction.train_libs_progress_model(
        save_results=False
    )

    candidate_results = result["validation_results"]

    # The persistence baseline is a comparator, not a candidate model.
    candidate_models = {
        name: metrics
        for name, metrics in candidate_results.items()
        if name in result["models"]
    }

    expected_best_model = min(
        candidate_models,
        key=lambda name: candidate_models[name]["mae"],
    )

    assert result["best_model_name"] == expected_best_model



def test_repeated_validation_returns_all_models_and_splits():
    """Repeated validation should report all models for each split."""

    # Run the five repeated participant-level validation splits.
    result = progress_prediction.evaluate_repeated_group_validation(
        n_splits=5
    )

    detailed = result["detailed_results"]
    summary = result["summary"]

    # Three models should be evaluated in each of five splits.
    assert len(detailed) == 15
    assert set(detailed["split"]) == {1, 2, 3, 4, 5}
    assert set(detailed["model"]) == {
        "Persistence Baseline",
        "Linear Regression",
        "Random Forest",
    }

    # The summary should contain one row per model.
    assert len(summary) == 3
    assert detailed[["mae", "rmse", "r2"]].notna().all().all()


def test_repeated_validation_keeps_participants_separate(monkeypatch):
    """Each validation split should keep participant groups separate."""

    # Create fictional longitudinal records for 20 participants.
    rows = []
    for participant_id in range(1, 21):
        rows.extend(
            [
                (participant_id, 1, 20 + participant_id),
                (participant_id, 2, 25 + participant_id),
                (participant_id, 3, 30 + participant_id),
            ]
        )

    fake_data = make_reading_data(rows)

    # Use the fictional data instead of loading the LIBS dataset.
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: fake_data.copy(),
    )

    result = progress_prediction.evaluate_repeated_group_validation(
        n_splits=5,
        test_size=0.20,
    )

    # Recreate the splits to verify participant separation.
    from sklearn.model_selection import GroupShuffleSplit

    data = progress_prediction.prepare_libs_prediction_data()
    participant_column = progress_prediction.PARTICIPANT_COLUMN

    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=progress_prediction.RANDOM_STATE + 10,
    )

    for fit_indices, validation_indices in splitter.split(
        data,
        data["target_score"],
        groups=data[participant_column],
    ):
        fit_ids = set(data.iloc[fit_indices][participant_column])
        validation_ids = set(
            data.iloc[validation_indices][participant_column]
        )

        assert fit_ids.isdisjoint(validation_ids)

    # All five validation rounds should have been evaluated.
    assert set(result["detailed_results"]["split"]) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("n_splits", [0, 1])
def test_repeated_validation_rejects_too_few_splits(n_splits):
    """Repeated validation should reject fewer than two splits."""

    # Invalid settings should raise an informative error.
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        progress_prediction.evaluate_repeated_group_validation(
            n_splits=n_splits
        )


def test_repeated_validation_rejects_invalid_test_size():
    """Repeated validation should reject an invalid validation fraction."""

    # A validation fraction of 1.0 leaves no fitting participants.
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        progress_prediction.evaluate_repeated_group_validation(
            n_splits=5,
            test_size=1.0,
        )


def test_prediction_intervals_keep_participants_separate():
    """Fitting, calibration, and test participants must not overlap."""

    # Run the experimental interval evaluation.
    result = progress_prediction.evaluate_prediction_intervals()

    # Verify that the participant groups are disjoint.
    fit_ids = result["fit_participants"]
    calibration_ids = result["calibration_participants"]
    test_ids = result["test_participants"]

    assert fit_ids.isdisjoint(calibration_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)


def test_prediction_intervals_have_valid_bounds():
    """Every lower bound must be no greater than its upper bound."""

    # Evaluate intervals on held-out participants.
    result = progress_prediction.evaluate_prediction_intervals()
    intervals = result["test_results"]

    assert (intervals["lower_bound"] >= 0).all()
    assert (intervals["upper_bound"] <= 90).all()
    assert (intervals["lower_bound"] <= intervals["upper_bound"]).all()


def test_prediction_interval_metrics_are_valid():
    """Coverage and interval width must be within valid ranges."""

    # Obtain the interval evaluation summary.
    result = progress_prediction.evaluate_prediction_intervals()
    summary = result["summary"]

    assert 0 <= summary["record_coverage"] <= 1
    assert 0 <= summary["participant_coverage"] <= 1
    assert summary["mean_interval_width"] >= 0
    assert summary["test_records"] > 0


def test_prediction_intervals_reject_invalid_confidence_level():
    """An invalid confidence level should raise a clear error."""

    # Confidence levels must be strictly between zero and one.
    with pytest.raises(
        ValueError,
        match="confidence_level must be between 0 and 1",
    ):
        progress_prediction.evaluate_prediction_intervals(
            confidence_level=1.0
        )
