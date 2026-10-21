
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
    assert "forecast_status" in result.columns
    assert "forecast_explanation" in result.columns
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
    assert "forecast_status" in result.columns
    assert "forecast_explanation" in result.columns
    assert "forecasted_next_score" in result.columns


def test_prediction_split_has_no_participant_overlap():
    """Inspect groups from the actual training operation, not copied split code."""
    result = progress_prediction.train_libs_progress_model(save_results=False)
    column = progress_prediction.PARTICIPANT_COLUMN
    fit_ids = set(result["fit_data"][column])
    validation_ids = set(result["validation_data"][column])
    test_ids = set(result["test_data"][column])
    assert fit_ids.isdisjoint(validation_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert fit_ids | validation_ids == set(result["train_data"][column])



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

    # Inspect the participants actually used by the evaluation function.
    test_ids = result["test_participants"]
    assert len(test_ids) == 4
    for split in result["split_participants"]:
        fit_ids = split["fit_participants"]
        validation_ids = split["validation_participants"]
        assert fit_ids.isdisjoint(validation_ids)
        assert fit_ids.isdisjoint(test_ids)
        assert validation_ids.isdisjoint(test_ids)
        assert fit_ids | validation_ids | test_ids == set(range(1, 21))

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


def test_prediction_intervals_return_full_range_for_small_calibration_sample(
    monkeypatch,
):
    """An unavailable finite quantile should produce the full score range."""

    # Create fictional assessment records for 10 participants.
    rows = []
    for participant_id in range(1, 11):
        rows.extend(
            [
                (participant_id, 1, 20 + participant_id),
                (participant_id, 2, 25 + participant_id),
                (participant_id, 3, 30 + participant_id),
            ]
        )

    fake_data = make_reading_data(rows)

    # Replace the real dataset with predictable test data.
    monkeypatch.setattr(
        progress_prediction,
        "load_libs_reading_data",
        lambda: fake_data.copy(),
    )

    # With two calibration participants, ceil(3 * 0.999) = 3 exceeds
    # the two available errors. The infinite threshold becomes [0, 90].
    result = progress_prediction.evaluate_prediction_intervals(
        calibration_size=0.20,
        test_size=0.20,
        confidence_level=0.999,
    )
    summary = result["summary"]
    intervals = result["test_results"]

    assert summary["calibration_participants"] == 2
    assert np.isposinf(summary["calibration_threshold"])
    assert not intervals.empty
    assert intervals["lower_bound"].eq(0).all()
    assert intervals["upper_bound"].eq(90).all()
    assert intervals["interval_width"].eq(90).all()
    assert intervals["covered"].all()
    assert summary["mean_interval_width"] == pytest.approx(90)
    assert summary["record_coverage"] == pytest.approx(1)
    assert summary["participant_coverage"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("confidence_level", "expected_rank"),
    [(0.40, 2), (0.80, 4)],
)
def test_prediction_intervals_use_ranked_participant_maximum_errors(
    monkeypatch, confidence_level, expected_rank,
):
    """Check participant aggregation and finite-sample ranks exactly."""

    # Each fictional participant has two prediction records. A constant
    # prediction of 50 gives errors [0, participant_id], so the maximum
    # error per participant is known without reproducing production logic.
    rows = []
    for participant_id in range(1, 11):
        for time_point, target_score in [(2, 50), (3, 50 + participant_id)]:
            rows.append(
                {
                    progress_prediction.PARTICIPANT_COLUMN: participant_id,
                    "previous_score": 50,
                    "assessment_score": 50,
                    "previous_time_point": time_point - 1,
                    "time_point": time_point,
                    "target_time_point": time_point + 1,
                    "target_score": target_score,
                }
            )
    fake_data = pd.DataFrame(rows)

    class ConstantModel:
        """Keep predictions fixed so model randomness cannot hide errors."""

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.full(len(features), 50.0)

    monkeypatch.setattr(
        progress_prediction,
        "prepare_libs_prediction_data",
        lambda: fake_data.copy(),
    )
    monkeypatch.setattr(
        progress_prediction,
        "_create_models",
        lambda: {"Random Forest": ConstantModel()},
    )

    result = progress_prediction.evaluate_prediction_intervals(
        calibration_size=0.40,
        test_size=0.20,
        confidence_level=confidence_level,
    )
    summary = result["summary"]
    intervals = result["test_results"]

    # Four calibration participants give ranks ceil(5 * 0.40) = 2 and
    # ceil(5 * 0.80) = 4. The latter is the last supported finite rank.
    assert summary["calibration_participants"] == 4
    known_errors = sorted(result["calibration_participants"])
    expected_threshold = known_errors[expected_rank - 1]
    assert summary["calibration_threshold"] == pytest.approx(expected_threshold)
    assert intervals["lower_bound"].eq(50 - expected_threshold).all()
    assert intervals["upper_bound"].eq(50 + expected_threshold).all()
    assert summary["mean_interval_width"] == pytest.approx(2 * expected_threshold)

    # The first record is always covered. The second is covered exactly
    # when its participant ID (its known error) is within the threshold.
    expected_covered = (
        intervals["target_time_point"].eq(3)
        | intervals[progress_prediction.PARTICIPANT_COLUMN].le(expected_threshold)
    )
    assert intervals["covered"].equals(expected_covered)
    expected_participant_coverage = sum(
        participant_id <= expected_threshold
        for participant_id in result["test_participants"]
    ) / len(result["test_participants"])
    assert summary["participant_coverage"] == pytest.approx(
        expected_participant_coverage
    )
    assert summary["record_coverage"] == pytest.approx(
        (1 + expected_participant_coverage) / 2
    )


def test_evaluation_workflows_share_default_test_participants(monkeypatch):
    """Main, repeated, and interval evaluation must reserve identical learners."""
    data = make_reading_data([
        (participant_id, time_point, 10 + participant_id + 5 * time_point)
        for participant_id in range(1, 21)
        for time_point in range(1, 5)
    ])
    monkeypatch.setattr(
        progress_prediction, "load_libs_reading_data", lambda: data.copy(),
    )
    main = progress_prediction.train_libs_progress_model(save_results=False)
    repeated = progress_prediction.evaluate_repeated_group_validation(n_splits=2)
    intervals = progress_prediction.evaluate_prediction_intervals()
    test_ids = set(main["test_data"][progress_prediction.PARTICIPANT_COLUMN])
    assert test_ids == repeated["test_participants"] == intervals["test_participants"]
    assert test_ids.isdisjoint(intervals["fit_participants"])
    assert test_ids.isdisjoint(intervals["calibration_participants"])


def test_repeated_validation_never_passes_test_records_to_models(monkeypatch):
    """Observe actual fit/predict calls, rather than only returned split metadata."""
    data = make_reading_data([
        (participant_id, time_point, participant_id)
        for participant_id in range(1, 21)
        for time_point in range(1, 4)
    ])
    monkeypatch.setattr(
        progress_prediction, "load_libs_reading_data", lambda: data.copy(),
    )
    seen_fit, seen_predict = [], []

    class RecordingModel:
        def fit(self, features, target):
            seen_fit.append(set(features["assessment_score"]))
            return self

        def predict(self, features):
            seen_predict.append(set(features["assessment_score"]))
            return features["assessment_score"].to_numpy()

    monkeypatch.setattr(
        progress_prediction, "_create_models",
        lambda: {"Recording Model": RecordingModel()},
    )
    result = progress_prediction.evaluate_repeated_group_validation(n_splits=2)
    assert len(seen_fit) == len(seen_predict) == 2
    for fit_ids, validation_ids in zip(seen_fit, seen_predict):
        assert fit_ids.isdisjoint(validation_ids)
        assert result["test_participants"].isdisjoint(fit_ids | validation_ids)
        assert fit_ids | validation_ids | result["test_participants"] == set(range(1, 21))


def test_repeated_validation_requires_development_and_test_groups(monkeypatch):
    """Two participants cannot provide separate fit, validation, and test groups."""
    data = make_reading_data([
        (participant_id, time_point, 10 * time_point)
        for participant_id in (1, 2)
        for time_point in (1, 2, 3)
    ])
    monkeypatch.setattr(
        progress_prediction, "load_libs_reading_data", lambda: data.copy(),
    )
    with pytest.raises(ValueError, match="At least three participants"):
        progress_prediction.evaluate_repeated_group_validation()


@pytest.mark.parametrize(
    ("latest_time", "expected_status"),
    [(2, "observed_training_time_point"),
     (3, "observed_training_time_point"),
     (4, "unseen_training_time_point")],
)
def test_forecast_status_uses_fitting_time_points_only(
    monkeypatch, latest_time, expected_status,
):
    """A time point seen only in held-out data must still be flagged."""
    data = make_reading_data([(1, 1, 20), (1, latest_time, 40)])
    monkeypatch.setattr(
        progress_prediction, "load_libs_reading_data", lambda: data.copy(),
    )
    model_results = make_model_results(FakeModel([50]))
    model_results["train_data"] = pd.DataFrame({"time_point": [2, 3]})
    model_results["test_data"] = pd.DataFrame({"time_point": [4]})
    result = progress_prediction.forecast_future_progress(model_results)
    assert result.iloc[0]["forecast_status"] == expected_status
    assert "Experimental estimate" in result.iloc[0]["forecast_explanation"]
    assert result.iloc[0]["forecasted_next_score"] == 50


@pytest.mark.parametrize("training_data", [None, pd.DataFrame({"time_point": []})])
def test_forecast_status_does_not_assume_missing_training_metadata(
    monkeypatch, training_data,
):
    """Missing metadata must not be interpreted as supported forecasting."""
    data = make_reading_data([(1, 1, 20), (1, 2, 40)])
    monkeypatch.setattr(
        progress_prediction, "load_libs_reading_data", lambda: data.copy(),
    )
    model_results = make_model_results(FakeModel([50]))
    model_results["train_data"] = training_data
    result = progress_prediction.forecast_future_progress(model_results)
    assert result.iloc[0]["forecast_status"] == "training_time_points_unknown"
    assert "cannot be checked" in result.iloc[0]["forecast_explanation"]


def test_features_preserve_chronology_and_never_include_future_targets(monkeypatch):
    data = make_reading_data([
        (2, 4, 60), (1, 3, 30), (2, 1, 0), (1, 1, 0),
        (2, 3, 40), (1, 2, 10),
    ])
    monkeypatch.setattr(progress_prediction, "load_libs_reading_data", lambda: data.copy())
    result = progress_prediction.prepare_libs_prediction_data()
    assert result["previous_score"].tolist() == [0, 0]
    assert result["assessment_score"].tolist() == [10, 40]
    assert result["target_score"].tolist() == [30, 60]
    assert result["time_point"].tolist() == [2, 3]
    assert result["target_time_point"].tolist() == [3, 4]


def test_holdout_membership_survives_row_reordering():
    data = pd.DataFrame({progress_prediction.PARTICIPANT_COLUMN: list(range(20)) * 2})
    first = progress_prediction._split_development_and_test(data)
    shuffled = progress_prediction._split_development_and_test(data.sample(frac=1, random_state=7))
    for expected, actual in zip(first, shuffled):
        assert set(expected[progress_prediction.PARTICIPANT_COLUMN]) == set(actual[progress_prediction.PARTICIPANT_COLUMN])


def test_interval_split_rounding_reports_empty_fitting_group(monkeypatch):
    data = make_reading_data([
        (person, time, 10 * time) for person in range(10) for time in (1, 2, 3)
    ])
    monkeypatch.setattr(progress_prediction, 'load_libs_reading_data', lambda: data.copy())
    with pytest.raises(ValueError, match='no fitting participants'):
        progress_prediction.evaluate_prediction_intervals(test_size=0.90, calibration_size=0.05)


def test_model_choice_uses_validation_even_when_test_ranking_reverses(monkeypatch):
    # Encode IDs in current scores solely to observe which rows are predicted.
    data = pd.DataFrame({
        progress_prediction.PARTICIPANT_COLUMN: range(1, 21),
        'previous_score': [10] * 20, 'assessment_score': range(1, 21),
        'previous_time_point': [1] * 20, 'time_point': [2] * 20,
        'target_time_point': [3] * 20, 'target_score': list(range(31, 51)),
    })
    _, test = progress_prediction._split_development_and_test(data)
    test_ids = set(test[progress_prediction.PARTICIPANT_COLUMN])

    class ReversedRankingModel:
        def __init__(self, validation_winner):
            self.validation_winner = validation_winner

        def fit(self, features, targets):
            return self

        def predict(self, features):
            in_test = features.assessment_score.isin(test_ids).to_numpy()
            # A wins validation and B wins test. Selection must still choose A.
            error = np.where(in_test == self.validation_winner, 10, 0)
            return features.assessment_score.to_numpy() + 30 + error

    monkeypatch.setattr(progress_prediction, 'prepare_libs_prediction_data', lambda: data.copy())
    monkeypatch.setattr(progress_prediction, '_create_models', lambda: {
        'Validation winner': ReversedRankingModel(True),
        'Test winner': ReversedRankingModel(False),
    })
    result = progress_prediction.train_libs_progress_model(save_results=False)
    assert result['best_model_name'] == 'Validation winner'
    assert result['results']['Test winner']['mae'] < result['results']['Validation winner']['mae']
    assert 'target_score' not in result['features']
    assert 'target_time_point' not in result['features']
