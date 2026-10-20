
"""Tests for the combined Component 4 learning progress report."""

import pandas as pd
import pytest

from backend.app.services.learning_progress import progress_reporting as reporting


@pytest.fixture
def sample_reading_data():
    """Provide a small longitudinal dataset for two fictional participants."""
    return pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101, 101, 202, 202],
            "reading_score": [40, 50, 60, 65],
            "time_point": [1, 2, 1, 2],
        }
    )


@pytest.fixture
def sample_model_results():
    """Provide minimal model metadata without training a real model."""
    return {"best_model_name": "Linear Regression"}


@pytest.fixture
def sample_forecasts():
    """Provide example forecasts for participants with sufficient history."""
    return pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101, 202],
            "forecasted_next_score": [55.0, 68.0],
            "forecasted_score_change": [5.0, 3.0],
            "prediction_model": ["Linear Regression", "Linear Regression"],
        }
    )


@pytest.fixture
def sample_risk_results():
    """Provide example stagnation-screening results."""
    return pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101, 202],
            "stagnation_risk": ["Low", "Medium"],
            "explanation": ["Progress observed", "Slow recent progress"],
            "average_recent_change": [10.0, 1.5],
            "number_of_observed_assessments": [2, 2],
        }
    )


@pytest.fixture
def sample_intervention_results():
    """Provide example preliminary intervention suggestions."""
    return pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101, 202],
            "recommended_activities": [
                "Continue reading practice",
                "Guided reading practice",
            ],
            "frequency": ["2 sessions per week", "3 sessions per week"],
            "session_duration": ["15 minutes", "15 minutes"],
            "recommendation_basis": [
                "Low-risk rule",
                "Medium-risk rule",
            ],
        }
    )


def configure_report_dependencies(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_forecasts,
    sample_risk_results,
    sample_intervention_results,
):
    """Replace data and model functions so tests remain fast and repeatable."""
    monkeypatch.setattr(
        reporting, "load_libs_reading_data", lambda: sample_reading_data.copy()
    )
    monkeypatch.setattr(
        reporting,
        "train_libs_progress_model",
        lambda save_results=False: sample_model_results,
    )
    monkeypatch.setattr(
        reporting,
        "forecast_future_progress",
        lambda model_results: sample_forecasts.copy(),
    )
    monkeypatch.setattr(
        reporting, "detect_stagnation", lambda: sample_risk_results.copy()
    )
    monkeypatch.setattr(
        reporting,
        "generate_intervention_plans",
        lambda risk_data: sample_intervention_results.copy(),
    )


def test_report_contains_expected_columns(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_forecasts,
    sample_risk_results,
    sample_intervention_results,
):
    """The report should include scores, forecasts, risk, and plans."""
    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        sample_forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    report = reporting.generate_progress_report()

    expected_columns = {
        "UnlinkedParticipantNumber",
        "latest_reading_score",
        "latest_time_point",
        "forecasted_next_score",
        "forecasted_score_change",
        "has_future_forecast",
        "prediction_model",
        "stagnation_risk",
        "recommended_activities",
        "frequency",
        "session_duration",
    }

    assert expected_columns.issubset(report.columns)


def test_report_uses_latest_assessment_per_participant(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_forecasts,
    sample_risk_results,
    sample_intervention_results,
):
    """Only the latest observed score should appear for each participant."""
    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        sample_forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    report = reporting.generate_progress_report().set_index(
        "UnlinkedParticipantNumber"
    )

    assert len(report) == 2
    assert report.loc[101, "latest_reading_score"] == 50
    assert report.loc[101, "latest_time_point"] == 2
    assert report.loc[202, "latest_reading_score"] == 65
    assert report.loc[202, "latest_time_point"] == 2


def test_report_combines_forecasts_risk_and_interventions(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_forecasts,
    sample_risk_results,
    sample_intervention_results,
):
    """Each participant should receive their matching forecast and plan."""
    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        sample_forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    report = reporting.generate_progress_report().set_index(
        "UnlinkedParticipantNumber"
    )

    assert report.loc[101, "forecasted_next_score"] == 55.0
    assert report.loc[101, "stagnation_risk"] == "Low"
    assert (
        report.loc[101, "recommended_activities"]
        == "Continue reading practice"
    )

    assert report.loc[202, "forecasted_next_score"] == 68.0
    assert report.loc[202, "stagnation_risk"] == "Medium"
    assert (
        report.loc[202, "recommended_activities"]
        == "Guided reading practice"
    )


def test_report_marks_missing_forecasts(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_risk_results,
    sample_intervention_results,
):
    """Participants without a forecast should be marked correctly."""
    forecasts = pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101],
            "forecasted_next_score": [55.0],
            "forecasted_score_change": [5.0],
            "prediction_model": ["Linear Regression"],
        }
    )

    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    report = reporting.generate_progress_report().set_index(
        "UnlinkedParticipantNumber"
    )

    assert bool(report.loc[101, "has_future_forecast"]) is True
    assert bool(report.loc[202, "has_future_forecast"]) is False
    assert pd.isna(report.loc[202, "forecasted_next_score"])

    # Missing forecast labels should fall back to the selected model name.
    assert report.loc[202, "prediction_model"] == "Linear Regression"


def test_report_is_sorted_by_participant_id(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_forecasts,
    sample_risk_results,
    sample_intervention_results,
):
    """The final report should be sorted by participant identifier."""
    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        sample_forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    report = reporting.generate_progress_report()

    assert report["UnlinkedParticipantNumber"].tolist() == [101, 202]


def test_report_rejects_incomplete_forecast_schema(
    monkeypatch,
    sample_reading_data,
    sample_model_results,
    sample_risk_results,
    sample_intervention_results,
):
    """Missing required forecast columns should raise a clear pandas error."""
    incomplete_forecasts = pd.DataFrame(
        {
            "UnlinkedParticipantNumber": [101, 202],
            "forecasted_next_score": [55.0, 68.0],
        }
    )

    configure_report_dependencies(
        monkeypatch,
        sample_reading_data,
        sample_model_results,
        incomplete_forecasts,
        sample_risk_results,
        sample_intervention_results,
    )

    with pytest.raises(KeyError):
        reporting.generate_progress_report()
