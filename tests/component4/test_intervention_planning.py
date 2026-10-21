
"""Unit tests for Component 4 intervention planning."""

import pandas as pd
import pytest

from backend.app.services.learning_progress.intervention_planning import (
    generate_intervention_plans,
)


def make_risk_data(risk="Low"):
    """Create one fictional participant with a specified risk category."""

    # Use the same required columns produced by stagnation detection.
    return pd.DataFrame(
        [
            {
                "UnlinkedParticipantNumber": 101,
                "latest_reading_score": 50,
                "number_of_observed_assessments": 3,
                "average_recent_change": 1.0,
                "stagnation_risk": risk,
            }
        ]
    )


@pytest.mark.parametrize(
    "risk",
    ["High", "Medium", "Low", "Insufficient Data"],
)
def test_plan_generated_for_each_risk_category(risk):
    """Every supported risk category should produce a support plan."""

    # Supply prepared risk data so the real dataset is not loaded.
    result = generate_intervention_plans(risk_data=make_risk_data(risk))

    # Verify that a plan is returned for the correct participant and category.
    assert len(result) == 1
    assert result.iloc[0]["UnlinkedParticipantNumber"] == 101
    assert result.iloc[0]["stagnation_risk"] == risk
    assert isinstance(result.iloc[0]["recommended_activities"], str)
    assert result.iloc[0]["recommended_activities"].strip() != ""
    assert isinstance(result.iloc[0]["frequency"], str)
    assert isinstance(result.iloc[0]["session_duration"], str)


def test_high_risk_plan_recommends_educator_review():
    """High-risk plans should include educator review."""

    result = generate_intervention_plans(risk_data=make_risk_data("High"))
    activities = result.iloc[0]["recommended_activities"]

    # Check the key action defined in the current implementation.
    assert "educator" in activities.lower()
    assert "guided practice" in activities.lower()


def test_insufficient_data_plan_requests_more_assessments():
    """Insufficient data should trigger more data collection, not a diagnosis."""

    result = generate_intervention_plans(
        risk_data=make_risk_data("Insufficient Data")
    )
    activities = result.iloc[0]["recommended_activities"]

    # Check that the plan asks for more assessment data.
    assert "additional reading-assessment data" in activities
    assert "enough observations" in activities


def test_invalid_input_type_raises_type_error():
    """Non-DataFrame input should be rejected."""

    # The function requires a pandas DataFrame.
    with pytest.raises(TypeError, match="risk_data must be a pandas DataFrame"):
        generate_intervention_plans(risk_data=[{"stagnation_risk": "High"}])


def test_missing_required_column_raises_value_error():
    """DataFrames missing required fields should be rejected."""

    # This input intentionally omits the required participant and score fields.
    incomplete_data = pd.DataFrame(
        [{"stagnation_risk": "High"}]
    )

    with pytest.raises(
        ValueError,
        match="Stagnation results are missing columns",
    ):
        generate_intervention_plans(risk_data=incomplete_data)


def test_empty_input_returns_empty_plan_dataframe():
    """An empty DataFrame with valid columns should return no plans."""

    # Preserve the required schema while supplying zero participants.
    empty_data = pd.DataFrame(
        columns=[
            "UnlinkedParticipantNumber",
            "latest_reading_score",
            "number_of_observed_assessments",
            "average_recent_change",
            "stagnation_risk",
        ]
    )

    result = generate_intervention_plans(risk_data=empty_data)

    # Empty input should be handled without raising an exception.
    assert result.empty
    assert "recommended_activities" in result.columns


def forecast(status, change, participant=101):
    return pd.DataFrame([{'UnlinkedParticipantNumber': participant,
                          'forecast_status': status, 'forecasted_score_change': change}])


def test_recent_decline_prompts_review_despite_positive_forecast():
    risk = make_risk_data('Low').assign(latest_score_change=-5)
    row = generate_intervention_plans(risk, forecast('observed_training_time_point', 10)).iloc[0]
    assert row.support_priority == 'Educator review'
    assert 'latest observed decline' in row.recommended_activities
    assert 'Latest observed change: -5.00' in row.recommendation_basis
    assert row.review_required


@pytest.mark.parametrize('status', ['unseen_training_time_point', 'training_time_points_unknown', 'no_forecast'])
def test_unsupported_forecast_cannot_escalate_plan(status):
    row = generate_intervention_plans(make_risk_data(), forecast(status, -30)).iloc[0]
    assert row.support_priority == 'Continue and monitor'
    assert row.forecast_use.startswith('Not used')
    assert 'predicted decline' not in row.recommended_activities


def test_supported_negative_forecast_prompts_review_not_diagnosis():
    row = generate_intervention_plans(make_risk_data(), forecast('observed_training_time_point', -5)).iloc[0]
    assert row.support_priority == 'Educator review'
    assert 'before changing support' in row.recommended_activities
    assert 'not established' in row.recommendation_basis


def test_forecast_is_matched_by_participant_id():
    row = generate_intervention_plans(make_risk_data(), forecast('observed_training_time_point', -5, 999)).iloc[0]
    assert row.support_priority == 'Continue and monitor'


def test_insufficient_history_still_requests_evidence_with_forecast():
    row = generate_intervention_plans(make_risk_data('Insufficient Data'), forecast('observed_training_time_point', -5)).iloc[0]
    assert row.support_priority == 'Collect evidence'
    assert 'additional reading-assessment data' in row.recommended_activities


@pytest.mark.parametrize('risk', ['Unknown', None])
def test_unknown_categories_are_not_silently_treated_as_insufficient(risk):
    with pytest.raises(ValueError, match='Unknown stagnation_risk'):
        generate_intervention_plans(make_risk_data(risk))


def test_duplicate_participants_and_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match='one non-missing ID'):
        generate_intervention_plans(pd.concat([make_risk_data(), make_risk_data()]))
    with pytest.raises(ValueError, match='valid BAS'):
        generate_intervention_plans(make_risk_data().assign(latest_reading_score=91))
    with pytest.raises(ValueError, match='Invalid score changes'):
        generate_intervention_plans(make_risk_data().assign(latest_score_change=float('inf')))


def test_missing_forecast_status_is_not_used_for_planning():
    row = generate_intervention_plans(make_risk_data(), forecast(pd.NA, -5)).iloc[0]
    assert row.support_priority == 'Continue and monitor'
    assert row.forecast_use.startswith('Not used')
