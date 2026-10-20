
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
