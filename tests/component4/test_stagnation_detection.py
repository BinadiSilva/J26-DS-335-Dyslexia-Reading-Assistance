
"""Unit tests for Component 4 stagnation detection."""

import pandas as pd
import pytest

from backend.app.services.learning_progress import stagnation_detection


@pytest.fixture
def mock_reading_data(monkeypatch):
    """Provide a helper to replace the real dataset with test data."""

    def _mock_data(rows):
        # Use a controlled dataset so tests do not depend on LIBS files.
        df = pd.DataFrame(
            rows,
            columns=[
                "UnlinkedParticipantNumber",
                "time_point",
                "reading_score",
            ],
        )

        # Replace the data loader used by the real detection function.
        monkeypatch.setattr(
            stagnation_detection,
            "load_libs_reading_data",
            lambda: df.copy(),
        )

    return _mock_data


@pytest.mark.parametrize(
    ("scores", "expected_risk"),
    [
        ([50, 55, 60], "Low"),  # Consistent improvement.
        ([50, 51, 51.5], "Medium"),  # Small average improvement.
        ([50, 50, 50], "High"),  # No improvement.
        ([60, 58, 57], "High"),  # Declining scores.
    ],
)
def test_stagnation_categories_for_score_trends(
    mock_reading_data,
    scores,
    expected_risk,
):
    """Check that score trends receive the expected rule-based category."""

    # Create three consecutive observed assessments for one participant.
    mock_reading_data(
        [
            (1, index + 1, score)
            for index, score in enumerate(scores)
        ]
    )

    # Call the actual detection function.
    result = stagnation_detection.detect_stagnation()

    # Check the category assigned by the function.
    assert len(result) == 1
    assert result.iloc[0]["stagnation_risk"] == expected_risk


def test_insufficient_assessment_history(mock_reading_data):
    """A participant with only two assessments has one score change."""

    # The current rule requires at least two score changes.
    mock_reading_data(
        [
            (1, 1, 50),
            (1, 2, 55),
        ]
    )

    result = stagnation_detection.detect_stagnation()

    # One score change is insufficient for the current trend rule.
    assert result.iloc[0]["stagnation_risk"] == "Insufficient Data"
    assert pd.isna(result.iloc[0]["average_recent_change"])


def test_missing_intermediate_time_point(mock_reading_data):
    """Missing assessment points must not be treated as zero scores."""

    # T2 is missing; the observed change is calculated from T1 to T3.
    mock_reading_data(
        [
            (1, 1, 50),
            (1, 3, 55),
            (1, 4, 60),
        ]
    )

    result = stagnation_detection.detect_stagnation()

    # The function should use observed scores without inventing a T2 score.
    assert result.iloc[0]["number_of_observed_assessments"] == 3
    assert result.iloc[0]["latest_time_point"] == 4
    assert result.iloc[0]["stagnation_risk"] == "Low"


def test_recent_decline_survives_positive_long_term_average(mock_reading_data):
    mock_reading_data([(1, 1, 10), (1, 2, 50), (1, 3, 40)])
    row = stagnation_detection.detect_stagnation().iloc[0]
    assert row.stagnation_risk == 'Low'
    assert row.recent_decline
    assert row.latest_score_change == -10
    assert row.latest_change_direction == 'Decline'
    assert 'latest score declined' in row.explanation


def test_rate_uses_study_age_gaps_not_time_point_steps(mock_reading_data):
    # Ages 4, 6, 9 give gaps of two and three years, not 2 and 1.
    mock_reading_data([(1, 1, 10), (1, 3, 20), (1, 4, 35)])
    row = stagnation_detection.detect_stagnation().iloc[0]
    assert row.latest_study_year_gap == 3
    assert row.recent_change_per_study_year == 5
    assert row.has_missing_time_points


def test_two_observations_allow_latest_change_but_not_trend_category(mock_reading_data):
    mock_reading_data([(1, 1, 30), (1, 2, 20)])
    row = stagnation_detection.detect_stagnation().iloc[0]
    assert row.stagnation_risk == 'Insufficient Data'
    assert row.recent_decline
    assert row.latest_score_change == -10


def test_single_observation_and_empty_input_have_stable_schema(mock_reading_data):
    mock_reading_data([(1, 1, 30)])
    row = stagnation_detection.detect_stagnation().iloc[0]
    assert pd.isna(row.latest_score_change)
    assert pd.isna(row.recent_change_per_study_year)
    mock_reading_data([])
    result = stagnation_detection.detect_stagnation()
    assert result.empty
    assert 'latest_score_change' in result.columns


@pytest.mark.parametrize(('score', 'risk'), [(12, 'Medium'), (14, 'Low')])
def test_exact_average_threshold(mock_reading_data, score, risk):
    mock_reading_data([(1, 1, 10), (1, 2, 11), (1, 3, score)])
    assert stagnation_detection.detect_stagnation().iloc[0].stagnation_risk == risk
