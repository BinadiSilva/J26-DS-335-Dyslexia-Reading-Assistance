
from app.services.ml_recommendation_service import (
    get_target_difficulty,
)


def test_low_score_reduces_difficulty():
    result = get_target_difficulty(45, "Medium")
    assert result == "easy"


def test_middle_score_keeps_difficulty():
    result = get_target_difficulty(70, "Medium")
    assert result == "medium"


def test_high_score_increases_difficulty():
    result = get_target_difficulty(90, "Medium")
    assert result == "hard"


def test_easy_difficulty_does_not_go_lower():
    result = get_target_difficulty(40, "Easy")
    assert result == "easy"


def test_hard_difficulty_does_not_go_higher():
    result = get_target_difficulty(95, "Hard")
    assert result == "hard"
