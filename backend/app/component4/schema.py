"""Shared session contract for storage, training and inference.

Fluency is an educator-assigned 0–100 rubric score, not a BAS conversion.
Accuracy is percent correct; WER is a ratio and may exceed one (insertions).
"""

from datetime import date
import math

TARGETS = ("reading_speed_wpm", "reading_accuracy", "fluency_score", "word_error_rate")
FEATURES = TARGETS + ("difficulty_level", "practice_minutes")
# CSV import and the web form share raw observations; derived metrics are
# always recalculated so an imported accuracy or WER cannot contradict counts.
SESSION_INPUTS = (
    "assessed_on", "words_total", "correct_words", "insertions",
    "duration_seconds", "fluency_score", "difficulty_level", "practice_minutes",
)
WINDOW = 4
HORIZON_DAYS = 7


def number(value, name, low, high, integer=False):
    try:
        if isinstance(value, bool):
            raise ValueError()
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError()
        if integer and not value.is_integer():
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} between {low} and {high}.") from None
    return int(value) if integer else value


def text(value, name, maximum=200):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{name} is required (maximum {maximum} characters).")
    return value.strip()


def day(value):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("Dates must use YYYY-MM-DD.") from None
    return parsed.isoformat()


def validate_session(data):
    """Reject bad input rather than replacing missing measurements with zeros."""
    result = {"assessed_on": day(data.get("assessed_on"))}
    if date.fromisoformat(result["assessed_on"]) > date.today():
        raise ValueError("An observed assessment cannot be in the future.")
    for name, low, high, integer in (
        ("words_total", 1, 5000, True), ("correct_words", 0, 5000, True),
        ("insertions", 0, 5000, True), ("duration_seconds", 1, 7200, False),
        ("fluency_score", 0, 100, False), ("difficulty_level", 1, 5, True),
        ("practice_minutes", 0, 1680, False),
    ):
        result[name] = number(data.get(name), name, low, high, integer)
    if result["correct_words"] > result["words_total"]:
        raise ValueError("Correct words cannot exceed passage words.")
    result["reading_accuracy"] = 100 * result["correct_words"] / result["words_total"]
    result["reading_speed_wpm"] = 60 * result["correct_words"] / result["duration_seconds"]
    result["word_error_rate"] = (
        result["words_total"] - result["correct_words"] + result["insertions"]
    ) / result["words_total"]
    result["notes"] = str(data.get("notes", "")).strip()[:2000]
    reference = data.get("reference_stagnation")
    result["reference_stagnation"] = None
    if reference is not None and str(reference).strip().lower() not in {"", "nan"}:
        # This optional label must come from an independent reviewer. It is
        # retained for evaluation only and never added to model features.
        labels = {"true": True, "false": False, "1": True, "0": False,
                  "1.0": True, "0.0": False}
        if str(reference).strip().lower() not in labels:
            raise ValueError("Reference stagnation must be blank, true or false.")
        result["reference_stagnation"] = labels[str(reference).strip().lower()]
    return result
