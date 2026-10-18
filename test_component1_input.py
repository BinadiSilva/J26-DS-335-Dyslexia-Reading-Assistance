"""Tests for the Component 1 input contract (validator + idempotent log). No ML needed."""
import copy
import json
import tempfile
from pathlib import Path

from backend.app.services.reading_difficulty.component1_input import (
    upsert_session_log, validate_component1_payload,
)

GOOD = {
    "child_id": "demo_child_01", "session_id": "S-001", "session_date": "2026-10-05",
    "age": 8, "grade": 3, "gender": "Female", "passage_id": "P12", "passage_level": "Grade3",
    "reading_duration_sec": 64.0, "words_total": 90, "words_correct": 78, "words_incorrect": 12,
    "reading_speed_wpm": 84.4, "pronunciation_accuracy": 81.0, "fluency_score": 62.0,
    "word_error_rate": 0.13, "hesitation_count": 4, "repetition_count": 2, "omission_count": 3,
    "substitution_count": 5, "insertion_count": 1, "transcribed_text": "the cat sat",
    "pitch_mean": 210.0, "energy_mean": 0.42, "avg_pause_duration": 0.6,
    "pitch_samples": [200, 210, 220, 190], "energy_samples": [0.4, 0.5, 0.3],
    "pause_durations": [0.4, 0.8, 0.6],
}


def run():
    ok = validate_component1_payload(GOOD)
    assert ok.ok, ok.errors
    assert ok.record["pitch_samples"] == "200.0;210.0;220.0;190.0"
    assert not ok.warnings

    for field in ["child_id", "grade", "words_total", "session_date"]:
        bad = copy.deepcopy(GOOD); bad.pop(field)
        r = validate_component1_payload(bad)
        assert not r.ok and any(field in e for e in r.errors), field

    bad = copy.deepcopy(GOOD); bad["pronunciation_accuracy"] = 140
    assert not validate_component1_payload(bad).ok
    bad = copy.deepcopy(GOOD); bad["session_date"] = "2999-01-01"
    assert not validate_component1_payload(bad).ok
    bad = copy.deepcopy(GOOD); bad["session_date"] = "05/10/2026"
    assert not validate_component1_payload(bad).ok
    bad = copy.deepcopy(GOOD); bad["words_correct"] = 10
    assert not validate_component1_payload(bad).ok
    bad = copy.deepcopy(GOOD); bad["hesitation_count"] = "many"
    assert not validate_component1_payload(bad).ok
    bad = copy.deepcopy(GOOD); bad["age"] = 8.5
    assert not validate_component1_payload(bad).ok

    # missing sample lists are rejected
    for k in ("pitch_samples", "energy_samples", "pause_durations"):
        bad = copy.deepcopy(GOOD); bad.pop(k)
        r = validate_component1_payload(bad)
        assert not r.ok and any(f"missing required field: {k}" in e for e in r.errors)
    bad = copy.deepcopy(GOOD); bad["word_error_rate"] = 3.0
    assert not validate_component1_payload(bad).ok

    # outputs / derived fields in the input are ignored with a warning
    x = copy.deepcopy(GOOD); x["difficulty_level"] = "High"; x["normalized_speed"] = 9
    r = validate_component1_payload(x)
    assert r.ok and "difficulty_level" not in r.record and "normalized_speed" not in r.record
    assert len(r.warnings) == 2

    # upsert is idempotent
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "log.jsonl"
        for score in (0.3, 0.4):
            upsert_session_log({"child_id": "c1", "session_id": "s1", "difficulty_score": score}, log)
        upsert_session_log({"child_id": "c1", "session_id": "s2", "difficulty_score": 0.2}, log)
        rows = [json.loads(l) for l in log.read_text().splitlines()]
        assert len(rows) == 2 and rows[0]["difficulty_score"] == 0.4
    print("component1 input tests passed")


if __name__ == "__main__":
    run()
