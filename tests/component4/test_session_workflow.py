"""Persistent standalone workflows on fictional children; no original data writes."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from backend.app.component4.analytics import learner_report, monitor, intervention_outcomes, propose_plan
from backend.app.component4.schema import validate_session
from backend.app.component4.store import Store
from backend.app.component4.synthetic import generate_cohort, seed_demo


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "sessions.sqlite3")
    for name in ("one", "two"):
        db.create_learner({"alias": name, "age": 9, "grade": 4, "source": "synthetic",
                           "target_speed": 80, "target_accuracy": 90, "target_fluency": 75}, name)
    return db


@pytest.fixture
def rows():
    return generate_cohort(1, 8).to_dict("records")


def plan(start):
    return {"started_on": start, "target_skill": "fluency", "activity": "Fictional guided reading",
            "minutes": 10, "frequency": 3, "review_days": 14}


def test_raw_counts_compute_metrics_including_wer_above_one(rows):
    row = {**rows[0], "words_total": 100, "correct_words": 50, "insertions": 80, "duration_seconds": 60}
    valid = validate_session(row)
    assert valid["reading_accuracy"] == 50
    assert valid["reading_speed_wpm"] == 50
    assert valid["word_error_rate"] == 1.3


@pytest.mark.parametrize("field,value", [("words_total", 0), ("correct_words", 201), ("insertions", -1),
    ("duration_seconds", 0), ("fluency_score", float("nan")), ("practice_minutes", float("inf")),
    ("difficulty_level", 1.5), ("correct_words", True), ("assessed_on", "2025/01/01")])
def test_invalid_measurements_rejected(store, rows, field, value):
    with pytest.raises(ValueError):
        store.add_session("one", {**rows[0], field: value})
    assert store.sessions("one") == []


def test_future_assessments_rejected(store, rows):
    with pytest.raises(ValueError, match="future"):
        store.add_session("one", {**rows[0], "assessed_on": (date.today() + timedelta(days=1)).isoformat()})


def test_storage_survives_restart_orders_dates_and_rejects_duplicates(store, rows):
    store.add_session("one", rows[1])
    store.add_session("one", rows[0])
    restarted = Store(store.path)
    assert [s["assessed_on"] for s in restarted.sessions("one")] == [r["assessed_on"] for r in rows[:2]]
    with pytest.raises(ValueError, match="already"):
        restarted.add_session("one", rows[0])
    assert len(restarted.sessions("one")) == 2


def test_outcomes_are_linked_to_correct_child_and_after_start(store, rows):
    baseline_id = store.add_session("one", rows[0])
    intervention_id = store.start_intervention("one", plan(rows[0]["assessed_on"]))
    other_id = store.add_session("two", rows[2])
    for outcome_id in [baseline_id, other_id]:
        with pytest.raises(ValueError, match="after"):
            store.complete_intervention("one", intervention_id, outcome_id, 90, "")
    with pytest.raises(ValueError, match="active"):
        store.start_intervention("one", plan(rows[0]["assessed_on"]))
    followup_id = store.add_session("one", rows[3])
    store.complete_intervention("one", intervention_id, followup_id, 90, "Observed follow-up")
    results = intervention_outcomes(store.sessions("one"), store.interventions("one"))
    assert results[0]["outcome"] == "Observed improvement"
    assert results[0]["changes"]["fluency_score"] == round(rows[3]["fluency_score"] - rows[0]["fluency_score"], 3)
    with pytest.raises(ValueError, match="Active"):
        store.complete_intervention("one", intervention_id, followup_id, 90, "overwrite")


def test_different_passage_difficulty_blocks_effectiveness_interpretation(store, rows):
    store.add_session("one", rows[0])
    intervention_id = store.start_intervention("one", plan(rows[0]["assessed_on"]))
    followup = store.add_session("one", {**rows[3], "difficulty_level": 5})
    store.complete_intervention("one", intervention_id, followup, 80, "")
    result = learner_report(store, "one")["interventions"][0]
    assert not result["comparable"]
    assert "Different difficulty" in result["outcome"]


def test_recent_decline_cannot_be_hidden_by_earlier_gains(rows):
    history = deepcopy(rows[:4])
    for row, value in zip(history, [20, 40, 60, 50]):
        row["fluency_score"] = value
    result = monitor(history)
    assert result["level"] == "High" and result["recent_decline"]
    history[-1]["difficulty_level"] = 5
    assert monitor(history)["level"] == "Insufficient data"


def test_short_observation_period_and_ceiling_do_not_claim_stagnation(rows):
    history = deepcopy(rows[:3])
    for index, row in enumerate(history):
        row["assessed_on"] = f"2025-01-0{index + 1}"
    assert monitor(history)["level"] == "Insufficient data"
    history = deepcopy(rows[:4])
    for row in history:
        row["fluency_score"] = row["reading_accuracy"] = 99
    assert monitor(history)["level"] == "Review ceiling"


def test_prior_low_adherence_changes_next_proposal(store, rows):
    learner = store.learner("one")
    history = deepcopy(rows[:4])
    history[-1]["fluency_score"] = 5  # ensure fluency is the largest goal shortfall
    risk = monitor(history)
    outcomes = [{"status": "completed", "plan": {"target_skill": "fluency"}, "outcome": "Little observed change", "adherence": 20, "comparable": True}]
    result = propose_plan(learner, history, risk, {"status": "unavailable"}, outcomes)
    assert "barriers" in result["activity"]
    assert result["expected_improvement"] is None
    assert result["requires_educator_approval"]


def test_demo_seed_is_idempotent_and_has_recorded_followups(store):
    seed_demo(store)
    report = learner_report(store, "DEMO0001")
    assert len(report["sessions"]) == 13
    assert len(report["interventions"]) == 2
    assert report["interventions"][0]["status"] == "completed"
    assert report["active_intervention"] is not None
    seed_demo(store)
    assert len(store.sessions("DEMO0001")) == 13


def test_report_refreshes_on_saved_session_and_notes_persist(store, rows):
    store.add_session("one", rows[0])
    first = learner_report(store, "one")
    store.add_session("one", rows[1])
    store.add_note("one", "educator", "Review the next passage")
    second = learner_report(store, "one")
    assert len(first["sessions"]) == 1 and len(second["sessions"]) == 2
    assert second["notes"][0]["body"] == "Review the next passage"


def test_passwords_hashed_and_parent_scope_required(store):
    store.create_user("teacher", "testing-password", "teacher")
    with store.connection() as db:
        assert db.execute("SELECT password_hash FROM users").fetchone()[0] != "testing-password"
    assert store.authenticate("teacher", "wrong-password") is None
    assert store.authenticate("teacher", "testing-password")["role"] == "teacher"
    with pytest.raises(ValueError, match="assigned"):
        store.create_user("parent", "testing-password", "parent")


def test_goal_updates_do_not_rewrite_approved_plan(store, rows):
    store.add_session("one", rows[0])
    proposal = learner_report(store, "one")["proposed_plan"]
    store.start_intervention("one", {**proposal, "started_on": rows[0]["assessed_on"]})
    store.update_goals("one", {"target_speed": 100, "target_accuracy": 95, "target_fluency": 85})
    assert store.learner("one")["target_speed"] == 100
    assert store.interventions("one")[0]["plan"]["goal_snapshot"]["speed"] == 80


def test_independent_labels_export_and_do_not_override_detector(store, rows):
    for row in rows[:4]:
        store.add_session("one", {**row, "reference_stagnation": "true", "notes": "Private review"})
    report = learner_report(store, "one")
    assert report["risk"]["level"] == "Low"  # observations determine screening
    assert report["sessions"][-1]["reference_stagnation"] is True
    records = store.training_records("synthetic")
    assert records[0]["reference_stagnation"] is True
    assert "notes" not in records[0] and "alias" not in records[0]
    assert store.training_records("collected") == []
    with pytest.raises(ValueError, match="Reference"):
        validate_session({**rows[0], "reference_stagnation": "unknown"})


def test_password_reset_invalidates_the_old_password(store):
    store.create_user("teacher", "old-testing-password", "teacher")
    store.reset_password("teacher", "new-testing-password")
    assert store.authenticate("teacher", "old-testing-password") is None
    assert store.authenticate("teacher", "new-testing-password")


def test_supported_accuracy_forecast_can_prompt_review(store, rows):
    history = rows[:4]
    forecast_values = dict(history[-1])
    forecast_values["reading_accuracy"] -= 5
    result = propose_plan(store.learner("one"), history, monitor(history),
                          {"status": "available", "values": forecast_values}, [])
    assert result["priority"] == "Medium"
    assert any("lower accuracy" in reason for reason in result["rationale"])
