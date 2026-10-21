"""Reproducible fictional longitudinal sessions, separate from the LIBS data.

Latent plateau/decline labels describe the generator, not educator diagnoses.
Synthetic treatment responses are assumptions, never evidence of efficacy.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd

from .schema import validate_session


def generate_cohort(participants=180, weeks=20, seed=335, start="2025-01-06", prefix="SYN"):
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(participants):
        # Balanced latent scenarios with overlapping noisy observations.
        scenario = participant % 3
        growth = (rng.uniform(0.9, 1.8), rng.uniform(-0.15, 0.15), rng.uniform(-1.1, -0.4))[scenario]
        speed, accuracy, fluency = rng.uniform(30, 70), rng.uniform(55, 80), rng.uniform(35, 65)
        difficulty = int(rng.integers(1, 5))
        for week in range(weeks):
            practice = int(rng.choice([0, 20, 40, 60, 80]))
            speed = float(np.clip(speed + growth * 1.2 + rng.normal(0, 0.55), 10, 200))
            accuracy = float(np.clip(accuracy + growth * 0.65 + rng.normal(0, 0.28), 10, 98))
            fluency = float(np.clip(fluency + growth + rng.normal(0, 0.45), 5, 98))
            correct = int(round(2 * accuracy))
            session = validate_session({
                "assessed_on": (date.fromisoformat(start) + timedelta(days=7 * week)).isoformat(),
                "words_total": 200, "correct_words": correct,
                "insertions": int(rng.integers(0, 5)), "duration_seconds": round(correct * 60 / speed, 3),
                "fluency_score": round(fluency, 2), "difficulty_level": difficulty,
                "practice_minutes": practice, "notes": "Fictional assessment generated for software demonstration.",
            })
            rows.append({**session, "learner_id": f"{prefix}{participant + 1:04}", "source": "synthetic",
                         "reference_stagnation": scenario != 0})
    return pd.DataFrame(rows)


DEMO_NAMES = ("Emma", "Noah", "Olivia", "Liam", "Sophia", "Oliver")


def seed_demo(store):
    """Idempotent seed; never clears or overwrites existing learners."""
    start = (date.today() - timedelta(days=85)).isoformat()
    data = generate_cohort(6, 13, seed=7781, start=start, prefix="DEMO")
    existing = {child["id"] for child in store.learners()}
    for index, (learner_id, group) in enumerate(data.groupby("learner_id")):
        if learner_id in existing:
            continue
        store.create_learner({"alias": DEMO_NAMES[index], "age": 9, "grade": 4,
                              "source": "synthetic", "target_speed": 80, "target_accuracy": 90,
                              "target_fluency": 75}, learner_id)
        for position, row in enumerate(group.to_dict("records")):
            session_id = store.add_session(learner_id, row)
            if position in {3, 9}:
                store.start_intervention(learner_id, {
                    "activity": "Supported repeated reading of an educator-selected passage.",
                    "target_skill": "fluency", "minutes": 10, "frequency": 3,
                    "review_days": 14, "started_on": row["assessed_on"],
                    "rationale": ["Fictional saved plan for demonstrating outcome monitoring."],
                })
            if position == 7:
                plan = store.interventions(learner_id)[-1]
                store.complete_intervention(learner_id, plan["id"], session_id, 80, "Fictional follow-up; no real efficacy claim.")
        store.add_note(learner_id, "Demo educator", "All records for this learner are synthetic. Goals and schedules are demonstration settings.")
