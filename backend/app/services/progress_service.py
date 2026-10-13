
from datetime import datetime, timezone
import json
from pathlib import Path

# Store progress records locally during prototype development.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROGRESS_DIR = PROJECT_ROOT / "datasets" / "c3" / "progress"
PROGRESS_FILE = PROGRESS_DIR / "learner_progress.json"


def _load_records() -> list[dict]:
    """Load previously recorded learner progress."""
    if not PROGRESS_FILE.exists():
        return []

    with PROGRESS_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Progress file must contain a JSON list.")

    return records


def record_activity(
    learner_id: str,
    item_id: str,
    score: float,
    completed: bool = True,
) -> dict:
    """Save an activity result and return updated learner progress."""

    learner_id = learner_id.strip()
    item_id = item_id.strip()

    if not learner_id or not item_id:
        raise ValueError("Learner ID and item ID are required.")

    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100.")

    records = _load_records()

    record = {
        "learner_id": learner_id,
        "item_id": item_id,
        "score": float(score),
        "completed": completed,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    records.append(record)

    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    with PROGRESS_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    learner_records = [
        item for item in records
        if item["learner_id"] == learner_id
    ]

    scores = [item["score"] for item in learner_records]

    return {
        "learner_id": learner_id,
        "recorded_activity": record,
        "total_attempts": len(learner_records),
        "completed_activities": sum(
            item["completed"] for item in learner_records
        ),
        "average_score": round(sum(scores) / len(scores), 2),
        "latest_score": scores[-1],
        "progress_status": "progress_recorded",
    }


def get_learner_progress(learner_id: str) -> dict:
    """Retrieve saved progress for one learner."""

    learner_id = learner_id.strip()

    if not learner_id:
        raise ValueError("Learner ID is required.")

    records = [
        item for item in _load_records()
        if item["learner_id"] == learner_id
    ]

    if not records:
        return {
            "learner_id": learner_id,
            "total_attempts": 0,
            "completed_activities": 0,
            "average_score": None,
            "latest_score": None,
            "records": [],
        }

    scores = [item["score"] for item in records]

    return {
        "learner_id": learner_id,
        "total_attempts": len(records),
        "completed_activities": sum(
            item["completed"] for item in records
        ),
        "average_score": round(sum(scores) / len(scores), 2),
        "latest_score": scores[-1],
        "records": records,
    }
