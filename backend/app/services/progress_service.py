
from datetime import datetime, timezone

from app.db.models import ActivityProgress
from app.db.session import SessionLocal


def _record_to_dict(record: ActivityProgress) -> dict:
    """Convert a database record into the existing API format."""
    recorded_at = record.recorded_at

    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)

    return {
        "learner_id": record.learner_id,
        "item_id": record.item_id,
        "score": float(record.score),
        "completed": record.completed,
        "recorded_at": recorded_at.isoformat(),
    }


def record_activity(
    learner_id: str,
    item_id: str,
    score: float,
    completed: bool = True,
) -> dict:
    """Save an activity result to PostgreSQL and return learner progress."""

    learner_id = learner_id.strip()
    item_id = item_id.strip()

    if not learner_id or not item_id:
        raise ValueError("Learner ID and item ID are required.")

    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100.")

    with SessionLocal() as db:
        try:
            record = ActivityProgress(
                learner_id=learner_id,
                item_id=item_id,
                score=float(score),
                completed=completed,
                recorded_at=datetime.now(timezone.utc),
            )

            db.add(record)
            db.commit()
            db.refresh(record)

            recorded_activity = _record_to_dict(record)

            learner_records = (
                db.query(ActivityProgress)
                .filter(ActivityProgress.learner_id == learner_id)
                .order_by(ActivityProgress.id.asc())
                .all()
            )

            scores = [float(item.score) for item in learner_records]

            return {
                "learner_id": learner_id,
                "recorded_activity": recorded_activity,
                "total_attempts": len(learner_records),
                "completed_activities": sum(
                    item.completed for item in learner_records
                ),
                "average_score": round(sum(scores) / len(scores), 2),
                "latest_score": scores[-1],
                "progress_status": "progress_recorded",
            }

        except Exception:
            db.rollback()
            raise


def get_learner_progress(learner_id: str) -> dict:
    """Retrieve saved progress for one learner from PostgreSQL."""

    learner_id = learner_id.strip()

    if not learner_id:
        raise ValueError("Learner ID is required.")

    with SessionLocal() as db:
        records = (
            db.query(ActivityProgress)
            .filter(ActivityProgress.learner_id == learner_id)
            .order_by(ActivityProgress.id.asc())
            .all()
        )

        if not records:
            return {
                "learner_id": learner_id,
                "total_attempts": 0,
                "completed_activities": 0,
                "average_score": None,
                "latest_score": None,
                "records": [],
            }

        scores = [float(record.score) for record in records]

        return {
            "learner_id": learner_id,
            "total_attempts": len(records),
            "completed_activities": sum(
                record.completed for record in records
            ),
            "average_score": round(sum(scores) / len(scores), 2),
            "latest_score": scores[-1],
            "records": [
                _record_to_dict(record) for record in records
            ],
        }
