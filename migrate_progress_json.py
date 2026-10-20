
import json
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import ActivityProgress
from app.db.session import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parent
JSON_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "c3"
    / "progress"
    / "learner_progress.json"
)


def normalize_timestamp(value: datetime) -> datetime:
    """Normalize timestamps to UTC for duplicate checks."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def migrate_progress():
    if not JSON_FILE.exists():
        raise FileNotFoundError(
            f"Progress JSON file not found: {JSON_FILE}"
        )

    with JSON_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Progress JSON must contain a list.")

    inserted = 0
    skipped = 0

    with SessionLocal() as db:
        try:
            existing_records = db.query(ActivityProgress).all()

            existing_keys = {
                (
                    row.learner_id,
                    row.item_id,
                    float(row.score),
                    row.completed,
                    normalize_timestamp(row.recorded_at),
                )
                for row in existing_records
            }

            for item in records:
                timestamp = normalize_timestamp(
                    datetime.fromisoformat(
                        item["recorded_at"].replace("Z", "+00:00")
                    )
                )

                key = (
                    item["learner_id"].strip(),
                    item["item_id"].strip(),
                    float(item["score"]),
                    bool(item["completed"]),
                    timestamp,
                )

                if key in existing_keys:
                    skipped += 1
                    continue

                db.add(
                    ActivityProgress(
                        learner_id=key[0],
                        item_id=key[1],
                        score=key[2],
                        completed=key[3],
                        recorded_at=timestamp,
                    )
                )

                existing_keys.add(key)
                inserted += 1

            db.commit()

            print("Migration completed successfully.")
            print(f"Records in JSON: {len(records)}")
            print(f"New records inserted: {inserted}")
            print(f"Duplicate records skipped: {skipped}")

        except Exception:
            db.rollback()
            raise


if __name__ == "__main__":
    migrate_progress()
