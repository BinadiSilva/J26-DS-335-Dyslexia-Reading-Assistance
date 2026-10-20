"""Generate and persist personalized seven-day learning plans."""

from datetime import datetime, timezone

from app.db.models import WeeklyPlan
from app.db.session import SessionLocal
from app.services.ml_recommendation_service import (
    get_target_difficulty,
    load_catalogue,
    predict_recommended_activity,
    update_features_from_progress,
)
from app.services.progress_service import get_learner_progress


PLAN_LENGTH = 7

ACTIVITY_INSTRUCTIONS = {
    "word games": (
        "Look at each word, say its sounds, and complete the word-matching activity."
    ),
    "reading stories": (
        "Read the story slowly. Sound out unfamiliar words and try reading them again."
    ),
    "comprehension": (
        "Read the text carefully and answer the questions using clues from the text."
    ),
    "pronunciation": (
        "Say each word aloud slowly, repeat it, and focus on the sounds."
    ),
}


def _normalise(value):
    """Convert values to comparable lowercase text."""
    return str(value or "").strip().lower()


def _number(value, default=0.0):
    """Safely convert a value to a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_activity(row, day_number, model_result=None, target_difficulty=None):
    """Convert a catalogue row into a child-friendly daily activity."""
    activity_type = str(row.get("activity_type", ""))
    title = str(row.get("title", ""))
    skill = str(row.get("target_skill", ""))
    content = str(row.get("content", ""))

    instructions = ACTIVITY_INSTRUCTIONS.get(
        _normalise(activity_type),
        "Complete the activity carefully and ask an adult for help if needed.",
    )

    objective = (
        f"Practise {skill.lower()} through the activity "
        f"'{title}'."
        if skill and title
        else f"Practise reading skills through '{title}'."
    )

    return {
        "day_number": day_number,
        "item_id": str(row.get("item_id", "")),
        "title": title,
        "activity_type": activity_type,
        "target_skill": skill,
        "difficulty_level": str(row.get("difficulty_level", "")),
        "duration_minutes": int(_number(row.get("duration_minutes"), 10)),
        "topic": str(row.get("topic", "")),
        "content": content,
        "instructions": instructions,
        "learning_objective": objective,
        "model_probability": _number(
            (model_result or {}).get("model_probability"), 0.0
        ),
        "recommendation_score": _number(
            (model_result or {}).get("score"), 0.0
        ),
        "reason": str(
            (model_result or {}).get(
                "reason",
                "Selected using the learner's preferences, skill needs, and difficulty.",
            )
        ),
    }


def _fallback_score(row, features, target_difficulty):
    """Rank catalogue activities when the ML engine returns too few candidates."""
    score = 0.0

    if _normalise(row.get("target_skill")) == _normalise(
        features.get("weak_skill")
    ):
        score += 0.35

    if _normalise(row.get("topic")) == _normalise(
        features.get("preferred_topic")
    ):
        score += 0.20

    if _normalise(row.get("activity_type")) == _normalise(
        features.get("preferred_activity")
    ):
        score += 0.15

    candidate_difficulty = _normalise(row.get("difficulty_level"))
    target_difficulty = _normalise(target_difficulty)

    if candidate_difficulty == target_difficulty:
        score += 0.20
    elif {candidate_difficulty, target_difficulty} in (
        {"easy", "medium"},
        {"medium", "hard"},
    ):
        score += 0.10

    preferred_duration = _number(features.get("preferred_duration_min"), 15)
    duration = _number(row.get("duration_minutes"), 15)
    duration_match = max(
        0.0,
        1.0 - abs(duration - preferred_duration) / max(preferred_duration, 1),
    )
    score += 0.10 * duration_match

    return score


def generate_weekly_plan(learner_id, week_number, learner_features):
    """
    Generate and save a weekly plan.

    An existing plan for the same learner and week is returned unchanged.
    Previous-week activities and completed activities are excluded.
    """
    learner_id = str(learner_id).strip()

    if not learner_id:
        raise ValueError("learner_id cannot be empty.")

    if isinstance(week_number, bool) or not isinstance(week_number, int):
        raise ValueError("week_number must be an integer.")

    if week_number < 1:
        raise ValueError("week_number must be at least 1.")

    if not isinstance(learner_features, dict):
        raise ValueError("learner_features must be a dictionary.")

    features = dict(learner_features)
    features["learner_id"] = learner_id
    
    db = SessionLocal()

    try:
        # Do not silently overwrite a plan already saved for this week.
        existing_plan = (
            db.query(WeeklyPlan)
            .filter(
                WeeklyPlan.learner_id == learner_id,
                WeeklyPlan.week_number == week_number,
            )
            .first()
        )

        if existing_plan:
            return {
                "learner_id": existing_plan.learner_id,
                "week_number": existing_plan.week_number,
                "activities": existing_plan.activities,
                "learner_features": existing_plan.learner_features,
                "generated_at": (
                    existing_plan.generated_at.isoformat()
                    if existing_plan.generated_at
                    else None
                ),
                "status": "existing",
                "message": "An existing plan was returned without changes.",
            }

        # Collect previously completed activity IDs from recorded progress.
        progress = get_learner_progress(learner_id)
        progress_records = progress.get("records", []) if isinstance(
            progress, dict
        ) else []

        excluded_ids = {
            str(record.get("item_id"))
            for record in progress_records
            if record.get("item_id") is not None
            and record.get("completed") is True
        }

        # Week 2+ should not simply repeat activities assigned in the previous week.
        previous_plan = None
        if week_number > 1:
            previous_plan = (
                db.query(WeeklyPlan)
                .filter(
                    WeeklyPlan.learner_id == learner_id,
                    WeeklyPlan.week_number == week_number - 1,
                )
                .first()
            )

        if previous_plan and isinstance(previous_plan.activities, list):
            excluded_ids.update(
                str(activity.get("item_id"))
                for activity in previous_plan.activities
                if activity.get("item_id") is not None
            )

        # Update the plan's saved/displayed features from recorded progress.
        # This also keeps the response's activity_completion_rate current.
        update_features_from_progress(
            learner_features=features,
            completed_ids=excluded_ids,
        )

        # Determine the preferred difficulty.
        current_difficulty = features.get("difficulty_level", "Easy")

        # A saved activity score takes priority over the supplied score.
        saved_score = (
            progress.get("latest_score")
            if isinstance(progress, dict)
            else None
        )
        supplied_score = features.get("previous_score")

        if saved_score is not None:
            previous_score = _number(saved_score)
        elif supplied_score is not None:
            previous_score = _number(supplied_score)
        else:
            previous_score = None

        # Do not adapt difficulty using a score that does not exist.
        if previous_score is None:
            target_difficulty = current_difficulty
        else:
            target_difficulty = get_target_difficulty(
                previous_score, current_difficulty
            )


        # Obtain ML-ranked activities. The catalogue fallback below fills any gaps.
        ml_by_id = {}
        ml_error = None

        try:
            # Use a separate copy for the model input.
            ml_features = dict(features)

            # Use a neutral model input if no previous score exists.
            # This is not an actual learner assessment score.
            if ml_features.get("previous_score") is None:
                ml_features["previous_score"] = 60

            ml_result = predict_recommended_activity(
                learner_features=ml_features,
                top_k=10,
                completed_item_ids=sorted(excluded_ids),
            )
            for recommendation in ml_result.get("recommendations", []):
                item_id = str(recommendation.get("item_id", ""))
                if item_id and item_id not in excluded_ids:
                    ml_by_id[item_id] = recommendation
        except Exception as exc:
            # The catalogue fallback still allows a plan to be generated.
            ml_error = str(exc)

        catalogue = load_catalogue()
        rows = catalogue.to_dict(orient="records")

        eligible = [
            row for row in rows
            if str(row.get("item_id", "")) not in excluded_ids
        ]

        if len(eligible) < PLAN_LENGTH:
            raise ValueError(
                f"Only {len(eligible)} distinct activities remain for learner "
                f"'{learner_id}'. Seven are required. Add more catalogue "
                "activities or review the previous assignments before generating "
                "another plan."
            )

        # Use ML-ranked items when available; rank remaining items as fallback.
        ranked = []
        for row in eligible:
            item_id = str(row.get("item_id", ""))
            ml_item = ml_by_id.get(item_id)

            if ml_item:
                ranking_score = _number(ml_item.get("score"))
                ranking_reason = str(
                    ml_item.get("reason", "Ranked by the ML recommendation engine.")
                )
            else:
                ranking_score = _fallback_score(
                    row, features, target_difficulty
                )
                ranking_reason = (
                    "Selected using skill needs, preferences, duration, "
                    "and target difficulty."
                )

            ranked.append(
                {
                    "row": row,
                    "ml_item": ml_item,
                    "ranking_score": ranking_score,
                    "ranking_reason": ranking_reason,
                }
            )
        # Use ML recommendations only when enough are available
        ml_candidate_count = sum(
            candidate["ml_item"] is not None for candidate in ranked
        )

        if ml_candidate_count >= PLAN_LENGTH:
            ranked = [
                candidate
                for candidate in ranked
                if candidate["ml_item"] is not None
            ]
        # Prefer varied activity types while preserving the relevance ranking.
        ranked.sort(
            key=lambda item: (
                item["ranking_score"],
                _normalise(item["row"].get("difficulty_level"))
                == _normalise(target_difficulty),
            ),
            reverse=True,
        )

        selected = []
        selected_ids = set()
        type_counts = {}

        # First pass: favour variety across the week.
        for candidate in ranked:
            row = candidate["row"]
            item_id = str(row.get("item_id", ""))
            activity_type = _normalise(row.get("activity_type"))

            if item_id in selected_ids:
                continue

            if type_counts.get(activity_type, 0) >= 2:
                continue

            selected.append(candidate)
            selected_ids.add(item_id)
            type_counts[activity_type] = type_counts.get(activity_type, 0) + 1

            if len(selected) == PLAN_LENGTH:
                break

        # Second pass: fill any remaining days with the best distinct activities.
        if len(selected) < PLAN_LENGTH:
            for candidate in ranked:
                item_id = str(candidate["row"].get("item_id", ""))

                if item_id in selected_ids:
                    continue

                selected.append(candidate)
                selected_ids.add(item_id)

                if len(selected) == PLAN_LENGTH:
                    break

        activities = []
        for day_number, candidate in enumerate(selected, start=1):
            activity = _make_activity(
                candidate["row"],
                day_number,
                candidate["ml_item"],
                target_difficulty,
            )
            activity["reason"] = candidate["ranking_reason"]
            activities.append(activity)

        saved_features = dict(features)
        saved_features["target_difficulty"] = target_difficulty
        saved_features["excluded_activity_ids"] = sorted(excluded_ids)

        plan = WeeklyPlan(
            learner_id=learner_id,
            week_number=week_number,
            activities=activities,
            learner_features=saved_features,
        )

        db.add(plan)
        db.commit()
        db.refresh(plan)

        return {
            "learner_id": plan.learner_id,
            "week_number": plan.week_number,
            "activities": plan.activities,
            "learner_features": plan.learner_features,
            "generated_at": (
                plan.generated_at.isoformat() if plan.generated_at else None
            ),
            "status": "created",
            "target_difficulty": target_difficulty,
            "ml_used": bool(ml_by_id),
            "ml_fallback_note": (
                f"ML recommendations could not be used: {ml_error}"
                if ml_error and not ml_by_id
                else None
            ),
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()