from pathlib import Path

import joblib
import pandas as pd

from app.services.progress_service import get_learner_progress


# --------------------------------------------------
# 1. Project paths and trained model
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    APP_DIR
    / "models"
    / "trained"
    / "c3_activity_model.joblib"
)

CATALOGUE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "c3"
    / "learning_content_catalogue.csv"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained C3 model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Model training features
# Keep this order consistent with model training.
# --------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "grade",
    "preferred_topic",
    "preferred_activity",
    "preferred_duration_min",
    "reading_score",
    "reading_speed_wpm",
    "pronunciation_accuracy",
    "fluency_score",
    "difficulty_level",
    "difficulty_score",
    "weak_skill",
    "previous_score",
    "activity_completion_rate",
    "previous_activity",
]


# Model class labels must match the training dataset.
def normalize(value):
    """Normalize text for consistent comparisons."""
    if pd.isna(value):
        return ""

    return " ".join(
        str(value).strip().casefold().split()
    )


ACTIVITY_TYPE_MAP = {
    "word games": "word games",
    "reading stories": "reading stories",
    "comprehension": "comprehension",
    "pronunciation": "pronunciation",
}


DIFFICULTY_LEVELS = [
    "easy",
    "medium",
    "hard",
]


# --------------------------------------------------
# 3. Validate and load the learning-content catalogue
# --------------------------------------------------

def load_catalogue():
    """Load and validate available learning activities."""

    if not CATALOGUE_PATH.exists():
        raise ValueError(
            "Learning-content catalogue not found: "
            f"{CATALOGUE_PATH}"
        )

    catalogue = pd.read_csv(CATALOGUE_PATH)

    required_columns = [
        "item_id",
        "title",
        "activity_type",
        "target_skill",
        "difficulty_level",
        "duration_minutes",
        "topic",
        "content",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in catalogue.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Catalogue is missing columns: {missing_columns}"
        )

    if catalogue.empty:
        raise ValueError(
            "The learning-content catalogue is empty."
        )

    catalogue["item_id"] = (
        catalogue["item_id"].astype(str).str.strip()
    )

    if catalogue["item_id"].eq("").any():
        raise ValueError(
            "The catalogue contains empty item IDs."
        )

    if catalogue["item_id"].duplicated().any():
        raise ValueError(
            "The catalogue contains duplicate item IDs."
        )

    return catalogue


# --------------------------------------------------
# 4. Performance-based difficulty adaptation
# --------------------------------------------------

def get_target_difficulty(previous_score, current_difficulty):
    """
    Select a target difficulty based on recent performance.

    Score below 60:
        Move one level easier where possible.

    Score from 60 to below 85:
        Keep the current difficulty.

    Score 85 or above:
        Move one level harder where possible.

    These are prototype thresholds and should be evaluated
    during testing. They are not clinical cut-offs.
    """

    current = normalize(current_difficulty)

    if current not in DIFFICULTY_LEVELS:
        current = "easy"

    current_index = DIFFICULTY_LEVELS.index(current)

    try:
        score = float(previous_score)
    except (TypeError, ValueError):
        score = 0.0

    if pd.isna(score):
        score = 0.0

    score = max(0.0, min(100.0, score))

    if score < 60:
        target_index = max(0, current_index - 1)
    elif score >= 85:
        target_index = min(
            len(DIFFICULTY_LEVELS) - 1,
            current_index + 1,
        )
    else:
        target_index = current_index

    return DIFFICULTY_LEVELS[target_index]


# --------------------------------------------------
# 5. Update features from saved learner progress
# --------------------------------------------------

def update_features_from_progress(
    learner_features: dict,
    completed_ids: set[str],
):
    """
    Use saved progress to update the latest score and
    activity completion rate.

    Returns progress metadata for the API response.
    """

    learner_id = learner_features.get("learner_id")

    if not isinstance(learner_id, str) or not learner_id.strip():
        return {
            "progress_found": False,
            "progress_summary": None,
        }

    learner_id = learner_id.strip()
    progress = get_learner_progress(learner_id)
    records = progress.get("records", [])

    if not isinstance(records, list):
        records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        item_id = record.get("item_id")

        if (
            record.get("completed") is True
            and isinstance(item_id, str)
            and item_id.strip()
        ):
            completed_ids.add(item_id.strip())

    latest_score = progress.get("latest_score")

    if latest_score is not None:
        try:
            latest_score = float(latest_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Saved latest score must be numeric."
            ) from exc

        if pd.isna(latest_score) or not 0 <= latest_score <= 100:
            raise ValueError(
                "Saved latest score must be between 0 and 100."
            )

        # Saved progress is the source of truth when available.
        learner_features["previous_score"] = latest_score

    total_attempts = progress.get("total_attempts", 0)
    completed_count = progress.get("completed_activities", 0)

    try:
        total_attempts = int(total_attempts)
        completed_count = int(completed_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Saved activity counts must be integers."
        ) from exc

    if total_attempts < 0 or completed_count < 0:
        raise ValueError(
            "Saved activity counts cannot be negative."
        )

    if completed_count > total_attempts:
        raise ValueError(
            "Completed activities cannot exceed total attempts."
        )

    if total_attempts > 0:
        completion_rate = completed_count / total_attempts

        try:
            incoming_rate = float(
                learner_features["activity_completion_rate"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "activity_completion_rate must be numeric."
            ) from exc

        if pd.isna(incoming_rate) or not 0 <= incoming_rate <= 100:
            raise ValueError(
                "activity_completion_rate must be between 0 and 100."
            )

        # Preserve the input scale used by the trained model.
        if incoming_rate > 1:
            learner_features["activity_completion_rate"] = (
                completion_rate * 100
            )
        else:
            learner_features["activity_completion_rate"] = (
                completion_rate
            )

    return {
        "progress_found": total_attempts > 0,
        "progress_summary": {
            "learner_id": learner_id,
            "total_attempts": total_attempts,
            "completed_activities": completed_count,
            "average_score": progress.get("average_score"),
            "latest_score": latest_score,
        },
    }


# --------------------------------------------------
# 6. Generate ML-based recommendations
# --------------------------------------------------

def predict_recommended_activity(
    learner_features: dict,
    top_k: int = 3,
    completed_item_ids: list[str] | None = None,
) -> dict:
    """
    Generate personalized recommendations by combining:

    1. Trained model activity probabilities.
    2. Weak-skill matching.
    3. Preferred-topic matching.
    4. Performance-based difficulty adaptation.
    5. Preferred-duration matching.

    Completed activities are excluded.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    if not isinstance(learner_features, dict):
        raise ValueError(
            "learner_features must be a JSON object."
        )

    learner_features = learner_features.copy()

    if completed_item_ids is None:
        completed_item_ids = []

    if not isinstance(completed_item_ids, list):
        raise ValueError(
            "completed_item_ids must be a list."
        )

    completed_ids = set()

    for item_id in completed_item_ids:
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                "Each completed item ID must be a non-empty string."
            )

        completed_ids.add(item_id.strip())

    # Preserve the learner ID for progress retrieval.
    learner_id = learner_features.get("learner_id")

    if not isinstance(learner_id, str) or not learner_id.strip():
        raise ValueError(
            "learner_id is required to retrieve learner progress."
        )

    learner_id = learner_id.strip()
    learner_features["learner_id"] = learner_id

    # Validate required model inputs.
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in learner_features
        or learner_features[column] is None
    ]

    if missing:
        raise ValueError(
            f"Missing required learner features: {missing}"
        )

    # Merge saved progress into the request.
    progress_info = update_features_from_progress(
        learner_features=learner_features,
        completed_ids=completed_ids,
    )

    # Validate numeric model inputs.
    numeric_columns = [
        "age",
        "grade",
        "preferred_duration_min",
        "reading_score",
        "reading_speed_wpm",
        "pronunciation_accuracy",
        "fluency_score",
        "difficulty_score",
        "previous_score",
        "activity_completion_rate",
    ]

    for column in numeric_columns:
        try:
            value = float(learner_features[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column} must be numeric."
            ) from exc

        if pd.isna(value):
            raise ValueError(
                f"{column} cannot be missing."
            )

        learner_features[column] = value

    for column in [
        "reading_score",
        "pronunciation_accuracy",
        "fluency_score",
        "previous_score",
    ]:
        if not 0 <= learner_features[column] <= 100:
            raise ValueError(
                f"{column} must be between 0 and 100."
            )

    if learner_features["age"] <= 0:
        raise ValueError("age must be positive.")

    if learner_features["grade"] <= 0:
        raise ValueError("grade must be positive.")

    if learner_features["preferred_duration_min"] <= 0:
        raise ValueError(
            "preferred_duration_min must be positive."
        )

    if learner_features["reading_speed_wpm"] < 0:
        raise ValueError(
            "reading_speed_wpm cannot be negative."
        )

    if not 0 <= learner_features["difficulty_score"] <= 1:
        raise ValueError(
            "difficulty_score must be between 0 and 1."
        )

    if not 0 <= learner_features["activity_completion_rate"] <= 100:
        raise ValueError(
            "activity_completion_rate must be between 0 and 100."
        )

    # Build model input in the exact training column order.
    input_df = pd.DataFrame(
        [[learner_features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The trained model does not support probability estimates."
        )

    try:
        probabilities = model.predict_proba(input_df)[0]
    except Exception as exc:
        raise ValueError(
            "The trained model could not process the learner features. "
            "Check that the feature names, types, and categories match "
            "the training pipeline."
        ) from exc

    classifier = (
        model.named_steps["classifier"]
        if hasattr(model, "named_steps")
        and "classifier" in model.named_steps
        else model
    )

    classes = classifier.classes_

    activity_probabilities = {
        normalize(activity): float(probability)
        for activity, probability in zip(classes, probabilities)
    }

    # Load available learning content.
    catalogue = load_catalogue()

    # Exclude activities completed by this learner.
    catalogue = catalogue[
        ~catalogue["item_id"].isin(completed_ids)
    ].copy()

    if catalogue.empty:
        raise ValueError(
            "No new learning-content items are available. "
            "All catalogue items may have been completed."
        )

    # Normalize catalogue values for comparisons.
    catalogue["_activity_key"] = (
        catalogue["activity_type"].map(normalize)
    )
    catalogue["_skill_key"] = (
        catalogue["target_skill"].map(normalize)
    )
    catalogue["_topic_key"] = (
        catalogue["topic"].map(normalize)
    )
    catalogue["_difficulty_key"] = (
        catalogue["difficulty_level"].map(normalize)
    )

    weak_skill = normalize(learner_features["weak_skill"])
    preferred_topic = normalize(learner_features["preferred_topic"])
    current_difficulty = normalize(
        learner_features["difficulty_level"]
    )
    preferred_activity = normalize(
        learner_features["preferred_activity"]
    )

    preferred_duration = learner_features["preferred_duration_min"]
    previous_score = learner_features["previous_score"]

    # Adapt target difficulty according to learner performance.
    target_difficulty = get_target_difficulty(
        previous_score=previous_score,
        current_difficulty=current_difficulty,
    )

    ranked_items = []

    # --------------------------------------------------
    # 7. Calculate a combined score for each item
    # --------------------------------------------------

    for _, item in catalogue.iterrows():
        activity_key = item["_activity_key"]

        mapped_activity = ACTIVITY_TYPE_MAP.get(activity_key)

        if mapped_activity is None:
            continue

        model_probability = activity_probabilities.get(
            mapped_activity,
            0.0,
        )

        skill_match = item["_skill_key"] == weak_skill
        topic_match = item["_topic_key"] == preferred_topic

        difficulty_match = (
            item["_difficulty_key"] == target_difficulty
        )

        preferred_activity_match = (
            activity_key == preferred_activity
        )

        try:
            duration = float(item["duration_minutes"])
        except (TypeError, ValueError):
            continue

        if pd.isna(duration) or duration <= 0:
            continue

        duration_match = (
            abs(duration - preferred_duration) <= 5
        )

        # Weights are prototype ranking parameters.
        # Evaluate and tune them with validation experiments.
        final_score = (
            0.50 * model_probability
            + 0.20 * float(skill_match)
            + 0.10 * float(topic_match)
            + 0.10 * float(difficulty_match)
            + 0.05 * float(duration_match)
            + 0.05 * float(preferred_activity_match)
        )

        reasons = []

        if skill_match:
            reasons.append(
                f"targets the weak skill {item['target_skill']}"
            )

        if topic_match:
            reasons.append(
                f"matches the preferred topic {item['topic']}"
            )

        if difficulty_match:
            reasons.append(
                f"matches the performance-adjusted difficulty "
                f"{item['difficulty_level']}"
            )

        if duration_match:
            reasons.append(
                f"fits the preferred duration of "
                f"{preferred_duration:g} minutes"
            )

        if preferred_activity_match:
            reasons.append(
                f"matches the preferred activity type "
                f"{item['activity_type']}"
            )

        if not reasons:
            reasons.append(
                "ranked partly using the trained model prediction"
            )

        ranked_items.append({
            "item_id": str(item["item_id"]),
            "title": str(item["title"]),
            "activity_type": str(item["activity_type"]),
            "target_skill": str(item["target_skill"]),
            "difficulty_level": str(item["difficulty_level"]),
            "duration_minutes": int(duration),
            "topic": str(item["topic"]),
            "content": str(item["content"]),
            "model_probability": round(model_probability, 4),
            "score": round(final_score, 4),
            "reason": (
                "This item was recommended because it "
                + "; ".join(reasons)
                + "."
            ),
        })

    if not ranked_items:
        raise ValueError(
            "No usable catalogue items matched the model's activity classes."
        )

    # Sort by combined score, then item ID for stable ordering.
    ranked_items.sort(
        key=lambda item: (
            -item["score"],
            item["item_id"],
        )
    )

    recommendations = ranked_items[:top_k]

    for rank, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        recommendation["rank"] = rank

    # --------------------------------------------------
    # 8. Return recommendations and progress information
    # --------------------------------------------------

    return {
        "learner_id": learner_id,
        "recommendations": recommendations,
        "model": "Logistic Regression",
        "status": "prediction_generated",
        "progress": progress_info["progress_summary"],
        "progress_used": progress_info["progress_found"],
        "performance_adaptation": {
            "previous_score_used": previous_score,
            "current_difficulty": current_difficulty,
            "target_difficulty": target_difficulty,
        },
        "ranking_note": (
            "The ranking combines trained-model activity probabilities "
            "with weak-skill, topic, performance-adjusted difficulty, "
            "duration, and activity-preference matches. Scores are "
            "ranking values, not verified measures of educational benefit."
        ),
        "completed_items_excluded": sorted(completed_ids),
        "validation_note": (
            "Prototype trained on synthetic data; not yet validated "
            "on the secondary dataset."
        ),
    }