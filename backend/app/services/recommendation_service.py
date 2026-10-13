
from typing import List

from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendedItem,
)


# Temporary catalogue for developing and testing C3.
# Later, replace this with materials from your dataset/database.
READING_CATALOGUE = [
    {
        "item_id": "R001",
        "title": "The Clever Little Fox",
        "item_type": "reading",
        "topic": "animals",
        "target_skill": "fluency",
        "difficulty_level": "easy",
        "duration_minutes": 10,
    },
    {
        "item_id": "R002",
        "title": "A Day at the Zoo",
        "item_type": "reading",
        "topic": "animals",
        "target_skill": "pronunciation",
        "difficulty_level": "medium",
        "duration_minutes": 15,
    },
    {
        "item_id": "A001",
        "title": "Read Aloud Practice",
        "item_type": "activity",
        "topic": "general",
        "target_skill": "fluency",
        "difficulty_level": "easy",
        "duration_minutes": 10,
    },
    {
        "item_id": "A002",
        "title": "Sound and Word Practice",
        "item_type": "activity",
        "topic": "general",
        "target_skill": "decoding",
        "difficulty_level": "medium",
        "duration_minutes": 15,
    },
    {
        "item_id": "R003",
        "title": "The Little School Adventure",
        "item_type": "reading",
        "topic": "school",
        "target_skill": "comprehension",
        "difficulty_level": "medium",
        "duration_minutes": 20,
    },
    {
        "item_id": "A003",
        "title": "Word Accuracy Challenge",
        "item_type": "activity",
        "topic": "general",
        "target_skill": "pronunciation",
        "difficulty_level": "easy",
        "duration_minutes": 10,
    },
]


def generate_recommendations(
    request: RecommendationRequest,
    top_k: int = 3,
) -> List[RecommendedItem]:

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    weak_skills = {
        skill.strip().lower()
        for skill in request.difficulty_assessment.weak_skills
    }

    preferred_topics = {
        topic.strip().lower()
        for topic in request.preferences.preferred_topics
    }

    preferred_activities = {
        activity.strip().lower()
        for activity in request.preferences.preferred_activity_types
    }

    difficulty = (
        request.difficulty_assessment.difficulty_level.strip().lower()
    )

    preferred_duration = (
        request.preferences.preferred_duration_minutes
    )

    completed_ids = set(
        request.learning_history.completed_activity_ids
    )

    scored_items = []

    for item in READING_CATALOGUE:

        # Avoid recommending previously completed items.
        if item["item_id"] in completed_ids:
            continue

        score = 0.0
        reasons = []

        # 1. Match the learner's weak reading skills.
        if item["target_skill"].lower() in weak_skills:
            score += 0.35
            reasons.append(
                f"targets the weak skill {item['target_skill']}"
            )

        # 2. Match the learner's preferred topics.
        if item["topic"].lower() in preferred_topics:
            score += 0.20
            reasons.append(
                f"matches the preferred topic {item['topic']}"
            )

        # 3. Match the preferred activity type.
        if item["item_type"].lower() in preferred_activities:
            score += 0.15
            reasons.append(
                f"matches the preferred activity type {item['item_type']}"
            )

        # 4. Match the difficulty identified by C2.
        if item["difficulty_level"].lower() == difficulty:
            score += 0.15
            reasons.append("matches the identified difficulty level")

        # 5. Prefer activities close to the requested duration.
        duration_difference = abs(
            item["duration_minutes"] - preferred_duration
        )

        if duration_difference <= 5:
            score += 0.15
            reasons.append("fits the preferred practice duration")

        # Explanations should reflect the actual matching factors.
        if not reasons:
            reasons.append(
                "included as an alternative for further practice"
            )

        scored_items.append(
            {
                "item": item,
                "score": round(score, 2),
                "reason": "Recommended because it " + "; ".join(reasons),
            }
        )

    # Highest-scoring items appear first.
    scored_items.sort(
        key=lambda result: (
            result["score"],
            result["item"]["item_id"],
        ),
        reverse=True,
    )

    recommendations = []

    for result in scored_items[:top_k]:
        item = result["item"]

        recommendations.append(
            RecommendedItem(
                **item,
                score=result["score"],
                reason=result["reason"],
            )
        )

    return recommendations
