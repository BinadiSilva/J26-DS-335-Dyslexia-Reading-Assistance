from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ml_recommendations_requires_learner_id():
    response = client.post(
        "/ml-recommendations",
        json={"reading_score": 60},
    )

    assert response.status_code == 400
    assert "learner_id" in response.json()["detail"]


def test_ml_recommendations_rejects_invalid_completed_ids():
    response = client.post(
        "/ml-recommendations",
        json={
            "learner_id": "TEST_API_001",
            "completed_item_ids": "not-a-list",
        },
    )

    assert response.status_code == 400
    assert "completed_item_ids" in response.json()["detail"]


def test_progress_record_rejects_score_above_100():
    response = client.post(
        "/progress/record",
        json={
            "learner_id": "TEST_API_001",
            "item_id": "TEST_ITEM_001",
            "score": 101,
            "completed": True,
        },
    )

    assert response.status_code == 422


def test_progress_record_rejects_negative_score():
    response = client.post(
        "/progress/record",
        json={
            "learner_id": "TEST_API_001",
            "item_id": "TEST_ITEM_001",
            "score": -1,
            "completed": True,
        },
    )

    assert response.status_code == 422


def test_unknown_learner_has_empty_progress():
    response = client.get("/progress/TEST_UNKNOWN_987654")

    assert response.status_code == 200

    data = response.json()
    assert data["learner_id"] == "TEST_UNKNOWN_987654"
    assert data["total_attempts"] == 0
    assert data["completed_activities"] == 0
    assert data["records"] == []


def test_ml_recommendations_returns_personalized_items():
    response = client.post(
        "/ml-recommendations",
        json={
            "learner_id": "TEST_RECOMMENDATION_001",
            "age": 8,
            "grade": 3,
            "preferred_topic": "Space",
            "preferred_activity": "Word Games",
            "preferred_duration_min": 15,
            "reading_score": 65,
            "reading_speed_wpm": 45,
            "pronunciation_accuracy": 78,
            "fluency_score": 60,
            "difficulty_level": "Medium",
            "difficulty_score": 0.6,
            "weak_skill": "Phonics",
            "previous_score": 70,
            "activity_completion_rate": 0.5,
            "previous_activity": "Word Games",
            "completed_item_ids": [],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["learner_id"] == "TEST_RECOMMENDATION_001"
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

    for item in data["recommendations"]:
        assert "item_id" in item
        assert "title" in item
        assert "score" in item
        assert "reason" in item


def test_ml_recommendations_excludes_completed_items():
    completed_ids = ["A003", "A009"]

    response = client.post(
        "/ml-recommendations",
        json={
            "learner_id": "TEST_RECOMMENDATION_002",
            "age": 8,
            "grade": 3,
            "preferred_topic": "Space",
            "preferred_activity": "Word Games",
            "preferred_duration_min": 15,
            "reading_score": 65,
            "reading_speed_wpm": 45,
            "pronunciation_accuracy": 78,
            "fluency_score": 60,
            "difficulty_level": "Medium",
            "difficulty_score": 0.6,
            "weak_skill": "Phonics",
            "previous_score": 70,
            "activity_completion_rate": 0.5,
            "previous_activity": "Word Games",
            "completed_item_ids": completed_ids,
        },
    )

    assert response.status_code == 200

    data = response.json()
    recommended_ids = [
        item["item_id"] for item in data["recommendations"]
    ]

    for item_id in completed_ids:
        assert item_id not in recommended_ids

    assert set(completed_ids).issubset(
        set(data["completed_items_excluded"])
    )


def test_ml_recommendations_respects_top_k():
    response = client.post(
        "/ml-recommendations?top_k=2",
        json={
            "learner_id": "TEST_RECOMMENDATION_003",
            "age": 8,
            "grade": 3,
            "preferred_topic": "Space",
            "preferred_activity": "Word Games",
            "preferred_duration_min": 15,
            "reading_score": 65,
            "reading_speed_wpm": 45,
            "pronunciation_accuracy": 78,
            "fluency_score": 60,
            "difficulty_level": "Medium",
            "difficulty_score": 0.6,
            "weak_skill": "Phonics",
            "previous_score": 70,
            "activity_completion_rate": 0.5,
            "previous_activity": "Word Games",
            "completed_item_ids": [],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert len(data["recommendations"]) <= 2
    assert len(data["recommendations"]) > 0

