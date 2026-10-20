from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.models import WeeklyPlan
from app.db.session import SessionLocal
from app.services.weekly_plan_service import generate_weekly_plan

from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.recommendation_service import (
    generate_recommendations,
)
from app.services.ml_recommendation_service import (
    predict_recommended_activity,
)
from app.services.progress_service import (
    record_activity,
    get_learner_progress,
)



# --------------------------------------------------
# 1. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Dyslexia Reading Assistance System",
    description=(
        "Backend API for the AI-Based Adaptive Reading "
        "Assistance System for Children with Dyslexia"
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 2. Root and health endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Dyslexia Reading Assistance API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# --------------------------------------------------
# 3. Rule-based recommendations
# --------------------------------------------------

@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
def get_recommendations(
    request: RecommendationRequest,
    top_k: int = Query(default=3, ge=1, le=10),
):
    try:
        recommendations = generate_recommendations(
            request=request,
            top_k=top_k,
        )

        return RecommendationResponse(
            learner_id=request.learner_id,
            recommendations=recommendations,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# 4. ML-based personalized recommendations
# --------------------------------------------------

@app.post("/ml-recommendations")
def get_ml_recommendations(
    learner_features: dict,
    top_k: int = Query(default=3, ge=1, le=10),
):
    try:
        learner_features = learner_features.copy()

        # Validate learner ID.
        learner_id = learner_features.get("learner_id")

        if (
            not isinstance(learner_id, str)
            or not learner_id.strip()
        ):
            raise ValueError(
                "learner_id is required to retrieve learner progress."
            )

        learner_id = learner_id.strip()

        # Validate additional completed item IDs.
        supplied_ids = learner_features.pop(
            "completed_item_ids", []
        )

        if not isinstance(supplied_ids, list):
            raise ValueError(
                "completed_item_ids must be a list of item IDs."
            )

        completed_item_ids = []

        for item_id in supplied_ids:
            if (
                not isinstance(item_id, str)
                or not item_id.strip()
            ):
                raise ValueError(
                    "Each completed item ID must be a "
                    "non-empty string."
                )

            completed_item_ids.append(item_id.strip())

        # IMPORTANT:
        # Keep learner_id in learner_features.
        # ml_recommendation_service.py needs it to retrieve
        # saved scores and update learner features.
        result = predict_recommended_activity(
            learner_features=learner_features,
            top_k=top_k,
            completed_item_ids=completed_item_ids,
        )

        # The ML service returns progress and performance
        # adaptation information in its response.
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# 5. Activity progress request schema
# --------------------------------------------------
# --------------------------------------------------
# Weekly plan request schema
# --------------------------------------------------

class WeeklyPlanRequest(BaseModel):
    learner_id: str = Field(min_length=1)
    week_number: int = Field(ge=1)
    learner_features: dict
    
class ActivityProgressRequest(BaseModel):
    learner_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=100)
    completed: bool = True


# --------------------------------------------------
# 6. Record activity progress
# --------------------------------------------------

@app.post("/progress/record")
def save_activity_progress(
    request: ActivityProgressRequest,
):
    try:
        return record_activity(
            learner_id=request.learner_id.strip(),
            item_id=request.item_id.strip(),
            score=request.score,
            completed=request.completed,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# 7. Retrieve learner progress
# --------------------------------------------------

@app.get("/progress/{learner_id}")
def read_learner_progress(learner_id: str):
    try:
        learner_id = learner_id.strip()

        if not learner_id:
            raise ValueError(
                "learner_id must not be empty."
            )

        return get_learner_progress(learner_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
        
        
# --------------------------------------------------
# 8. Generate a weekly learning plan
# --------------------------------------------------

@app.post("/weekly-plans")
def create_weekly_plan(request: WeeklyPlanRequest):
    try:
        learner_id = request.learner_id.strip()

        if not learner_id:
            raise ValueError("learner_id must not be empty.")

        result = generate_weekly_plan(
            learner_id=learner_id,
            week_number=request.week_number,
            learner_features=request.learner_features,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# 9. Retrieve a saved weekly learning plan
# --------------------------------------------------

@app.get("/weekly-plans/{learner_id}/{week_number}")
def read_weekly_plan(learner_id: str, week_number: int):
    learner_id = learner_id.strip()

    if not learner_id:
        raise HTTPException(
            status_code=400,
            detail="learner_id must not be empty.",
        )

    if week_number < 1:
        raise HTTPException(
            status_code=400,
            detail="week_number must be at least 1.",
        )

    db = SessionLocal()

    try:
        plan = (
            db.query(WeeklyPlan)
            .filter(
                WeeklyPlan.learner_id == learner_id,
                WeeklyPlan.week_number == week_number,
            )
            .first()
        )

        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No saved plan found for learner "
                    f"'{learner_id}', week {week_number}."
                ),
            )

        return {
            "learner_id": plan.learner_id,
            "week_number": plan.week_number,
            "activities": plan.activities,
            "learner_features": plan.learner_features,
            "generated_at": (
                plan.generated_at.isoformat()
                if plan.generated_at
                else None
            ),
            "status": "retrieved",
        }

    finally:
        db.close()
