
from typing import List, Optional
from pydantic import BaseModel, Field


class ReadingAssessment(BaseModel):
    reading_speed: Optional[float] = Field(
        default=None, ge=0, description="Reading speed, using the agreed unit"
    )
    pronunciation_accuracy: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    fluency_score: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    word_error_rate: Optional[float] = Field(
        default=None, ge=0, le=100
    )


class DifficultyAssessment(BaseModel):
    difficulty_level: str
    weak_skills: List[str] = Field(default_factory=list)


class LearnerPreferences(BaseModel):
    preferred_topics: List[str] = Field(default_factory=list)
    preferred_activity_types: List[str] = Field(default_factory=list)
    preferred_duration_minutes: int = Field(default=15, gt=0)


class LearningHistory(BaseModel):
    completed_activity_ids: List[str] = Field(default_factory=list)
    previous_scores: List[float] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    learner_id: str
    reading_assessment: ReadingAssessment
    difficulty_assessment: DifficultyAssessment
    preferences: LearnerPreferences
    learning_history: LearningHistory = Field(
        default_factory=LearningHistory
    )


class RecommendedItem(BaseModel):
    item_id: str
    title: str
    item_type: str
    topic: str
    target_skill: str
    difficulty_level: str
    duration_minutes: int
    reason: str
    score: float = Field(ge=0, le=1)


class RecommendationResponse(BaseModel):
    learner_id: str
    recommendations: List[RecommendedItem]
