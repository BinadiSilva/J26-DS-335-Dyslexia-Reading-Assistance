
from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    JSON,
    UniqueConstraint,
)


class ActivityProgress(Base):
    __tablename__ = "activity_progress"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    learner_id: Mapped[str] = mapped_column(
        String(100), index=True, nullable=False
    )
    item_id: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    score: Mapped[float] = mapped_column(
        Float, nullable=False
    )
    completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

class WeeklyPlan(Base):
    __tablename__ = "weekly_plans"

    __table_args__ = (
        UniqueConstraint(
            "learner_id",
            "week_number",
            name="uq_learner_week",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    learner_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )

    week_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    activities: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    learner_features: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )