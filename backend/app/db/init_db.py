from app.db.session import Base, engine
from app.db.models import ActivityProgress, WeeklyPlan


def initialize_database():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    initialize_database()
    print("Database tables created successfully.")