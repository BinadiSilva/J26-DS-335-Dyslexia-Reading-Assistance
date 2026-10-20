
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise RuntimeError("DATABASE_URL is missing from .env")

engine = create_engine(database_url, pool_pre_ping=True)

try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        print("Supabase connection successful!")
except Exception as error:
    print(f"Connection failed: {type(error).__name__}: {error}")
finally:
    engine.dispose()
