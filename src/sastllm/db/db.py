from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from sastllm.configs import settings

DATABASE_URL = f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
