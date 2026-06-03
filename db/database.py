"""
db/database.py
--------------
SQLAlchemy engine + session factory.
Updated to enable pgvector extension on startup.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from config.settings import get_settings
from db.models import Base
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

engine = create_engine(
    settings.db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """
    Create all tables if they don't exist.
    Also enables pgvector extension — required before creating vector columns.
    """
    try:
        with engine.connect() as conn:
            # Enable pgvector extension — must run before creating vector columns
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled.")

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified / created successfully.")
    except OperationalError as e:
        logger.critical("Cannot connect to PostgreSQL: %s", e)
        raise


def check_db_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return False


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    session: Session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_session_context() -> Generator[Session, None, None]:
    """Context manager for use outside FastAPI routes."""
    session: Session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session_fastapi() -> Generator[Session, None, None]:
    """FastAPI dependency injection version."""
    session: Session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()