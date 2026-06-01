"""
core/database.py
----------------
SQLAlchemy engine + session for FastAPI dependency injection.
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
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified / created.")
    except OperationalError as e:
        logger.critical("Cannot connect to PostgreSQL: %s", e)
        raise


# ── FastAPI dependency injection ──────────────────────────────────────────────
def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a DB session per request.
    Automatically commits on success, rolls back on error, closes always.

    Usage in route:
        async def my_route(db: Session = Depends(get_db_session)):
    """
    session = SessionFactory()
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
    """Context manager version for use outside FastAPI routes."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()