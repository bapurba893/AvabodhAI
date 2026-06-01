"""
api/routes/health.py
--------------------
Health check endpoints for monitoring.
"""

from fastapi import APIRouter
from sqlalchemy import text

from db.database import engine
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", summary="Basic health check")
async def health():
    return {"status": "ok", "service": "Avabodh API"}


@router.get("/db", summary="Database health check")
async def health_db():
    """Check if PostgreSQL is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return {"status": "error", "database": str(e)}