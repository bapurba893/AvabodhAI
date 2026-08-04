import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import documents, health
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.tenant_guard_middleware import TenantGuardMiddleware
from db.database import init_db
from utils.logger import get_logger
from api.routes import search
from api.routes import chat
from api.routes.web import router as web_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Avabodh API...")
    init_db()
    logger.info("Database tables verified.")
    yield
    logger.info("Shutting down Avabodh API...")


app = FastAPI(
    title="Avabodh API",
    description="Document Intelligence API — Upload documents, get AI summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters here: LoggingMiddleware must be added LAST so it ends up
# outermost — that way every request, including ones TenantGuardMiddleware
# rejects with a 400, still shows up in the standard access log with its
# real status code, not just the guard's own warning line.
app.add_middleware(TenantGuardMiddleware)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(web_router, prefix="/web", tags=["Web Scraping"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "Avabodh API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }