import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import documents, health
from api.middleware.logging_middleware import LoggingMiddleware
from db.database import init_db
from utils.logger import get_logger
from api.routes import search
from api.routes import chat

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

app.add_middleware(LoggingMiddleware)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(search.router, prefix="/search", tags=["Search"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "Avabodh API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }