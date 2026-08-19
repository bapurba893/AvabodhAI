"""
config/settings.py
------------------
Central configuration loaded from environment variables.
Never hardcode secrets - use .env file locally, secrets manager in production.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _preferred_env_files() -> list[str]:
    """Prefer a local demo env file when it exists, then fall back to root .env."""
    candidates: list[Path] = []
    demo_env = Path("local_demo/.env")
    if demo_env.exists():
        candidates.append(demo_env)
    root_env = Path(".env")
    if root_env.exists():
        candidates.append(root_env)
    return [str(path) for path in candidates]


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    # Ollama runs locally — no API key needed
    HUGGINGFACEHUB_API_TOKEN: str = ""   # keep empty, not used anymore

    MAP_MODEL: str = "llama3.2"
    REDUCE_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"   # local embedding model for Ollama
    # ── pgvector
    EMBEDDING_DIMENSIONS: int = 1536    # text-embedding-3-small = 1536
                                        # text-embedding-3-large = 3072
    EMBEDDING_BATCH_SIZE: int = 100     # chunks per batch to OpenAI

    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Provider selection. "auto" keeps OpenAI for production when a key is
    # supplied and switches to the local Ollama server for developer/demo use.
    LLM_PROVIDER: str = "auto"       # auto | openai | ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DOCKER_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_DIMENSIONS: int = 768

    MAP_MAX_TOKENS: int = 512
    REDUCE_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_REQUEST_TIMEOUT: int = 120
    LLM_MAX_RETRIES: int = 3

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_BREAKPOINT_THRESHOLD: float = 0.85
    MAX_CHUNK_SIZE: int = 3000
    MIN_CHUNK_SIZE: int = 100

    # ── Summary ──────────────────────────────────────────────────────────
    SUMMARY_MAX_LENGTH: int = 2000
    SUMMARY_LANGUAGE: str = "English"

    # ── PostgreSQL ────────────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "Avabodh"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres123"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # ── Restricted application role — used for ALL normal request-serving
    # queries. DB_USER/DB_PASSWORD above (typically the Postgres superuser
    # in this docker-compose setup) are used ONLY once, at startup, to
    # bootstrap the schema (create tables, the FTS trigger, this role
    # itself, and the Row-Level Security policies below).
    APP_DB_USER: str = "avabodh_app"
    APP_DB_PASSWORD: str = "CHANGE-ME-app-role-password"

    # ── Full-text search / hybrid search ──────────────────────────────────
    FTS_LANGUAGE: str = "english"
    HYBRID_RRF_K: int = 60

    # ── Document loader ──────────────────────────────────────────────────
    DOCUMENTS_DIR: str = "./documents"
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".csv"]

    # ── Image pipeline (GPT-4o Vision) ───────────────────────────────────
    VISION_MODEL: str = "gpt-4o"
    IMAGE_MIN_SIZE_BYTES: int = 5000
    IMAGE_MAX_DIMENSION: int = 1024
    IMAGE_MAX_WORKERS: int = 4

    # ── Signed preview file links ─────────────────────────────────────────
    SECRET_KEY: str = "dev-only-CHANGE-ME-in-production"
    FILE_LINK_TTL_SECONDS: Optional[int] = None

    # ── Public base URL for preview_link ──────────────────────────────────
    PUBLIC_BASE_URL: str = ""

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "avabodh.log"

    # ── LangSmith tracing ─────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Avabodh Project"

    model_config = SettingsConfigDict(
        env_file=_preferred_env_files() or None,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator(
        "DB_PORT",
        "DB_POOL_SIZE",
        "DB_MAX_OVERFLOW",
        "DB_POOL_TIMEOUT",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
        "OLLAMA_EMBEDDING_DIMENSIONS",
        "LLM_REQUEST_TIMEOUT",
        "LLM_MAX_RETRIES",
        "MAP_MAX_TOKENS",
        "REDUCE_MAX_TOKENS",
        "SUMMARY_MAX_LENGTH",
        "MAX_CHUNK_SIZE",
        "MIN_CHUNK_SIZE",
        "IMAGE_MIN_SIZE_BYTES",
        "IMAGE_MAX_DIMENSION",
        "IMAGE_MAX_WORKERS",
        "HYBRID_RRF_K",
        mode="before",
    )
    @classmethod
    def _coerce_blank_ints(cls, value: Any, info) -> Any:
        """Treat empty or whitespace-only env values as unset so defaults apply."""
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value

    @property
    def db_url(self) -> str:
        """Admin/superuser connection — used ONLY for schema bootstrap in init_db(). Not for regular queries."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def app_db_url(self) -> str:
        """Restricted app-role connection — used for all normal request-serving queries. RLS actually applies to this one."""
        return (
            f"postgresql+psycopg2://{self.APP_DB_USER}:{self.APP_DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def async_db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def use_ollama(self) -> bool:
        """Use local Ollama when explicitly requested or no OpenAI key exists."""
        provider = self.LLM_PROVIDER.strip().lower()
        if provider not in {"auto", "openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be one of: auto, openai, ollama")
        return provider == "ollama" or (provider == "auto" and not self.OPENAI_API_KEY)

    @property
    def ollama_url(self) -> str:
        """Return the first reachable Ollama base URL for the current runtime."""
        candidates = [self.OLLAMA_BASE_URL, self.OLLAMA_DOCKER_BASE_URL]
        for base_url in candidates:
            base_url = (base_url or "").strip().rstrip("/")
            if not base_url:
                continue
            try:
                with urlopen(f"{base_url}/api/version", timeout=2):
                    return base_url
            except (URLError, HTTPError, TimeoutError, ValueError):
                continue
        return self.OLLAMA_BASE_URL.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings loaded once at startup."""
    return Settings()
