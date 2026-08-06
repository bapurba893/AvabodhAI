"""
config/settings.py
------------------
Central configuration loaded from environment variables.
Never hardcode secrets — use .env file locally, secrets manager in production.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
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

    #OLLAMA_BASE_URL: str = "http://localhost:11434"  # default Ollama port
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    MAP_MAX_TOKENS: int = 512
    REDUCE_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_REQUEST_TIMEOUT: int = 120
    LLM_MAX_RETRIES: int = 3

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_BREAKPOINT_THRESHOLD: float = 0.85
    MAX_CHUNK_SIZE: int = 3000
    MIN_CHUNK_SIZE: int = 100

    # ── Summary ────────────────────────────────────────────────────────────
    SUMMARY_MAX_LENGTH: int = 2000
    SUMMARY_LANGUAGE: str = "English"

    # ── PostgreSQL ─────────────────────────────────────────────────────────
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
    #
    # This separation exists because Postgres RLS CANNOT restrict a
    # superuser's queries, ever, by design — no policy setting or FORCE
    # clause changes that. If the application connected using the same
    # superuser credentials for everything, the RLS policies below would
    # provide zero real protection for the app's own traffic. This role
    # is deliberately NOT a superuser and does NOT have BYPASSRLS, so RLS
    # actually applies to it.
    #
    # Anyone (including teammates in pgAdmin) who wants tenant/org-unit
    # isolation to actually bound their ad-hoc queries must connect using
    # THESE credentials, not DB_USER/DB_PASSWORD. Connecting as the
    # superuser bypasses every isolation guarantee in this document —
    # see the reference document for the full explanation.
    APP_DB_USER: str = "avabodh_app"
    APP_DB_PASSWORD: str = "CHANGE-ME-app-role-password"

    # ── Full-text search / hybrid search ────────────────────────────────────
    FTS_LANGUAGE: str = "english"          # Postgres text search config used by the trigger and keyword queries
    HYBRID_RRF_K: int = 60                 # Reciprocal Rank Fusion constant — standard default (same as Elasticsearch's)

    # ── Document loader ────────────────────────────────────────────────────
    DOCUMENTS_DIR: str = "./documents"
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".csv"]

    # ── Image pipeline (GPT-4o Vision) ─────────────────────────────────────
    VISION_MODEL: str = "gpt-4o"
    IMAGE_MIN_SIZE_BYTES: int = 5000       # images smaller than this are treated as noise (icons, tracking pixels)
    IMAGE_MAX_DIMENSION: int = 1024        # resize longest side to this before sending to Vision
    IMAGE_MAX_WORKERS: int = 4             # ThreadPoolExecutor size for parallel image embedding

    # ── Signed preview file links ───────────────────────────────────────────
    # A plain <a href> or <iframe src> can't carry X-Tenant-ID/X-Org-Unit-ID
    # headers, so the document-file endpoint is authorized via a short-lived
    # HMAC-signed token in the URL instead. SECRET_KEY MUST be overridden in
    # production — the default below is fine for local dev only, and is
    # deliberately obviously-not-secret so nobody mistakes it for a real one.
    SECRET_KEY: str = "dev-only-CHANGE-ME-in-production"
    FILE_LINK_TTL_SECONDS: int = 900       # 15 minutes


    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "avabodh.log"

    # ── LangSmith tracing ──────────────────────────────────────────────────
    # Get your key: https://smith.langchain.com -> Settings -> API Keys
    # Set LANGCHAIN_TRACING_V2=true in .env to enable
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Avabodh Project"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

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


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings loaded once at startup."""
    return Settings()