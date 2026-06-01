

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pydantic — LLM output validation  (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummaryOutput(BaseModel):
    """
    Pydantic validates every field the LLM returns.
    - Missing fields get safe defaults (never crashes)
    - Wrong types get auto-converted
    - summary_text is always cleaned of extra whitespace
    """

    doc_name: str = Field(
        default="unknown",
        description="Original filename of the document",
    )
    summary_text: str = Field(
        description="Final condensed summary of the entire document",
        min_length=10,
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Main topics/themes found in the document",
    )
    page_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages in the document",
    )
    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks processed (for audit — not stored in DB)",
    )
    source_path: str = Field(
        default="",
        description="Absolute file path of the source document",
    )
    language: str = Field(
        default="English",
        description="Detected language of the document",
    )
    model_used: str = Field(
        default="",
        description="LLM model that generated this summary",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional quality confidence 0.0-1.0",
    )

    @field_validator("summary_text", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        """Strip extra whitespace and newlines from LLM output."""
        if isinstance(v, str):
            return " ".join(v.split())
        return v

    @field_validator("doc_name", mode="before")
    @classmethod
    def sanitise_doc_name(cls, v: str) -> str:
        """Remove path separators — store only the filename."""
        import os
        return os.path.basename(str(v)) if v else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SQLAlchemy ORM — PostgreSQL table definition  (Step 7)
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class DocumentSummary(Base):
    """
    One row per document.
    Indexed on doc_name + source_path so duplicate uploads are detected fast.
    """
    __tablename__ = "document_summaries"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    doc_name      = Column(String(512), nullable=False, index=True)
    summary_text  = Column(Text, nullable=False)
    key_topics    = Column(Text, nullable=True)        # stored as comma-separated
    page_count    = Column(Integer, default=0)
    chunk_count   = Column(Integer, default=0)         # audit only
    source_path   = Column(String(1024), nullable=True)
    language      = Column(String(64), default="English")
    model_used    = Column(String(128), nullable=True)
    confidence    = Column(Float, nullable=True)
    doc_hash      = Column(String(64), nullable=True, index=True)  # SHA-256 for dedup
    created_at    = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at    = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        # Composite index — fast lookup by file path
        Index("ix_doc_summaries_source_path", "source_path"),
    )

    def __repr__(self) -> str:
        return f"<DocumentSummary doc={self.doc_name} id={self.id}>"
