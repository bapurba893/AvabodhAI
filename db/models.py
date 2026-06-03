"""
db/models.py
------------
Three things live here:
1. Pydantic model       — validates LLM output (Step 5)
2. DocumentSummary ORM  — PostgreSQL table for summaries (Step 7)
3. DocumentChunk ORM    — pgvector table for chunk embeddings (NEW)
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float, Index, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic — LLM output validation (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummaryOutput(BaseModel):
    doc_name: str = Field(default="unknown")
    summary_text: str = Field(min_length=10)
    key_topics: list[str] = Field(default_factory=list)
    page_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    source_path: str = Field(default="")
    language: str = Field(default="English")
    model_used: str = Field(default="")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("summary_text", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        if isinstance(v, str):
            return " ".join(v.split())
        return v

    @field_validator("doc_name", mode="before")
    @classmethod
    def sanitise_doc_name(cls, v: str) -> str:
        import os
        return os.path.basename(str(v)) if v else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLAlchemy ORM — Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. DocumentSummary — existing table (updated with new columns)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummary(Base):
    """One row per document — summary + metadata."""
    __tablename__ = "document_summaries"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_name      = Column(String(512), nullable=False, index=True)
    summary_text  = Column(Text, nullable=False)
    key_topics    = Column(Text, nullable=True)
    page_count    = Column(Integer, default=0)
    chunk_count   = Column(Integer, default=0)
    source_path   = Column(String(1024), nullable=True)
    language      = Column(String(64), default="English")
    model_used    = Column(String(128), nullable=True)
    confidence    = Column(Float, nullable=True)
    doc_hash      = Column(String(64), nullable=True, index=True)

    # ── NEW columns for embedding tracking ───────────────────────────────────
    avg_chunk_size     = Column(Integer, nullable=True)       # avg chars per chunk
    embedding_model    = Column(String(128), nullable=True)   # e.g. text-embedding-3-small
    embedding_status   = Column(String(32), default="pending") # pending/completed/failed
    embedding_stored_at = Column(DateTime(timezone=True), nullable=True)

    created_at    = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at    = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Relationship — one summary has many chunks
    chunks = relationship("DocumentChunk", back_populates="summary",
                          cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_doc_summaries_source_path", "source_path"),
    )

    def __repr__(self) -> str:
        return f"<DocumentSummary doc={self.doc_name} id={self.id}>"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DocumentChunk — NEW pgvector table
#    One row per chunk — stores embedding vector + metadata
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk(Base):
    """
    One row per chunk of a document.
    Linked to DocumentSummary via doc_hash.
    embedding column stores 1536-dim OpenAI vector.
    """
    __tablename__ = "document_chunks"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # ── Link to document_summaries ────────────────────────────────────────────
    summary_id    = Column(UUID(as_uuid=True),
                           ForeignKey("document_summaries.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    doc_hash      = Column(String(64), nullable=False, index=True)  # for fast lookup
    doc_name      = Column(String(512), nullable=False, index=True)

    # ── Chunk content ─────────────────────────────────────────────────────────
    chunk_index   = Column(Integer, nullable=False)       # position in document
    total_chunks  = Column(Integer, nullable=False)       # total chunks in document
    chunk_text    = Column(Text, nullable=False)          # raw text of this chunk
    chunk_size    = Column(Integer, nullable=True)        # character count

    # ── Location metadata ─────────────────────────────────────────────────────
    page_number   = Column(Integer, nullable=True)        # page this chunk came from
    source_path   = Column(String(1024), nullable=True)   # original file path
    language      = Column(String(64), default="English")

    # ── Embedding ─────────────────────────────────────────────────────────────
    # Vector(1536) = OpenAI text-embedding-3-small dimension
    # Vector(3072) = OpenAI text-embedding-3-large dimension
    embedding       = Column(Vector(1536), nullable=False)  # the actual vector
    embedding_model = Column(String(128), nullable=True)    # model used

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at    = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship back to summary
    summary = relationship("DocumentSummary", back_populates="chunks")

    __table_args__ = (
        # Composite index for fast chunk retrieval by document
        Index("ix_chunks_doc_hash_index", "doc_hash", "chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk doc={self.doc_name} chunk={self.chunk_index}/{self.total_chunks}>"