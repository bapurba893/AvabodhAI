"""
db/models.py
------------
Four ORM tables:
1. DocumentSummaryOutput  — Pydantic LLM output validator
2. DocumentSummary        — document summaries table (+ document-level metadata)
3. DocumentChunk          — pgvector chunk embeddings table (+ chunk-level metadata)
4. ChatThread             — chat conversation threads
5. ChatMessage            — individual chat messages with embeddings
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float,
    Index, ForeignKey, JSON, Boolean
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic — LLM output validation
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummaryOutput(BaseModel):
    doc_name:         str   = Field(default="unknown")
    summary_text:     str   = Field(min_length=10)
    key_topics:       list[str] = Field(default_factory=list)
    page_count:       int   = Field(default=0, ge=0)
    chunk_count:      int   = Field(default=0, ge=0)
    source_path:      str   = Field(default="")
    language:         str   = Field(default="English")
    model_used:       str   = Field(default="")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("summary_text", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("doc_name", mode="before")
    @classmethod
    def sanitise_doc_name(cls, v: str) -> str:
        import os
        return os.path.basename(str(v)) if v else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# NEW — Pydantic schemas for metadata extraction (LLM structured output)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadataOutput(BaseModel):
    """
    Document-level metadata — extracted once per document via LLM,
    using structured output during/after the Reduce step.
    """
    title:                str  = Field(default="", description="Real document title, not filename")
    author:               Optional[str] = Field(default=None, description="Author if mentioned in document")
    document_type:        str  = Field(default="other",
                                       description="resume, research_paper, contract, report, invoice, manual, article, other")
    domain:                str  = Field(default="general",
                                       description="legal, technical, financial, academic, medical, general")
    detected_language:    str  = Field(default="English")
    key_entities:          list[str] = Field(default_factory=list, description="Organizations, people, locations mentioned")
    mentioned_dates:       list[str] = Field(default_factory=list, description="Any dates referenced in the document")
    target_audience:       str  = Field(default="general", description="technical, general, executive")
    sentiment:              str  = Field(default="neutral", description="positive, negative, neutral, critical")
    confidentiality_level: str  = Field(default="public", description="public, internal, confidential")

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v) -> str:
        return v.strip()[:512] if isinstance(v, str) else ""

    @field_validator("document_type", "domain", "target_audience", "sentiment", "confidentiality_level", mode="before")
    @classmethod
    def lowercase_enum(cls, v) -> str:
        return str(v).strip().lower() if v else "other"


class ChunkMetadataOutput(BaseModel):
    """
    Chunk-level metadata — extracted during the Map step alongside
    the existing summary, using the SAME LLM call (no extra cost).
    """
    summary:          str  = Field(min_length=1, description="Concise summary of this section")
    section_heading:  Optional[str] = Field(default=None, description="Section/heading this chunk likely belongs to")
    chunk_type:       str  = Field(default="paragraph", description="paragraph, table, list, heading, code")
    topic:             str  = Field(default="general", description="2-3 word topic label for this chunk")
    entities:           list[str] = Field(default_factory=list, description="Named entities mentioned in this chunk")
    contains_data:      bool = Field(default=False, description="True if chunk has numbers, statistics, or tabular data")
    confidence_score:   float = Field(default=0.8, ge=0.0, le=1.0, description="LLM confidence in this extraction")

    @field_validator("chunk_type", mode="before")
    @classmethod
    def validate_chunk_type(cls, v) -> str:
        allowed = {"paragraph", "table", "list", "heading", "code"}
        v = str(v).strip().lower() if v else "paragraph"
        return v if v in allowed else "paragraph"

    @field_validator("summary", "topic", mode="before")
    @classmethod
    def clean_text_field(cls, v) -> str:
        return " ".join(str(v).split()) if v else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLAlchemy Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. DocumentSummary table — with document-level metadata columns
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummary(Base):
    __tablename__ = "document_summaries"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_name            = Column(String(512), nullable=False, index=True)
    summary_text        = Column(Text, nullable=False)
    key_topics          = Column(Text, nullable=True)
    page_count          = Column(Integer, default=0)
    chunk_count         = Column(Integer, default=0)
    source_path         = Column(String(1024), nullable=True)
    language            = Column(String(64), default="English")
    model_used          = Column(String(128), nullable=True)
    confidence          = Column(Float, nullable=True)
    doc_hash            = Column(String(64), nullable=True, index=True)
    avg_chunk_size      = Column(Integer, nullable=True)
    embedding_model     = Column(String(128), nullable=True)
    embedding_status    = Column(String(32), default="pending")
    embedding_stored_at = Column(DateTime(timezone=True), nullable=True)

    # ── NEW — Document-level extracted metadata ───────────────────────────────
    title                  = Column(String(512), nullable=True, index=True)
    author                 = Column(String(256), nullable=True)
    document_type          = Column(String(64), nullable=True, index=True)   # resume/contract/report/etc
    domain                 = Column(String(64), nullable=True, index=True)   # legal/technical/financial/etc
    detected_language      = Column(String(64), nullable=True)
    key_entities            = Column(ARRAY(String), nullable=True)            # ["Avik Bhattacharya", "IIT Bombay"]
    mentioned_dates         = Column(ARRAY(String), nullable=True)
    target_audience         = Column(String(64), nullable=True)
    sentiment                = Column(String(32), nullable=True)
    confidentiality_level   = Column(String(32), nullable=True, default="public")
    metadata_status          = Column(String(32), default="pending")          # pending/completed/failed
    metadata_extracted_at   = Column(DateTime(timezone=True), nullable=True)

    created_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))

    chunks = relationship("DocumentChunk", back_populates="summary",
                          cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_doc_summaries_source_path", "source_path"),
        Index("ix_doc_summaries_doc_type_domain", "document_type", "domain"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. DocumentChunk table — pgvector — with chunk-level metadata columns
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    summary_id      = Column(UUID(as_uuid=True),
                             ForeignKey("document_summaries.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    doc_hash        = Column(String(64), nullable=False, index=True)
    doc_name        = Column(String(512), nullable=False, index=True)
    chunk_index     = Column(Integer, nullable=False)
    total_chunks    = Column(Integer, nullable=False)
    chunk_text      = Column(Text, nullable=False)
    chunk_size      = Column(Integer, nullable=True)
    page_number     = Column(Integer, nullable=True)
    source_path     = Column(String(1024), nullable=True)
    language        = Column(String(64), default="English")
    embedding       = Column(Vector(1536), nullable=False)
    embedding_model = Column(String(128), nullable=True)

    # ── NEW — Chunk-level extracted metadata ──────────────────────────────────
    section_heading   = Column(String(512), nullable=True, index=True)   # "Introduction", "Methodology"
    chunk_type         = Column(String(32), nullable=True, default="paragraph")  # paragraph/table/list/heading/code
    topic               = Column(String(128), nullable=True, index=True)
    entities             = Column(ARRAY(String), nullable=True)
    contains_data        = Column(Boolean, default=False)
    metadata_confidence  = Column(Float, nullable=True)   # LLM confidence in extracted metadata

    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc), nullable=False)

    summary = relationship("DocumentSummary", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_hash_index", "doc_hash", "chunk_index"),
        Index("ix_chunks_topic", "topic"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChatThread
# ─────────────────────────────────────────────────────────────────────────────

class ChatThread(Base):
    __tablename__ = "chat_threads"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title      = Column(String(512), nullable=True)
    user_id    = Column(String(256), nullable=True)
    doc_filter = Column(String(512), nullable=True)
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="thread",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")

    __table_args__ = (
        Index("ix_chat_threads_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id} title={self.title}>"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ChatMessage
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True),
                       ForeignKey("chat_threads.id", ondelete="CASCADE"),
                       nullable=False, index=True)

    role      = Column(String(16), nullable=False)
    content   = Column(Text, nullable=False)

    embedding       = Column(Vector(1536), nullable=True)
    embedding_model = Column(String(128), nullable=True)

    sources   = Column(JSON, nullable=True)

    prompt_tokens     = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)

    thread = relationship("ChatThread", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_thread_role", "thread_id", "role"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage role={self.role} thread={self.thread_id}>"