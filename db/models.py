"""
db/models.py
------------
Four ORM tables:
1. DocumentSummaryOutput  — Pydantic LLM output validator
2. DocumentSummary        — document summaries table
3. DocumentChunk          — pgvector chunk embeddings table
4. ChatThread             — NEW: chat conversation threads
5. ChatMessage            — NEW: individual chat messages with embeddings
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float,
    Index, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
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
# 2. SQLAlchemy Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. DocumentSummary table
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
    created_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))

    chunks = relationship("DocumentChunk", back_populates="summary",
                          cascade="all, delete-orphan")

    __table_args__ = (Index("ix_doc_summaries_source_path", "source_path"),)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DocumentChunk table — pgvector
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
    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc), nullable=False)

    summary = relationship("DocumentSummary", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_hash_index", "doc_hash", "chunk_index"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChatThread — NEW
#    One row per conversation thread
# ─────────────────────────────────────────────────────────────────────────────

class ChatThread(Base):
    """
    Represents a conversation thread (like a chat session).
    One thread has many ChatMessages.
    Title is auto-generated by LLM from first message.
    """
    __tablename__ = "chat_threads"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title      = Column(String(512), nullable=True)       # auto-generated by LLM
    user_id    = Column(String(256), nullable=True)       # optional user identifier
    doc_filter = Column(String(512), nullable=True)       # optional: filter to specific doc
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # One thread has many messages
    messages = relationship("ChatMessage", back_populates="thread",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")

    __table_args__ = (
        Index("ix_chat_threads_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id} title={self.title}>"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ChatMessage — NEW
#    One row per message (human OR ai — never mixed in one row)
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(Base):
    """
    One row per message.
    role = 'human' or 'ai' — never in same row.
    embedding stores the message vector for semantic search across chat history.
    sources stores which document chunks were used to generate AI response.
    """
    __tablename__ = "chat_messages"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True),
                       ForeignKey("chat_threads.id", ondelete="CASCADE"),
                       nullable=False, index=True)

    # ── Message content ───────────────────────────────────────────────────────
    role      = Column(String(16), nullable=False)         # 'human' or 'ai'
    content   = Column(Text, nullable=False)               # message text

    # ── Embedding — stored for semantic search across chat history ────────────
    embedding       = Column(Vector(1536), nullable=True)  # message vector
    embedding_model = Column(String(128), nullable=True)   # model used

    # ── Sources — which chunks were used (AI messages only) ──────────────────
    # Stored as JSON: [{"doc_name": "x.pdf", "chunk_index": 3, "similarity": 0.92}]
    sources   = Column(JSON, nullable=True)

    # ── Token usage tracking ─────────────────────────────────────────────────
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