"""
api/schemas/chat.py
-------------------
Pydantic schemas for all chat endpoints.
Every request and response is validated here.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """Validated on every incoming message."""
    query:      str  = Field(min_length=1, max_length=4000,
                             description="User question")
    thread_id:  Optional[UUID] = Field(default=None,
                                       description="Existing thread ID — null for new thread")
    doc_filter: Optional[str] = Field(default=None, max_length=512,
                                      description="Limit search to specific document name")
    top_k:      int  = Field(default=5, ge=1, le=20,
                             description="Number of chunks to retrieve")

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class ThreadCreateRequest(BaseModel):
    """Optional — create a thread manually before sending messages."""
    title:      Optional[str] = Field(default=None, max_length=512)
    user_id:    Optional[str] = Field(default=None, max_length=256)
    doc_filter: Optional[str] = Field(default=None, max_length=512)


class ThreadUpdateRequest(BaseModel):
    """Update thread title."""
    title: str = Field(min_length=1, max_length=512)

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    """One source chunk used in AI response."""
    doc_name:    str
    chunk_index: int
    chunk_text:  Optional[str] = None    # preview of the chunk
    similarity:  Optional[float] = None


class ChatMessageResponse(BaseModel):
    """
    Pydantic validates every AI response before saving and returning.
    This is the output parser schema.
    """
    message_id: UUID
    thread_id:  UUID
    role:       str  = "ai"
    content:    str  = Field(min_length=1)
    sources:    list[SourceReference] = Field(default_factory=list)
    thread_title: Optional[str] = None    # set on first message only
    created_at: datetime

    @field_validator("content", mode="before")
    @classmethod
    def clean_content(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("human", "ai", "system"):
            raise ValueError("role must be human, ai or system")
        return v

    class Config:
        from_attributes = True


class ThreadResponse(BaseModel):
    """Thread details."""
    id:            UUID
    title:         Optional[str]
    user_id:       Optional[str]
    doc_filter:    Optional[str]
    message_count: int
    created_at:    datetime
    updated_at:    Optional[datetime]

    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    total:   int
    threads: list[ThreadResponse]


class MessageListResponse(BaseModel):
    """All messages in a thread."""
    thread_id: UUID
    total:     int
    messages:  list[dict]


class DeleteResponse(BaseModel):
    id:      UUID
    message: str = "Deleted successfully"