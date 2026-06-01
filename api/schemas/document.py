"""
api/schemas/document.py
-----------------------
Pydantic schemas for request and response validation.

Every API input and output is validated here.
FastAPI uses these automatically — wrong data type = 422 error with clear message.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ── Request schemas (what user sends) ────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """
    Returned after successful document upload and summarisation.
    Every field validated by Pydantic before sending to client.
    """
    id:           UUID
    doc_name:     str
    summary_text: str
    key_topics:   Optional[str] = None
    page_count:   int
    chunk_count:  int
    source_path:  Optional[str] = None
    language:     str
    model_used:   Optional[str] = None
    doc_hash:     Optional[str] = None
    created_at:   datetime
    updated_at:   Optional[datetime] = None
    elapsed_sec:  Optional[float] = None    # how long summarisation took

    class Config:
        from_attributes = True              # allows creating from SQLAlchemy ORM


class DocumentListItem(BaseModel):
    """Lightweight schema for listing documents — no full summary text."""
    id:          UUID
    doc_name:    str
    page_count:  int
    chunk_count: int
    model_used:  Optional[str] = None
    created_at:  datetime
    # First 200 chars of summary as preview
    summary_preview: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Wrapper for list of documents with pagination info."""
    total:     int
    page:      int
    per_page:  int
    documents: list[DocumentListItem]


class DocumentDetailResponse(BaseModel):
    """Full document detail — same as upload response."""
    id:           UUID
    doc_name:     str
    summary_text: str
    key_topics:   Optional[str] = None
    page_count:   int
    chunk_count:  int
    source_path:  Optional[str] = None
    language:     str
    model_used:   Optional[str] = None
    doc_hash:     Optional[str] = None
    created_at:   datetime
    updated_at:   Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentUpdateRequest(BaseModel):
    """What user can update — only doc_name for now."""
    doc_name: str = Field(
        min_length=1,
        max_length=512,
        description="New name for the document",
    )

    @field_validator("doc_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


class DocumentUpdateResponse(BaseModel):
    """Returned after successful update."""
    id:         UUID
    doc_name:   str
    updated_at: Optional[datetime] = None
    message:    str = "Document updated successfully"

    class Config:
        from_attributes = True


class DeleteResponse(BaseModel):
    """Returned after successful delete."""
    id:      UUID
    message: str = "Document deleted successfully"


# ── Error schemas ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response shape."""
    error:   str
    detail:  Optional[str] = None
    status:  int


class UploadErrorResponse(BaseModel):
    """Specific error for upload failures."""
    error:     str
    step:      str    # which step failed: load, split, summarise, save
    detail:    Optional[str] = None
    status:    int = 500


# ── Processing status schema (for long operations) ────────────────────────────

class ProcessingStatus(BaseModel):
    """Shows current step during processing."""
    step:        str
    message:     str
    progress:    int    # 0-100
    completed:   bool = False