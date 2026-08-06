"""
api/routes/search.py
--------------------
Semantic search endpoint using pgvector.

Role-aware search — a query can be scoped to text chunks only, image
chunks only, or both (default). Image chunks surface their caption/type
alongside the usual chunk fields so callers can render them differently
from text results.

Multi-tenancy + department isolation: both endpoints take tenant_id AND
org_unit_id from the X-Tenant-ID / X-Org-Unit-ID headers and apply both
as mandatory filters, together — this is a search endpoint, exactly the
kind of query that would leak another tenant's, or another department's,
document content if either filter were ever dropped. is_ground_truth is
always applied too, unconditionally, same as every other retrieval path.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field, field_validator

from api.dependencies import get_tenant_id, get_org_unit_id
from db.database import get_db_session_fastapi
from db.models import DocumentChunk
from pipeline.retriever import embed_query
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

_ALLOWED_ROLES = {"text", "image"}


class SearchRequest(BaseModel):
    query:    str = Field(min_length=1, description="Search query")
    top_k:    int = Field(default=5, ge=1, le=20, description="Number of results")
    doc_name: Optional[str] = Field(default=None, description="Filter by document name (optional)")
    # Scope search to text chunks, image chunks, or both (default None = both)
    role:     Optional[str] = Field(
        default=None,
        description="Filter by chunk role: 'text' or 'image'. Omit to search both.",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(_ALLOWED_ROLES)}")
        return v


class ChunkResult(BaseModel):
    id:          str
    doc_name:    str
    chunk_index: int
    chunk_text:  str
    chunk_size:  int
    page_number: Optional[int] = None
    similarity:  float
    role:            str = "text"
    image_type:      Optional[str] = None
    image_caption:   Optional[str] = None
    section_heading: Optional[str] = None
    topic:           Optional[str] = None
    chunk_type:      Optional[str] = None


class SearchResponse(BaseModel):
    query:   str
    results: list[ChunkResult]
    total:   int


@router.post("/", response_model=SearchResponse, summary="Semantic search across documents")
async def semantic_search(
    payload: SearchRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    """
    Find most relevant chunks for a query using vector similarity.
    By default searches BOTH text and image chunks — pass role='text' or
    role='image' to scope the search to just one kind. Always scoped to
    the caller's tenant_id + org_unit_id, and always ground-truth-only.
    """
    try:
        query_vector = embed_query(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding query failed: {e}")

    where_clauses = [
        "tenant_id = :tenant_id",
        "org_unit_id = :org_unit_id",
        "is_ground_truth = true",
    ]
    params: dict = {
        "vector": str(query_vector), "top_k": payload.top_k,
        "tenant_id": tenant_id, "org_unit_id": org_unit_id,
    }

    if payload.doc_name:
        where_clauses.append("doc_name = :doc_name")
        params["doc_name"] = payload.doc_name

    if payload.role:
        where_clauses.append("role = :role")
        params["role"] = payload.role

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    try:
        results = db.execute(
            text(f"""
                SELECT id, doc_name, chunk_index, chunk_text, chunk_size,
                       page_number, role, image_type, image_caption,
                       section_heading, topic, chunk_type,
                       1 - (embedding <=> CAST(:vector AS vector)) as similarity
                FROM document_chunks
                {where_sql}
                ORDER BY embedding <=> CAST(:vector AS vector)
                LIMIT :top_k
            """),
            params,
        ).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return SearchResponse(
        query=payload.query,
        results=[
            ChunkResult(
                id=str(r.id), doc_name=r.doc_name,
                chunk_index=r.chunk_index, chunk_text=r.chunk_text,
                chunk_size=r.chunk_size, page_number=r.page_number,
                similarity=round(float(r.similarity), 4),
                role=r.role or "text",
                image_type=r.image_type,
                image_caption=r.image_caption,
                section_heading=r.section_heading,
                topic=r.topic,
                chunk_type=r.chunk_type,
            )
            for r in results
        ],
        total=len(results),
    )


@router.get("/chunks/{doc_id}", summary="Get all chunks for a document")
async def get_document_chunks(
    doc_id: uuid.UUID,
    role: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    """
    Get all stored chunks and metadata for a document, scoped to
    tenant_id + org_unit_id. Pass role='text' or role='image' to filter
    to just one kind — the response always reports both text_chunks and
    image_chunks counts regardless of the filter, so callers can see the
    full breakdown. NOT filtered by is_ground_truth — this is a document-
    management view of a document you already own, not a retrieval path,
    so it deliberately shows all chunks including non-ground-truth ones.
    """
    if role is not None:
        role = role.strip().lower()
        if role not in _ALLOWED_ROLES:
            raise HTTPException(status_code=422, detail=f"role must be one of {sorted(_ALLOWED_ROLES)}")

    base_query = db.query(DocumentChunk).filter(
        DocumentChunk.summary_id == doc_id,
        DocumentChunk.tenant_id == tenant_id,
        DocumentChunk.org_unit_id == org_unit_id,
    )

    # Counts by role — computed independent of the role filter below
    text_count = base_query.filter(DocumentChunk.role == "text").count()
    image_count = base_query.filter(DocumentChunk.role == "image").count()

    query = base_query
    if role:
        query = query.filter(DocumentChunk.role == role)

    chunks = query.order_by(DocumentChunk.chunk_index).all()

    if not chunks and text_count == 0 and image_count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for ID '{doc_id}'")

    return {
        "doc_id": str(doc_id),
        "doc_name": chunks[0].doc_name if chunks else None,
        "total_chunks": len(chunks),
        "text_chunks": text_count,
        "image_chunks": image_count,
        "chunks": [
            {
                "id": str(c.id), "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text, "chunk_size": c.chunk_size,
                "page_number": c.page_number, "model": c.embedding_model,
                "role": c.role or "text",
                "image_type": c.image_type,
                "image_caption": c.image_caption,
                "contains_chart": c.contains_chart,
                "contains_table": c.contains_table,
                "vision_confidence": c.vision_confidence,
                "is_ground_truth": c.is_ground_truth,
                "section_heading": c.section_heading,
                "topic": c.topic,
                "chunk_type": c.chunk_type,
                "created_at": str(c.created_at),
            }
            for c in chunks
        ]
    }