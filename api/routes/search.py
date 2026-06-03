"""
api/routes/search.py
--------------------
Semantic search endpoint using pgvector.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field

from db.database import get_db_session_fastapi
from db.models import DocumentChunk
from langchain_openai import OpenAIEmbeddings
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


class SearchRequest(BaseModel):
    query:    str = Field(min_length=1, description="Search query")
    top_k:    int = Field(default=5, ge=1, le=20, description="Number of results")
    doc_name: str = Field(default=None, description="Filter by document name (optional)")


class ChunkResult(BaseModel):
    id:          str
    doc_name:    str
    chunk_index: int
    chunk_text:  str
    chunk_size:  int
    page_number: int = None
    similarity:  float


class SearchResponse(BaseModel):
    query:   str
    results: list[ChunkResult]
    total:   int


@router.post("/", response_model=SearchResponse, summary="Semantic search across documents")
async def semantic_search(
    payload: SearchRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    """Find most relevant chunks for a query using vector similarity."""
    try:
        embeddings_client = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
        query_vector = embeddings_client.embed_query(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding query failed: {e}")

    try:
        if payload.doc_name:
            results = db.execute(
                text("""
                    SELECT id, doc_name, chunk_index, chunk_text, chunk_size,
                           page_number,
                           1 - (embedding <=> CAST(:vector AS vector)) as similarity
                    FROM document_chunks
                    WHERE doc_name = :doc_name
                    ORDER BY embedding <=> CAST(:vector AS vector)
                    LIMIT :top_k
                """),
                {"vector": str(query_vector), "doc_name": payload.doc_name, "top_k": payload.top_k}
            ).fetchall()
        else:
            results = db.execute(
                text("""
                    SELECT id, doc_name, chunk_index, chunk_text, chunk_size,
                           page_number,
                           1 - (embedding <=> CAST(:vector AS vector)) as similarity
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:vector AS vector)
                    LIMIT :top_k
                """),
                {"vector": str(query_vector), "top_k": payload.top_k}
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
            )
            for r in results
        ],
        total=len(results),
    )


@router.get("/chunks/{doc_id}", summary="Get all chunks for a document")
async def get_document_chunks(
    doc_id: uuid.UUID,
    db: Session = Depends(get_db_session_fastapi),
):
    """Get all stored chunks and metadata for a document."""
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.summary_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for ID '{doc_id}'")

    return {
        "doc_id": str(doc_id),
        "doc_name": chunks[0].doc_name,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": str(c.id), "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text, "chunk_size": c.chunk_size,
                "page_number": c.page_number, "model": c.embedding_model,
                "created_at": str(c.created_at),
            }
            for c in chunks
        ]
    }