"""
pipeline/embedder.py
--------------------
NEW Step — Generate and store chunk embeddings in pgvector.

Flow:
    chunks (from splitter) 
        → OpenAI Embeddings (parallel)
        → DocumentChunk records
        → PostgreSQL with pgvector

Features:
- Parallel embedding generation (all chunks at once)
- Pydantic validation per chunk
- Updates DocumentSummary.embedding_status
- Skips already-embedded documents (dedup)
"""

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import make_transient

from db.database import get_db_session_context
from db.models import DocumentChunk, DocumentSummary
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema — validates each chunk before saving to DB
# ─────────────────────────────────────────────────────────────────────────────

class ChunkEmbeddingInput(BaseModel):
    """Validates chunk data before embedding and storing."""
    chunk_index:  int   = Field(ge=0)
    total_chunks: int   = Field(ge=1)
    chunk_text:   str   = Field(min_length=1)
    chunk_size:   int   = Field(ge=0)
    page_number:  Optional[int] = Field(default=None)
    source_path:  Optional[str] = Field(default=None)
    language:     str   = Field(default="English")
    doc_name:     str   = Field(min_length=1)
    doc_hash:     str   = Field(min_length=1)

    @field_validator("chunk_text", mode="before")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


# ─────────────────────────────────────────────────────────────────────────────
# Embedding functions
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings_client() -> OpenAIEmbeddings:
    """Build OpenAI embeddings client."""
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
    )


def _embed_batch(texts: list[str], client: OpenAIEmbeddings) -> list[list[float]]:
    """
    Embed a batch of texts in one API call.
    OpenAI supports up to 2048 texts per batch.
    Batching is much cheaper than one call per chunk.
    """
    return client.embed_documents(texts)


def check_embeddings_exist(doc_hash: str) -> bool:
    """Check if embeddings already stored for this document."""
    with get_db_session_context() as session:
        count = (
            session.query(DocumentChunk)
            .filter(DocumentChunk.doc_hash == doc_hash)
            .count()
        )
        return count > 0


def store_chunk_embeddings(
    chunks: list[Document],
    summary_id: UUID,
    doc_hash: str,
    doc_name: str,
    source_path: str = "",
) -> dict:
    """
    Main function — generates and stores embeddings for all chunks.

    Steps:
    1. Validate all chunks with Pydantic
    2. Generate embeddings in batches (parallel)
    3. Save all DocumentChunk records to PostgreSQL
    4. Update DocumentSummary.embedding_status

    Returns summary dict with counts and timing.
    """
    if not chunks:
        raise ValueError(f"No chunks provided for '{doc_name}'")

    # Check dedup — skip if already embedded
    if check_embeddings_exist(doc_hash):
        logger.info("Embeddings already exist for '%s' — skipping", doc_name)
        return {
            "status": "skipped",
            "doc_name": doc_name,
            "chunk_count": 0,
            "elapsed_sec": 0.0,
        }

    logger.info(
        "Starting embedding for '%s' — %d chunks | model=%s",
        doc_name, len(chunks), settings.EMBEDDING_MODEL,
    )

    start = time.time()
    total_chunks = len(chunks)

    # ── Step 1: Validate all chunks with Pydantic ─────────────────────────
    validated_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            validated = ChunkEmbeddingInput(
                chunk_index  = i,
                total_chunks = total_chunks,
                chunk_text   = chunk.page_content,
                chunk_size   = len(chunk.page_content),
                page_number  = chunk.metadata.get("page", None),
                source_path  = source_path,
                language     = settings.SUMMARY_LANGUAGE,
                doc_name     = doc_name,
                doc_hash     = doc_hash,
            )
            validated_chunks.append(validated)
        except Exception as e:
            logger.warning("Chunk %d validation failed — skipping: %s", i, e)

    if not validated_chunks:
        raise ValueError("All chunks failed Pydantic validation")

    logger.info("%d/%d chunks passed validation", len(validated_chunks), total_chunks)

    # ── Step 2: Generate embeddings in batches ────────────────────────────
    client = _get_embeddings_client()
    texts = [c.chunk_text for c in validated_chunks]

    # Batch size 100 — safe for OpenAI rate limits
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = _embed_batch(batch, client)
            all_embeddings.extend(batch_embeddings)
            logger.info(
                "Embedded batch %d/%d (%d chunks)",
                (i // batch_size) + 1,
                (len(texts) + batch_size - 1) // batch_size,
                len(batch),
            )
        except Exception as e:
            logger.error("Embedding batch failed: %s", e)
            raise RuntimeError(f"Embedding failed at batch {i // batch_size}: {e}")

    logger.info("All %d embeddings generated", len(all_embeddings))

    # ── Step 3: Save to PostgreSQL ────────────────────────────────────────
    chunk_records = []
    for validated, embedding_vector in zip(validated_chunks, all_embeddings):
        record = DocumentChunk(
            summary_id      = summary_id,
            doc_hash        = validated.doc_hash,
            doc_name        = validated.doc_name,
            chunk_index     = validated.chunk_index,
            total_chunks    = validated.total_chunks,
            chunk_text      = validated.chunk_text,
            chunk_size      = validated.chunk_size,
            page_number     = validated.page_number,
            source_path     = validated.source_path,
            language        = validated.language,
            embedding       = embedding_vector,
            embedding_model = settings.EMBEDDING_MODEL,
        )
        chunk_records.append(record)

    with get_db_session_context() as session:
        # Bulk insert all chunks at once — much faster than one by one
        session.bulk_save_objects(chunk_records)

        # Update DocumentSummary embedding status
        summary = session.query(DocumentSummary).filter(
            DocumentSummary.id == summary_id
        ).first()

        if summary:
            summary.embedding_status    = "completed"
            summary.embedding_model     = settings.EMBEDDING_MODEL
            summary.embedding_stored_at = datetime.now(timezone.utc)
            avg_size = sum(c.chunk_size for c in validated_chunks) // len(validated_chunks)
            summary.avg_chunk_size      = avg_size
            session.add(summary)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Embeddings stored for '%s' — %d chunks | time=%.2fs",
        doc_name, len(chunk_records), elapsed,
    )

    # Free memory
    del chunk_records
    gc.collect()

    return {
        "status":      "completed",
        "doc_name":    doc_name,
        "chunk_count": len(validated_chunks),
        "elapsed_sec": elapsed,
        "model":       settings.EMBEDDING_MODEL,
    }