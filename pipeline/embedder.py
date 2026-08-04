"""
pipeline/embedder.py
--------------------
Generate and store chunk embeddings in pgvector.

Two entry points:
  store_chunk_embeddings()   — text chunks (existing, unchanged logic)
  embed_and_store_images()   — image chunks (NEW — called after image_processor)
"""

import gc
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator

from db.database import get_db_session_context
from db.models import DocumentChunk, DocumentSummary, ChunkMetadataOutput
from pipeline.image_processor import ImageCaptionOutput, build_image_embedding_text
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema — validates each chunk before saving to DB
# ─────────────────────────────────────────────────────────────────────────────

class ChunkEmbeddingInput(BaseModel):
    chunk_index:  int            = Field(ge=0)
    total_chunks: int            = Field(ge=1)
    chunk_text:   str            = Field(min_length=1)
    chunk_size:   int            = Field(ge=0)
    page_number:  Optional[int]  = Field(default=None)
    source_path:  Optional[str]  = Field(default=None)
    language:     str            = Field(default="English")
    doc_name:     str            = Field(min_length=1)
    doc_hash:     str            = Field(min_length=1)

    @field_validator("chunk_text", mode="before")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
    )


def _embed_batch(texts: list[str], client: OpenAIEmbeddings) -> list[list[float]]:
    return client.embed_documents(texts)


def check_embeddings_exist(doc_hash: str, tenant_id: str) -> bool:
    """
    True if TEXT chunks already exist for this doc_hash, within this
    tenant. Filtered to role='text' so a document that (somehow) only
    has image chunks stored doesn't cause text re-embedding to be
    skipped, and to tenant_id so it can't be tripped by another
    tenant's identical content.
    """
    with get_db_session_context() as session:
        count = (
            session.query(DocumentChunk)
            .filter(
                DocumentChunk.doc_hash == doc_hash,
                DocumentChunk.role == "text",
                DocumentChunk.tenant_id == tenant_id,
            )
            .count()
        )
        return count > 0


def check_image_hash_exists(image_bytes_hash: str, tenant_id: str) -> bool:
    """
    Check if this exact image is already embedded for this tenant — dedup.
    Scoped by tenant_id: if Tenant A and Tenant B both have a document
    containing the same stock image or shared logo, Tenant B's upload
    must NOT be skipped just because Tenant A already has that hash —
    that would leave Tenant B's own document missing the image chunk.
    """
    with get_db_session_context() as session:
        count = (
            session.query(DocumentChunk)
            .filter(
                DocumentChunk.image_bytes_hash == image_bytes_hash,
                DocumentChunk.role == "image",
                DocumentChunk.tenant_id == tenant_id,
            )
            .count()
        )
        return count > 0


# ─────────────────────────────────────────────────────────────────────────────
# Text chunk embeddings (UNCHANGED logic from original)
# ─────────────────────────────────────────────────────────────────────────────

def store_chunk_embeddings(
    chunks: list[Document],
    summary_id: UUID,
    doc_hash: str,
    doc_name: str,
    tenant_id: str,
    source_path: str = "",
    chunk_metadata: Optional[list[Optional[ChunkMetadataOutput]]] = None,
) -> dict:
    """
    Generate and store text chunk embeddings.
    tenant_id is stamped onto every DocumentChunk row (denormalized from
    the parent document) so vector_search() can filter directly without
    a JOIN. Image chunks go through embed_and_store_images().
    """
    if not chunks:
        raise ValueError(f"No chunks provided for '{doc_name}'")

    if check_embeddings_exist(doc_hash, tenant_id):
        logger.info("Embeddings already exist for tenant=%s '%s' — skipping", tenant_id, doc_name)
        return {"status": "skipped", "doc_name": doc_name, "chunk_count": 0, "elapsed_sec": 0.0}

    logger.info(
        "Starting embedding for '%s' — %d chunks | model=%s",
        doc_name, len(chunks), settings.EMBEDDING_MODEL,
    )

    start = time.time()
    total_chunks = len(chunks)

    meta_list: list[Optional[ChunkMetadataOutput]] = []
    if chunk_metadata:
        meta_list = list(chunk_metadata)
    while len(meta_list) < total_chunks:
        meta_list.append(None)

    # ── Pydantic validation ───────────────────────────────────────────────────
    validated_chunks: list[ChunkEmbeddingInput] = []
    validated_meta:   list[Optional[ChunkMetadataOutput]] = []

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
            validated_meta.append(meta_list[i])
        except Exception as e:
            logger.warning("Chunk %d validation failed — skipping: %s", i, e)

    if not validated_chunks:
        raise ValueError("All chunks failed Pydantic validation")

    # ── Embed in batches ──────────────────────────────────────────────────────
    client = _get_embeddings_client()
    texts = [c.chunk_text for c in validated_chunks]
    batch_size = 100
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            all_embeddings.extend(_embed_batch(batch, client))
            logger.info("Embedded batch %d/%d", (i // batch_size) + 1, total_batches)
        except Exception as e:
            raise RuntimeError(f"Embedding failed at batch {(i // batch_size) + 1}: {e}")

    # ── Build records ─────────────────────────────────────────────────────────
    chunk_records: list[DocumentChunk] = []

    for validated, embedding_vector, meta in zip(validated_chunks, all_embeddings, validated_meta):
        record = DocumentChunk(
            tenant_id           = tenant_id,
            summary_id          = summary_id,
            doc_hash            = validated.doc_hash,
            doc_name            = validated.doc_name,
            chunk_index         = validated.chunk_index,
            total_chunks        = validated.total_chunks,
            chunk_text          = validated.chunk_text,
            chunk_size          = validated.chunk_size,
            page_number         = validated.page_number,
            source_path         = validated.source_path,
            language            = validated.language,
            embedding           = embedding_vector,
            embedding_model     = settings.EMBEDDING_MODEL,
            role                = "text",
            section_heading     = meta.section_heading   if meta else None,
            chunk_type          = meta.chunk_type        if meta else "paragraph",
            topic               = meta.topic             if meta else None,
            entities            = meta.entities          if meta else None,
            contains_data       = meta.contains_data     if meta else False,
            metadata_confidence = meta.confidence_score  if meta else None,
        )
        chunk_records.append(record)

    # ── Bulk insert + update summary ──────────────────────────────────────────
    with get_db_session_context() as session:
        session.bulk_save_objects(chunk_records)

        summary = session.query(DocumentSummary).filter(
            DocumentSummary.id == summary_id,
            DocumentSummary.tenant_id == tenant_id,
        ).first()
        if summary:
            summary.embedding_status    = "completed"
            summary.embedding_model     = settings.EMBEDDING_MODEL
            summary.embedding_stored_at = datetime.now(timezone.utc)
            summary.avg_chunk_size = sum(c.chunk_size for c in validated_chunks) // len(validated_chunks)
            session.add(summary)

    elapsed = round(time.time() - start, 2)
    logger.info("Text embeddings stored for '%s' — %d chunks | %.2fs", doc_name, len(chunk_records), elapsed)

    del chunk_records
    gc.collect()

    return {
        "status":      "completed",
        "doc_name":    doc_name,
        "chunk_count": len(validated_chunks),
        "elapsed_sec": elapsed,
        "model":       settings.EMBEDDING_MODEL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image embeddings (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class ImageEmbeddingInput(BaseModel):
    """One image ready to be embedded and stored."""
    summary_id:        UUID
    doc_hash:          str
    doc_name:          str
    source_path:       str        = ""
    page_number:       Optional[int] = None
    image_bytes_hash:  str
    image_url:         Optional[str] = None
    image_format:      str        = "jpeg"
    image_width:       int        = 0
    image_height:      int        = 0
    image_size_bytes:  int        = 0
    image_caption:     str
    image_type:        str
    image_alt_text:    str        = ""
    image_context:     str        = ""
    contains_chart:    bool       = False
    contains_table:    bool       = False
    contains_text_img: bool       = False
    key_elements:      list[str]  = Field(default_factory=list)
    vision_model_used: str        = "gpt-4o"
    vision_confidence: float      = 0.8
    embedding_text:    str        = ""   # composite text fed to embedding model
    chunk_index:       int        = 0
    total_chunks:      int        = 1


def embed_and_store_images(
    images: list[ImageEmbeddingInput],
    summary_id: UUID,
    doc_name:   str,
    tenant_id:  str,
) -> dict:
    """
    Embed image caption composite texts and store as DocumentChunk rows
    with role='image'. tenant_id is stamped onto every record and used
    to scope the dedup check — see check_image_hash_exists() docstring
    for why this must be per-tenant, not global.

    Called after image_processor.py has already:
    - filtered noise
    - called GPT-4o Vision
    - built ImageCaptionOutput

    This function handles only embedding + DB storage.
    """
    if not images:
        logger.info("No images to embed for '%s'", doc_name)
        return {"status": "skipped", "doc_name": doc_name, "image_count": 0}

    start = time.time()
    client = _get_embeddings_client()

    # Filter out already-embedded images (dedup by image_bytes_hash, per tenant)
    new_images = []
    for img in images:
        if check_image_hash_exists(img.image_bytes_hash, tenant_id):
            logger.info("Image already embedded for tenant=%s (hash=%s) — skipping",
                        tenant_id, img.image_bytes_hash[:12])
        else:
            new_images.append(img)

    if not new_images:
        return {"status": "all_duplicate", "doc_name": doc_name, "image_count": 0}

    # Embed composite caption texts in batches
    texts = [img.embedding_text for img in new_images]
    batch_size = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            all_embeddings.extend(_embed_batch(batch, client))
        except Exception as e:
            logger.error("Image embedding batch failed: %s", e)
            raise RuntimeError(f"Image embedding failed: {e}")

    # Build DocumentChunk records with role='image'
    image_records: list[DocumentChunk] = []

    for img, embedding_vector in zip(new_images, all_embeddings):
        record = DocumentChunk(
            tenant_id          = tenant_id,
            summary_id         = img.summary_id,
            doc_hash           = img.doc_hash,
            doc_name           = img.doc_name,
            chunk_index        = img.chunk_index,
            total_chunks       = img.total_chunks,
            chunk_text         = img.embedding_text,   # composite caption text
            chunk_size         = len(img.embedding_text),
            page_number        = img.page_number,
            source_path        = img.source_path,
            language           = "English",
            embedding          = embedding_vector,
            embedding_model    = settings.EMBEDDING_MODEL,
            # ── Image-specific fields ──────────────────────────────────────
            role               = "image",
            image_url          = img.image_url,
            image_bytes_hash   = img.image_bytes_hash,
            image_format       = img.image_format,
            image_width        = img.image_width,
            image_height       = img.image_height,
            image_size_bytes   = img.image_size_bytes,
            image_caption      = img.image_caption,
            image_type         = img.image_type,
            image_alt_text     = img.image_alt_text,
            image_context      = img.image_context,
            contains_chart     = img.contains_chart,
            contains_table     = img.contains_table,
            contains_text_img  = img.contains_text_img,
            key_elements       = img.key_elements,
            vision_model_used  = img.vision_model_used,
            vision_confidence  = img.vision_confidence,
        )
        image_records.append(record)

    # Bulk insert + update image_count on summary
    with get_db_session_context() as session:
        session.bulk_save_objects(image_records)

        summary = session.query(DocumentSummary).filter(
            DocumentSummary.id == summary_id,
            DocumentSummary.tenant_id == tenant_id,
        ).first()
        if summary:
            summary.image_count = (summary.image_count or 0) + len(image_records)
            session.add(summary)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Image embeddings stored for '%s' — %d images | %.2fs",
        doc_name, len(image_records), elapsed,
    )

    del image_records
    gc.collect()

    return {
        "status":      "completed",
        "doc_name":    doc_name,
        "image_count": len(new_images),
        "elapsed_sec": elapsed,
        "model":       settings.EMBEDDING_MODEL,
    }