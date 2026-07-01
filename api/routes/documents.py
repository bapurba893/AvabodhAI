"""
api/routes/documents.py
-----------------------
All document-related API endpoints.

CHANGES FOR METADATA EXTRACTION:
- raw_result now contains "chunk_metadata" and "document_metadata"
- save_summary() now receives document_metadata
- store_chunk_embeddings() now receives chunk_metadata
- Upload response includes new metadata fields
"""

import os
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from api.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse,
    DocumentUpdateRequest,
    DocumentUpdateResponse,
    DeleteResponse,
)
from db.database import get_db_session_fastapi
from db.models import DocumentSummary
from pipeline.loader import load_single_document
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from pipeline.embedder import store_chunk_embeddings
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _validate_file_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{ext}' not supported. Allowed: {list(ALLOWED_EXTENSIONS)}"
        )
    return ext


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a document and generate summary",
    responses={
        201: {"description": "Summary created successfully"},
        200: {"description": "Duplicate detected — returning cached summary"},
        422: {"description": "Invalid file type or validation error"},
        500: {"description": "Processing failed"},
    }
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, DOCX, or CSV file"),
    db: Session = Depends(get_db_session_fastapi),
):
    _validate_file_extension(file.filename)

    # Save file permanently — needed for backfill later
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("File saved: %s", file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    try:
        docs = load_single_document(temp_path)
        file_hash  = docs[0].metadata.get("file_hash", "")
        page_count = len(docs)
        logger.info("Loaded %d pages from '%s'", page_count, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")

    # ── Dedup check ───────────────────────────────────────────────────────
    if file_hash:
        existing = check_duplicate(file_hash)
        if existing:
            logger.info("Duplicate detected for '%s' — checking embeddings", file.filename)

            # If embeddings are pending — generate them now using the re-uploaded file
            if existing.embedding_status == "pending":
                try:
                    chunks = split_documents(docs)
                    # NOTE: re-embedding an existing doc skips metadata extraction
                    # (no LLM call here) — embeddings only. Re-upload triggers
                    # full pipeline only for genuinely new/changed documents.
                    store_chunk_embeddings(
                        chunks      = chunks,
                        summary_id  = existing.id,
                        doc_hash    = file_hash,
                        doc_name    = file.filename,
                        source_path = temp_path,
                    )
                    logger.info("Embeddings generated for existing doc '%s'", file.filename)
                except Exception as e:
                    logger.warning("Embedding failed for existing doc: %s", e)

            return JSONResponse(
                status_code=200,
                content={
                    "id":           str(existing.id),
                    "doc_name":     existing.doc_name,
                    "summary_text": existing.summary_text,
                    "key_topics":   existing.key_topics,
                    "page_count":   existing.page_count,
                    "chunk_count":  existing.chunk_count,
                    "source_path":  existing.source_path,
                    "language":     existing.language,
                    "model_used":   existing.model_used,
                    "doc_hash":     existing.doc_hash,
                    "title":        existing.title,
                    "author":       existing.author,
                    "document_type": existing.document_type,
                    "domain":       existing.domain,
                    "key_entities": existing.key_entities,
                    "metadata_status": existing.metadata_status,
                    "created_at":   str(existing.created_at),
                    "updated_at":   str(existing.updated_at) if existing.updated_at else None,
                    "elapsed_sec":  0.0,
                    "message":      "Duplicate — returning cached summary",
                }
            )

    # ── Step 2: Split ─────────────────────────────────────────────────────
    try:
        chunks = split_documents(docs)
        del docs
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Document produced 0 chunks — file may be empty or unsupported format"
            )
        chunk_count = len(chunks)
        logger.info("Split into %d chunks", chunk_count)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {e}")

    # ── Step 3: Summarise + extract metadata ────────────────────────────────
    try:
        raw_result = summarise_document(chunks, doc_name=file.filename)
        logger.info("Summarisation complete in %.2fs", raw_result["elapsed_sec"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")

    # Extracted metadata from the summariser result
    chunk_metadata     = raw_result.get("chunk_metadata", [])
    document_metadata  = raw_result.get("document_metadata")

    # ── Step 4: Validate output (Pydantic) ────────────────────────────────
    try:
        metadata = {
            "doc_name":    file.filename,
            "source_path": temp_path,
            "page_count":  page_count,
        }
        validated = validate_summary(raw_result, metadata)
        logger.info("Pydantic validation passed for '%s'", file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Output validation failed: {e}")

    # ── Step 5: Save summary + document metadata to PostgreSQL ─────────────
    try:
        record = save_summary(
            validated,
            file_hash,
            chunk_count,
            document_metadata=document_metadata,
        )
        logger.info("Saved to PostgreSQL — ID: %s", record.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save failed: {e}")

    # ── Step 6: Store chunk embeddings + chunk metadata in pgvector ────────
    try:
        store_chunk_embeddings(
            chunks         = chunks,
            summary_id     = record.id,
            doc_hash       = file_hash,
            doc_name       = file.filename,
            source_path    = temp_path,
            chunk_metadata = chunk_metadata,
        )
        logger.info("Embeddings + chunk metadata stored for '%s'", file.filename)
    except Exception as e:
        # Embedding failure does NOT fail the whole request
        # Summary is already saved — embeddings can be backfilled later
        logger.warning("Embedding storage failed (non-fatal): %s", e)

    # ── Step 7: Return response ───────────────────────────────────────────
    return DocumentUploadResponse(
        id          = record.id,
        doc_name    = record.doc_name,
        summary_text= record.summary_text,
        key_topics  = record.key_topics,
        page_count  = record.page_count,
        chunk_count = record.chunk_count,
        source_path = record.source_path,
        language    = record.language,
        model_used  = record.model_used,
        doc_hash    = record.doc_hash,
        title           = record.title,
        author          = record.author,
        document_type   = record.document_type,
        domain          = record.domain,
        key_entities    = record.key_entities,
        mentioned_dates = record.mentioned_dates,
        target_audience = record.target_audience,
        sentiment       = record.sentiment,
        confidentiality_level = record.confidentiality_level,
        metadata_status = record.metadata_status,
        created_at  = record.created_at,
        updated_at  = record.updated_at,
        elapsed_sec = raw_result.get("elapsed_sec"),
    )


@router.get("/", response_model=DocumentListResponse, summary="List all document summaries")
async def list_documents(
    page:     int = Query(default=1, ge=1),
    per_page: int = Query(default=10, ge=1, le=100),
    document_type: Optional[str] = Query(default=None, description="Filter by document type"),
    domain:        Optional[str] = Query(default=None, description="Filter by domain"),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    query = db.query(DocumentSummary)

    # NEW — optional filtering by extracted metadata
    if document_type:
        query = query.filter(DocumentSummary.document_type == document_type.lower())
    if domain:
        query = query.filter(DocumentSummary.domain == domain.lower())

    total  = query.count()
    records = (
        query
        .order_by(DocumentSummary.created_at.desc())
        .offset(offset).limit(per_page).all()
    )
    items = [
        DocumentListItem(
            id             = rec.id,
            doc_name       = rec.doc_name,
            page_count     = rec.page_count or 0,
            chunk_count    = rec.chunk_count or 0,
            model_used     = rec.model_used,
            document_type  = rec.document_type,
            domain         = rec.domain,
            created_at     = rec.created_at,
            summary_preview= rec.summary_text[:200] if rec.summary_text else None,
        )
        for rec in records
    ]
    return DocumentListResponse(total=total, page=page, per_page=per_page, documents=items)


@router.get("/{doc_id}", response_model=DocumentDetailResponse, summary="Get full summary by ID")
async def get_document(doc_id: uuid.UUID, db: Session = Depends(get_db_session_fastapi)):
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentDetailResponse(
        id=record.id, doc_name=record.doc_name, summary_text=record.summary_text,
        key_topics=record.key_topics, page_count=record.page_count or 0,
        chunk_count=record.chunk_count or 0, source_path=record.source_path,
        language=record.language, model_used=record.model_used,
        doc_hash=record.doc_hash,
        title=record.title, author=record.author,
        document_type=record.document_type, domain=record.domain,
        key_entities=record.key_entities, mentioned_dates=record.mentioned_dates,
        target_audience=record.target_audience, sentiment=record.sentiment,
        confidentiality_level=record.confidentiality_level,
        metadata_status=record.metadata_status,
        created_at=record.created_at, updated_at=record.updated_at,
    )


@router.patch("/{doc_id}", response_model=DocumentUpdateResponse, summary="Update document name")
async def update_document(
    doc_id: uuid.UUID, payload: DocumentUpdateRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    record.doc_name = payload.doc_name
    db.commit()
    db.refresh(record)
    return DocumentUpdateResponse(id=record.id, doc_name=record.doc_name, updated_at=record.updated_at)


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document summary")
async def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db_session_fastapi)):
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    db.delete(record)
    db.commit()
    return DeleteResponse(id=doc_id)