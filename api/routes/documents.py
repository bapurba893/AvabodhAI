"""
api/routes/documents.py
-----------------------
All document-related API endpoints.

Endpoints:
    POST   /documents/upload        Upload file + generate summary
    GET    /documents/              List all summaries (paginated)
    GET    /documents/{id}          Get one summary by ID
    PATCH  /documents/{id}          Update document name
    DELETE /documents/{id}          Delete a summary

Pydantic validates:
    - Every request input (file type, update body)
    - Every response output (schema enforced)
    - Every DB operation (via ORM models)
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
    ErrorResponse,
)
from db.database import get_db_session
from db.models import DocumentSummary
from pipeline.loader import load_single_document
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _validate_file_extension(filename: str) -> str:
    """Pydantic-style validation for uploaded file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{ext}' not supported. Allowed: {list(ALLOWED_EXTENSIONS)}"
        )
    return ext


# ── POST /documents/upload ────────────────────────────────────────────────────
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
    db: Session = Depends(get_db_session),
):
    """
    Upload a document and generate an AI-powered structured summary.

    **Flow:**
    1. Validate file type (Pydantic)
    2. Save file temporarily
    3. Load document (LangChain DirectoryLoader)
    4. Split into semantic chunks (SemanticChunker)
    5. Summarise in parallel (OpenAI Map-Reduce)
    6. Validate output (Pydantic)
    7. Save to PostgreSQL
    8. Return structured response (Pydantic)

    **Duplicate detection:** If same file uploaded again (same SHA-256 hash),
    returns cached summary instantly without calling LLM.
    """
    # ── Step 1: Validate file ─────────────────────────────────────────────
    _validate_file_extension(file.filename)

    # Save uploaded file
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("File saved: %s", file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        # ── Step 2: Load ──────────────────────────────────────────────────
        try:
            docs = load_single_document(temp_path)
            file_hash = docs[0].metadata.get("file_hash", "")
            page_count = len(docs)
            logger.info("Loaded %d pages from '%s'", page_count, file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load failed: {e}")

        # ── Dedup check ───────────────────────────────────────────────────
        if file_hash:
            existing = check_duplicate(file_hash)
            if existing:
                logger.info("Duplicate detected for '%s' — returning cached", file.filename)
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
                        "created_at":   str(existing.created_at),
                        "updated_at":   str(existing.updated_at) if existing.updated_at else None,
                        "elapsed_sec":  0.0,
                        "message":      "Duplicate — returning cached summary",
                    }
                )

        # ── Step 3: Split ─────────────────────────────────────────────────
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

        # ── Step 4: Summarise ─────────────────────────────────────────────
        try:
            raw_result = summarise_document(chunks, doc_name=file.filename)
            del chunks
            logger.info("Summarisation complete in %.2fs", raw_result["elapsed_sec"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")

        # ── Step 5: Validate output (Pydantic) ────────────────────────────
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

        # ── Step 6: Save to PostgreSQL ────────────────────────────────────
        try:
            record = save_summary(validated, file_hash, chunk_count)
            logger.info("Saved to PostgreSQL — ID: %s", record.id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database save failed: {e}")

        # ── Step 7: Return validated response ─────────────────────────────
        return DocumentUploadResponse(
            id=record.id,
            doc_name=record.doc_name,
            summary_text=record.summary_text,
            key_topics=record.key_topics,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            source_path=record.source_path,
            language=record.language,
            model_used=record.model_used,
            doc_hash=record.doc_hash,
            created_at=record.created_at,
            updated_at=record.updated_at,
            elapsed_sec=raw_result.get("elapsed_sec"),
        )

    finally:
        # Clean up temp file after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Temp file cleaned up: %s", temp_path)


# ── GET /documents/ ───────────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all document summaries",
)
async def list_documents(
    page:     int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db_session),
):
    """
    List all document summaries with pagination.

    Returns lightweight list (no full summary text) for performance.
    Use GET /documents/{id} to get full summary of one document.
    """
    offset = (page - 1) * per_page

    total = db.query(DocumentSummary).count()
    records = (
        db.query(DocumentSummary)
        .order_by(DocumentSummary.created_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )

    items = []
    for rec in records:
        items.append(DocumentListItem(
            id=rec.id,
            doc_name=rec.doc_name,
            page_count=rec.page_count or 0,
            chunk_count=rec.chunk_count or 0,
            model_used=rec.model_used,
            created_at=rec.created_at,
            summary_preview=rec.summary_text[:200] if rec.summary_text else None,
        ))

    return DocumentListResponse(
        total=total,
        page=page,
        per_page=per_page,
        documents=items,
    )


# ── GET /documents/{id} ───────────────────────────────────────────────────────
@router.get(
    "/{doc_id}",
    response_model=DocumentDetailResponse,
    summary="Get full summary by document ID",
)
async def get_document(
    doc_id: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    """Get the full summary and metadata for one document by its UUID."""
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{doc_id}' not found"
        )

    return DocumentDetailResponse(
        id=record.id,
        doc_name=record.doc_name,
        summary_text=record.summary_text,
        key_topics=record.key_topics,
        page_count=record.page_count or 0,
        chunk_count=record.chunk_count or 0,
        source_path=record.source_path,
        language=record.language,
        model_used=record.model_used,
        doc_hash=record.doc_hash,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


# ── PATCH /documents/{id} ─────────────────────────────────────────────────────
@router.patch(
    "/{doc_id}",
    response_model=DocumentUpdateResponse,
    summary="Update document name",
)
async def update_document(
    doc_id:  uuid.UUID,
    payload: DocumentUpdateRequest,          # Pydantic validates request body
    db:      Session = Depends(get_db_session),
):
    """
    Update the name of a document summary.

    Request body is validated by Pydantic:
    - doc_name must be non-empty string
    - max 512 characters
    - whitespace is stripped automatically
    """
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{doc_id}' not found"
        )

    record.doc_name = payload.doc_name
    db.commit()
    db.refresh(record)

    logger.info("Updated doc_name for ID %s -> '%s'", doc_id, payload.doc_name)

    return DocumentUpdateResponse(
        id=record.id,
        doc_name=record.doc_name,
        updated_at=record.updated_at,
    )


# ── DELETE /documents/{id} ────────────────────────────────────────────────────
@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document summary",
)
async def delete_document(
    doc_id: uuid.UUID,
    db:     Session = Depends(get_db_session),
):
    """Permanently delete a document summary from PostgreSQL."""
    record = db.query(DocumentSummary).filter(DocumentSummary.id == doc_id).first()

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{doc_id}' not found"
        )

    db.delete(record)
    db.commit()

    logger.info("Deleted document ID: %s", doc_id)

    return DeleteResponse(id=doc_id)