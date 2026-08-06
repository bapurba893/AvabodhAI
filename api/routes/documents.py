"""
api/routes/documents.py
-----------------------
All document-related API endpoints.

Multi-tenancy + department isolation: every endpoint takes tenant_id AND
org_unit_id from the X-Tenant-ID / X-Org-Unit-ID headers (via
api.dependencies) and threads both through every pipeline call and every
DB query, together, never either one alone. Nothing in this file reads
or writes document_summaries / document_chunks without both filters
attached.

Preview: no separate preview endpoint — GET /documents/{id} already
returns everything a preview needs (summary, title, category,
org_unit_id, tenant_id, effective dates, ground-truth flag) plus a
`preview_link` field. That link points at a signed, short-lived URL
(GET /documents/{id}/file) since a plain clickable link/iframe can't
carry our auth headers — see utils/signed_link.py.
"""

import os
import shutil
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_tenant_id, get_org_unit_id
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
from pipeline.loader import load_single_document, extract_images_from_pdf
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from pipeline.embedder import store_chunk_embeddings, embed_and_store_images, ImageEmbeddingInput
from utils.signed_link import generate_file_token, verify_file_token
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


def _to_utc_datetime(d: Optional[date]) -> Optional[datetime]:
    """Convert an HTML <input type=date> value to a tz-aware datetime for storage."""
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _build_preview_link(record: DocumentSummary, tenant_id: str, org_unit_id: str) -> Optional[str]:
    """
    Web-scraped documents: source_path IS the original URL — just return it.
    Uploaded documents: build a signed, short-lived link to our own
    file-serving endpoint, since a plain link can't carry auth headers.
    """
    if not record.source_path:
        return None
    if record.source_path.startswith("http://") or record.source_path.startswith("https://"):
        return record.source_path
    if not os.path.exists(record.source_path):
        return None
    token = generate_file_token(str(record.id), tenant_id, org_unit_id)
    return f"/documents/{record.id}/file?token={token}"


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a document and generate summary",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, DOCX, or CSV file"),
    category: Optional[str] = Form(default=None, description="Client-defined category, e.g. 'Policy', 'Guide'"),
    effective_from: Optional[date] = Form(default=None, description="Validity window start — metadata only, no retrieval effect"),
    effective_to: Optional[date] = Form(default=None, description="Validity window end — metadata only, no retrieval effect"),
    is_ground_truth: bool = Form(default=False, description="Only ground-truth documents are ever retrievable by chat/search"),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    ext = _validate_file_extension(file.filename)

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("File saved: %s (tenant=%s org_unit=%s)", file.filename, tenant_id, org_unit_id)
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

    # ── Dedup check — scoped to THIS tenant + org unit only ─────────────────
    if file_hash:
        existing = check_duplicate(file_hash, tenant_id, org_unit_id)
        if existing:
            logger.info("Duplicate detected for '%s' (tenant=%s org_unit=%s)", file.filename, tenant_id, org_unit_id)
            if existing.embedding_status == "pending":
                try:
                    chunks = split_documents(docs)
                    store_chunk_embeddings(
                        chunks=chunks, summary_id=existing.id,
                        doc_hash=file_hash, doc_name=file.filename,
                        tenant_id=tenant_id, org_unit_id=org_unit_id,
                        is_ground_truth=existing.is_ground_truth,
                        source_path=temp_path,
                    )
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
                    "tenant_id":    existing.tenant_id,
                    "org_unit_id":  existing.org_unit_id,
                    "category":     existing.category,
                    "effective_from": str(existing.effective_from) if existing.effective_from else None,
                    "effective_to":   str(existing.effective_to) if existing.effective_to else None,
                    "is_ground_truth": existing.is_ground_truth,
                    "title":        existing.title,
                    "author":       existing.author,
                    "document_type": existing.document_type,
                    "domain":       existing.domain,
                    "key_entities": existing.key_entities,
                    "metadata_status": existing.metadata_status,
                    "image_count":  existing.image_count or 0,
                    "preview_link": _build_preview_link(existing, tenant_id, org_unit_id),
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
            raise HTTPException(status_code=422, detail="Document produced 0 chunks")
        chunk_count = len(chunks)
        logger.info("Split into %d chunks", chunk_count)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {e}")

    # ── Step 3: Summarise ─────────────────────────────────────────────────
    try:
        raw_result = summarise_document(chunks, doc_name=file.filename)
        logger.info("Summarisation complete in %.2fs", raw_result["elapsed_sec"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarisation failed: {e}")

    chunk_metadata    = raw_result.get("chunk_metadata", [])
    document_metadata = raw_result.get("document_metadata")

    # ── Step 4: Validate ──────────────────────────────────────────────────
    try:
        validated = validate_summary(raw_result, {
            "doc_name": file.filename, "source_path": temp_path, "page_count": page_count,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Output validation failed: {e}")

    # ── Step 5: Save summary — tenant/org_unit + client fields stamped ──────
    try:
        record = save_summary(
            validated, file_hash, chunk_count,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            document_metadata=document_metadata,
            category=category,
            effective_from=_to_utc_datetime(effective_from),
            effective_to=_to_utc_datetime(effective_to),
            is_ground_truth=is_ground_truth,
        )
        logger.info("Saved to PostgreSQL — ID: %s (tenant=%s org_unit=%s, ground_truth=%s)",
                   record.id, tenant_id, org_unit_id, is_ground_truth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save failed: {e}")

    # ── Step 6: Store text chunk embeddings ───────────────────────────────
    try:
        store_chunk_embeddings(
            chunks=chunks, summary_id=record.id,
            doc_hash=file_hash, doc_name=file.filename,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            is_ground_truth=is_ground_truth,
            source_path=temp_path, chunk_metadata=chunk_metadata,
        )
        logger.info("Text embeddings stored for '%s'", file.filename)
    except Exception as e:
        logger.warning("Text embedding storage failed (non-fatal): %s", e)

    # ── Step 7: Extract and embed images (PDF only) ────────────────────────
    if ext == ".pdf":
        try:
            image_inputs = extract_images_from_pdf(
                filepath    = temp_path,
                summary_id  = record.id,
                doc_hash    = file_hash,
                doc_name    = file.filename,
                tenant_id   = tenant_id,
                org_unit_id = org_unit_id,
                source_path = temp_path,
            )
            if image_inputs:
                embed_and_store_images(
                    images     = image_inputs,
                    summary_id = record.id,
                    doc_name   = file.filename,
                    tenant_id  = tenant_id,
                    org_unit_id = org_unit_id,
                    is_ground_truth = is_ground_truth,
                )
                logger.info("Image embeddings stored: %d images", len(image_inputs))
            else:
                logger.info("No images found in PDF '%s'", file.filename)
        except Exception as e:
            logger.warning("Image extraction/embedding failed (non-fatal): %s", e)

    # ── Step 8: Return response ───────────────────────────────────────────
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
        tenant_id       = record.tenant_id,
        org_unit_id     = record.org_unit_id,
        category        = record.category,
        effective_from  = record.effective_from,
        effective_to    = record.effective_to,
        is_ground_truth = record.is_ground_truth,
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
        image_count     = record.image_count or 0,
        preview_link    = _build_preview_link(record, tenant_id, org_unit_id),
        created_at  = record.created_at,
        updated_at  = record.updated_at,
        elapsed_sec = raw_result.get("elapsed_sec"),
    )


@router.get("/", response_model=DocumentListResponse, summary="List all document summaries")
async def list_documents(
    page:     int = Query(default=1, ge=1),
    per_page: int = Query(default=10, ge=1, le=100),
    document_type: Optional[str] = Query(default=None),
    domain:        Optional[str] = Query(default=None),
    category:      Optional[str] = Query(default=None, description="Filter by client-defined category"),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    # Every list is scoped to the caller's tenant AND org unit together —
    # this is the query that would leak another tenant's, or another
    # department's, document titles/summaries if either filter were ever
    # dropped.
    query = db.query(DocumentSummary).filter(
        DocumentSummary.tenant_id == tenant_id,
        DocumentSummary.org_unit_id == org_unit_id,
    )
    if document_type:
        query = query.filter(DocumentSummary.document_type == document_type.lower())
    if domain:
        query = query.filter(DocumentSummary.domain == domain.lower())
    if category:
        query = query.filter(DocumentSummary.category == category)

    total  = query.count()
    records = query.order_by(DocumentSummary.created_at.desc()).offset(offset).limit(per_page).all()
    items = [
        DocumentListItem(
            id             = rec.id,
            doc_name       = rec.doc_name,
            page_count     = rec.page_count or 0,
            chunk_count    = rec.chunk_count or 0,
            model_used     = rec.model_used,
            document_type  = rec.document_type,
            domain         = rec.domain,
            category       = rec.category,
            is_ground_truth = rec.is_ground_truth,
            image_count    = rec.image_count or 0,
            created_at     = rec.created_at,
            summary_preview= rec.summary_text[:200] if rec.summary_text else None,
        )
        for rec in records
    ]
    return DocumentListResponse(total=total, page=page, per_page=per_page, documents=items)


@router.get("/{doc_id}", response_model=DocumentDetailResponse, summary="Get full summary by ID (includes preview data)")
async def get_document(
    doc_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    # Filtered by id AND tenant_id AND org_unit_id together — a user
    # guessing or reusing a document UUID from another tenant OR another
    # department gets the same 404 as a nonexistent ID, never a
    # distinguishable "exists but isn't yours" response.
    record = (
        db.query(DocumentSummary)
        .filter(
            DocumentSummary.id == doc_id,
            DocumentSummary.tenant_id == tenant_id,
            DocumentSummary.org_unit_id == org_unit_id,
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentDetailResponse(
        id=record.id, doc_name=record.doc_name, summary_text=record.summary_text,
        key_topics=record.key_topics, page_count=record.page_count or 0,
        chunk_count=record.chunk_count or 0, source_path=record.source_path,
        language=record.language, model_used=record.model_used, doc_hash=record.doc_hash,
        tenant_id=record.tenant_id, org_unit_id=record.org_unit_id,
        category=record.category, effective_from=record.effective_from,
        effective_to=record.effective_to, is_ground_truth=record.is_ground_truth,
        title=record.title, author=record.author,
        document_type=record.document_type, domain=record.domain,
        key_entities=record.key_entities, mentioned_dates=record.mentioned_dates,
        target_audience=record.target_audience, sentiment=record.sentiment,
        confidentiality_level=record.confidentiality_level,
        metadata_status=record.metadata_status,
        image_count=record.image_count or 0,
        preview_link=_build_preview_link(record, tenant_id, org_unit_id),
        created_at=record.created_at, updated_at=record.updated_at,
    )


@router.get("/{doc_id}/file", summary="Stream the original uploaded file (signed link only)")
async def get_document_file(
    doc_id: uuid.UUID,
    token: str = Query(..., description="Signed token from a document's preview_link — not usable on its own"),
    db: Session = Depends(get_db_session_fastapi),
):
    """
    Authorized via a signed token in the URL, NOT via X-Tenant-ID/
    X-Org-Unit-ID headers — this endpoint is meant to be hit as a plain
    clickable link or <iframe src>, and browsers don't attach custom
    headers to those. The token (see utils/signed_link.py) encodes which
    document, which tenant, which org unit, and an expiry — generated
    fresh every time GET /documents/{id} is called, valid for
    settings.FILE_LINK_TTL_SECONDS (15 minutes by default).
    """
    payload = verify_file_token(token)
    if not payload or payload.get("doc_id") != str(doc_id):
        raise HTTPException(status_code=403, detail="Invalid or expired link")

    # This route gets no X-Tenant-ID/X-Org-Unit-ID headers (it's hit as a
    # plain link, not an authenticated API call), so request.state never
    # got populated and get_db_session_fastapi() set no RLS context.
    # With RLS FORCE-enabled on document_summaries, the query below would
    # silently return zero rows without this — set it explicitly from the
    # token's own verified payload instead. Same safe parameterized
    # set_config() pattern as db.database._apply_rls_context().
    db.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": payload["tenant_id"]})
    db.execute(text("SELECT set_config('app.org_unit_id', :v, true)"), {"v": payload["org_unit_id"]})

    record = (
        db.query(DocumentSummary)
        .filter(
            DocumentSummary.id == doc_id,
            DocumentSummary.tenant_id == payload["tenant_id"],
            DocumentSummary.org_unit_id == payload["org_unit_id"],
        )
        .first()
    )
    if not record or not record.source_path:
        raise HTTPException(status_code=404, detail="File not found")
    if record.source_path.startswith("http://") or record.source_path.startswith("https://"):
        raise HTTPException(status_code=400, detail="This document has no local file — it was web-scraped; use its source URL directly")
    if not os.path.exists(record.source_path):
        raise HTTPException(status_code=404, detail="File no longer exists on disk")

    return FileResponse(record.source_path, filename=record.doc_name)


@router.patch("/{doc_id}", response_model=DocumentUpdateResponse, summary="Update document name")
async def update_document(
    doc_id: uuid.UUID, payload: DocumentUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = (
        db.query(DocumentSummary)
        .filter(
            DocumentSummary.id == doc_id,
            DocumentSummary.tenant_id == tenant_id,
            DocumentSummary.org_unit_id == org_unit_id,
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    record.doc_name = payload.doc_name
    db.commit()
    db.refresh(record)
    return DocumentUpdateResponse(id=record.id, doc_name=record.doc_name, updated_at=record.updated_at)


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document summary")
async def delete_document(
    doc_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    record = (
        db.query(DocumentSummary)
        .filter(
            DocumentSummary.id == doc_id,
            DocumentSummary.tenant_id == tenant_id,
            DocumentSummary.org_unit_id == org_unit_id,
        )
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    # document_chunks cascade-deletes automatically via the DB-level
    # ON DELETE CASCADE on DocumentChunk.summary_id — no orphaned vectors.
    db.delete(record)
    db.commit()
    return DeleteResponse(id=doc_id)