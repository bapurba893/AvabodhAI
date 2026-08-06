"""
pipeline/storage.py
-------------------
Steps 5 + 6 + 7 — Pydantic validation → Structured object → Save to PostgreSQL

Also saves document-level metadata (title, author, document_type, domain,
key_entities, mentioned_dates, target_audience, sentiment, confidentiality_level)
extracted by summariser.py, into the same document_summaries row.

Also handles:
- Deduplication: if this exact file (same SHA-256 hash) was already summarised,
  skip the LLM entirely and return the cached summary — saves cost
- Upsert logic: if doc_name exists but hash changed (file updated), update the row

Both dedup lookups are scoped by (tenant_id, org_unit_id) together — never
either one alone. org_unit_id is a hard boundary WITHIN a tenant (department-
level), so a standalone org_unit_id match would leak across companies if two
different tenants happen to use the same department code.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import make_transient

from db.database import get_db_session_context
from db.models import DocumentSummary, DocumentSummaryOutput, DocumentMetadataOutput
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def validate_summary(raw: dict, metadata: dict) -> DocumentSummaryOutput:
    """
    Step 5 + 6 — Feed raw LLM output + document metadata into Pydantic.
    Pydantic validates, cleans, and fills defaults for any missing fields.
    Returns a structured DocumentSummaryOutput object.
    """
    try:
        output = DocumentSummaryOutput(
            doc_name       = metadata.get("doc_name", "unknown"),
            summary_text   = raw["summary_text"],
            key_topics     = metadata.get("key_topics", []),
            page_count     = metadata.get("page_count", 0),
            chunk_count    = raw.get("chunk_count", 0),
            source_path    = metadata.get("source_path", ""),
            language       = settings.SUMMARY_LANGUAGE,
            model_used     = raw.get("reduce_model", settings.REDUCE_MODEL),
        )
        logger.info("Pydantic validation passed for '%s'", output.doc_name)
        return output

    except Exception as e:
        logger.error("Pydantic validation failed: %s | raw=%s", e, raw)
        raise ValueError(f"Summary output failed validation: {e}") from e


def check_duplicate(file_hash: str, tenant_id: str, org_unit_id: str) -> Optional[DocumentSummary]:
    """
    Check if this exact file was already summarised — scoped to this
    tenant AND org unit together. Two tenants (or two departments within
    the same tenant) uploading the same publicly available file (a vendor
    spec sheet, a public report) must not see each other's cached summary.
    Returns the existing DB record if found, else None.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        existing = (
            session.query(DocumentSummary)
            .filter(
                DocumentSummary.doc_hash == file_hash,
                DocumentSummary.tenant_id == tenant_id,
                DocumentSummary.org_unit_id == org_unit_id,
            )
            .first()
        )
        if existing:
            logger.info(
                "Duplicate detected — '%s' already summarised for tenant=%s org_unit=%s (hash=%s). Skipping LLM.",
                existing.doc_name, tenant_id, org_unit_id, file_hash[:12],
            )
            session.expunge(existing)
            make_transient(existing)
            return existing
        return None


def _apply_document_metadata(record: DocumentSummary, doc_metadata: Optional[DocumentMetadataOutput]) -> None:
    """
    Apply extracted document-level metadata onto a DocumentSummary ORM
    instance. Shared by both insert and update paths.
    Sets metadata_status='completed' on success, 'failed' if extraction
    was unavailable (non-fatal — summary still saves either way).
    """
    if doc_metadata is None:
        record.metadata_status = "failed"
        return

    record.title                  = doc_metadata.title or None
    record.author                 = doc_metadata.author
    record.document_type          = doc_metadata.document_type
    record.domain                 = doc_metadata.domain
    record.detected_language      = doc_metadata.detected_language
    record.key_entities           = doc_metadata.key_entities
    record.mentioned_dates        = doc_metadata.mentioned_dates
    record.target_audience        = doc_metadata.target_audience
    record.sentiment              = doc_metadata.sentiment
    record.confidentiality_level  = doc_metadata.confidentiality_level
    record.metadata_status        = "completed"
    record.metadata_extracted_at  = datetime.now(timezone.utc)


def save_summary(
    validated: DocumentSummaryOutput,
    file_hash: str,
    chunk_count: int,
    tenant_id: str,
    org_unit_id: str,
    document_metadata: Optional[DocumentMetadataOutput] = None,
    category: Optional[str] = None,
    effective_from: Optional[datetime] = None,
    effective_to: Optional[datetime] = None,
    is_ground_truth: bool = False,
) -> DocumentSummary:
    """
    Step 7 — Save the structured summary to PostgreSQL.
    - If same tenant+org_unit + hash exists → skip (already saved)
    - If same tenant+org_unit + doc_name exists but different hash → update
    - Otherwise → insert new row

    tenant_id + org_unit_id together scope BOTH lookups below — a
    standalone tenant_id match isn't enough now that departments within
    one tenant are also isolated from each other; without org_unit_id
    too, Department A's "invoice.pdf" could silently overwrite
    Department B's via the doc_name upsert path.

    title is intentionally NOT a parameter here — it stays LLM-generated
    only, applied via _apply_document_metadata(), never user-supplied.

    Returns the saved/updated ORM record.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:

        # Check if this exact hash already in DB for this tenant+org_unit
        existing = (
            session.query(DocumentSummary)
            .filter(
                DocumentSummary.doc_hash == file_hash,
                DocumentSummary.tenant_id == tenant_id,
                DocumentSummary.org_unit_id == org_unit_id,
            )
            .first()
        )
        if existing:
            logger.info(
                "Record already exists for tenant=%s org_unit=%s hash=%s — no DB write needed.",
                tenant_id, org_unit_id, file_hash[:12],
            )
            session.expunge(existing)
            make_transient(existing)
            return existing

        # Check if same filename exists for this tenant+org_unit with a
        # different hash (file updated) — scoped by BOTH so Department A's
        # "invoice.pdf" can never match Department B's, even within the
        # same tenant.
        same_name = (
            session.query(DocumentSummary)
            .filter(
                DocumentSummary.doc_name == validated.doc_name,
                DocumentSummary.tenant_id == tenant_id,
                DocumentSummary.org_unit_id == org_unit_id,
            )
            .first()
        )

        if same_name:
            # Update existing record — file content changed
            same_name.summary_text = validated.summary_text
            same_name.key_topics   = ", ".join(validated.key_topics)
            same_name.chunk_count  = chunk_count
            same_name.doc_hash     = file_hash
            same_name.model_used   = validated.model_used
            same_name.language     = validated.language
            same_name.updated_at   = datetime.now(timezone.utc)
            same_name.embedding_status = "pending"
            same_name.embedding_model  = None
            same_name.category         = category
            same_name.effective_from   = effective_from
            same_name.effective_to     = effective_to
            same_name.is_ground_truth  = is_ground_truth
            _apply_document_metadata(same_name, document_metadata)
            session.add(same_name)
            session.flush()
            logger.info(
                "Updated existing summary for tenant=%s org_unit=%s '%s' (file changed)",
                tenant_id, org_unit_id, validated.doc_name,
            )
            session.expunge(same_name)
            make_transient(same_name)
            return same_name

        # New document — insert fresh row
        record = DocumentSummary(
            tenant_id         = tenant_id,
            org_unit_id       = org_unit_id,
            doc_name          = validated.doc_name,
            summary_text      = validated.summary_text,
            key_topics        = ", ".join(validated.key_topics),
            page_count        = validated.page_count,
            chunk_count       = chunk_count,
            source_path       = validated.source_path,
            language          = validated.language,
            model_used        = validated.model_used,
            confidence        = validated.confidence_score,
            doc_hash          = file_hash,
            embedding_status  = "pending",      # will be updated to 'completed' by embedder
            embedding_model   = None,           # filled by embedder after storing
            avg_chunk_size    = None,           # filled by embedder
            embedding_stored_at = None,         # filled by embedder
            category          = category,
            effective_from    = effective_from,
            effective_to      = effective_to,
            is_ground_truth   = is_ground_truth,
        )
        _apply_document_metadata(record, document_metadata)
        session.add(record)
        session.flush()  # get the ID before session closes
        logger.info(
            "Saved new summary for tenant=%s org_unit=%s '%s' to PostgreSQL (metadata_status=%s, ground_truth=%s)",
            tenant_id, org_unit_id, validated.doc_name, record.metadata_status, is_ground_truth,
        )
        session.expunge(record)  # detach from session so it can be used outside
        make_transient(record)
        return record