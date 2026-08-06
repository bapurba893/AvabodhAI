"""
pipeline/orchestrator.py
------------------------
Main pipeline orchestrator — wires all steps including embedding storage.

Full pipeline:
Step 1 — Load document
Step 2 — Split into chunks
Step 3+4 — Map-Reduce summarisation (OpenAI)
Step 5 — Pydantic validation
Step 6 — Save summary to PostgreSQL
Step 7 — Generate and store chunk embeddings in pgvector
"""

import os
import gc
from dataclasses import dataclass, field
from typing import Optional

from pipeline.loader import load_single_document, load_directory
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from pipeline.embedder import store_chunk_embeddings
from db.database import init_db
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ProcessingResult:
    filepath:        str
    doc_name:        str
    success:         bool
    summary_id:      Optional[str] = None
    chunk_count:     int = 0
    elapsed_sec:     float = 0.0
    skipped:         bool = False
    embedding_status: str = "pending"
    error:           Optional[str] = None


@dataclass
class BatchReport:
    total:     int = 0
    succeeded: int = 0
    skipped:   int = 0
    failed:    int = 0
    results:   list[ProcessingResult] = field(default_factory=list)

    def log(self) -> None:
        logger.info(
            "Batch complete — total=%d | success=%d | skipped=%d | failed=%d",
            self.total, self.succeeded, self.skipped, self.failed,
        )


def process_single_document(
    filepath: str,
    tenant_id: str,
    org_unit_id: str,
    is_ground_truth: bool = False,
) -> ProcessingResult:
    """
    Full 7-step pipeline for one document.
    Step 7 (embeddings) runs after summary is saved.

    tenant_id + org_unit_id: this is a standalone CLI/batch tool (not
    behind the API gateway), so there's no headers to read them from —
    the caller (e.g. process_directory, or a script) must supply both
    explicitly. Documents bulk-loaded this way are attributed exactly
    like an upload through the API would be.

    is_ground_truth defaults to False, matching the API upload default —
    bulk-loaded documents don't become retrievable until explicitly
    marked ground truth (via a future admin action, or by passing
    is_ground_truth=True here directly if the operator already knows
    that's appropriate for this batch).
    """
    doc_name = os.path.basename(filepath)
    logger.info("--- Processing: %s (tenant=%s org_unit=%s) ---", doc_name, tenant_id, org_unit_id)

    docs = None
    chunks = None
    chunks_for_embedding = None

    try:
        # ── Step 1: Load ──────────────────────────────────────────────────
        docs = load_single_document(filepath)
        if not docs:
            raise ValueError("Loader returned no documents")

        file_hash   = docs[0].metadata.get("file_hash", "")
        source_path = docs[0].metadata.get("source_path", filepath)
        page_count  = len(docs)

        # ── Dedup check — scoped to this tenant + org unit ──────────────────
        if file_hash:
            existing = check_duplicate(file_hash, tenant_id, org_unit_id)
            if existing:
                return ProcessingResult(
                    filepath=filepath, doc_name=doc_name,
                    success=True, skipped=True,
                    summary_id=str(existing.id),
                    embedding_status=existing.embedding_status or "unknown",
                )

        # ── Step 2: Split ─────────────────────────────────────────────────
        chunks = split_documents(docs)
        del docs
        docs = None
        gc.collect()

        if not chunks:
            raise ValueError("Splitting produced no chunks")

        chunk_count = len(chunks)

        # Keep a copy for embedding step
        chunks_for_embedding = chunks.copy()

        # ── Steps 3+4: Summarise ──────────────────────────────────────────
        raw_result = summarise_document(chunks, doc_name=doc_name)

        del chunks
        chunks = None
        gc.collect()

        # ── Steps 5+6: Validate + Save summary ───────────────────────────
        metadata = {
            "doc_name":    doc_name,
            "source_path": source_path,
            "page_count":  page_count,
        }
        validated = validate_summary(raw_result, metadata)
        record    = save_summary(
            validated, file_hash, chunk_count,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            is_ground_truth=is_ground_truth,
        )

        logger.info("Summary saved for '%s' | ID: %s", doc_name, record.id)

        # ── Step 7: Store chunk embeddings in pgvector ────────────────────
        embedding_result = store_chunk_embeddings(
            chunks      = chunks_for_embedding,
            summary_id  = record.id,
            doc_hash    = file_hash,
            doc_name    = doc_name,
            tenant_id   = tenant_id,
            org_unit_id = org_unit_id,
            is_ground_truth = is_ground_truth,
            source_path = source_path,
        )

        del chunks_for_embedding
        chunks_for_embedding = None
        gc.collect()

        logger.info(
            "Done: '%s' | chunks=%d | embedding=%s | %.2fs",
            doc_name, chunk_count,
            embedding_result["status"],
            raw_result["elapsed_sec"],
        )

        return ProcessingResult(
            filepath         = filepath,
            doc_name         = doc_name,
            success          = True,
            summary_id       = str(record.id),
            chunk_count      = chunk_count,
            elapsed_sec      = raw_result["elapsed_sec"],
            embedding_status = embedding_result["status"],
        )

    except Exception as e:
        for obj in [docs, chunks, chunks_for_embedding]:
            if obj is not None:
                del obj
        gc.collect()

        logger.error("FAILED: '%s' - %s", doc_name, e, exc_info=True)
        return ProcessingResult(
            filepath=filepath, doc_name=doc_name,
            success=False, error=str(e),
        )


def process_directory(
    tenant_id: str,
    org_unit_id: str,
    directory: str = None,
    is_ground_truth: bool = False,
) -> BatchReport:
    """
    Process all documents in a directory, attributed to tenant_id +
    org_unit_id. Both are required and required first — this is a
    batch-loading tool, so there's no gateway headers to infer them
    from; the operator running this script must say explicitly which
    tenant AND which department these documents belong to.
    """
    directory = directory or settings.DOCUMENTS_DIR
    report = BatchReport()
    init_db()

    logger.info("Starting batch for directory: %s (tenant=%s org_unit=%s)", directory, tenant_id, org_unit_id)

    for filepath, _ in load_directory(directory):
        report.total += 1
        result = process_single_document(filepath, tenant_id, org_unit_id, is_ground_truth)
        report.results.append(result)

        if result.skipped:
            report.skipped += 1
        elif result.success:
            report.succeeded += 1
        else:
            report.failed += 1

    report.log()
    return report