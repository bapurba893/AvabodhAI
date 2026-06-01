"""
pipeline/orchestrator.py
------------------------
The main entry point that wires all 7 steps together.

process_single_document()  — for one file
process_directory()        — for a whole folder (lazy, memory-safe)

Production features added beyond your diagram:
- Per-document error isolation (one bad file doesn't stop the batch)
- Memory cleanup after each document (explicit del)
- Processing report at the end
- Duplicate detection before even loading the LLM
"""

import os
import gc
from dataclasses import dataclass, field
from typing import Optional

from pipeline.loader import load_single_document, load_directory
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from db.database import init_db
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ProcessingResult:
    """Result object for one document — returned to caller."""
    filepath:    str
    doc_name:    str
    success:     bool
    summary_id:  Optional[str] = None
    chunk_count: int = 0
    elapsed_sec: float = 0.0
    skipped:     bool = False      # True if duplicate detected
    error:       Optional[str] = None


@dataclass
class BatchReport:
    """Summary of a full directory processing run."""
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
        for r in self.results:
            if not r.success and not r.skipped:
                logger.error("  FAILED: %s — %s", r.doc_name, r.error)


def process_single_document(filepath: str) -> ProcessingResult:
    """
    Run the full 7-step pipeline for one document.

    Step 1 — Load
    Step 2 — Split
    (Dedup check before LLM to save cost)
    Step 3 — Map (LLM per chunk)
    Step 4 — Reduce (LLM combine)
    Step 5 — Pydantic validation
    Step 6 — Structured object
    Step 7 — Save to PostgreSQL
    """
    doc_name = os.path.basename(filepath)
    logger.info("--- Processing: %s ---", doc_name)

    docs = None
    chunks = None

    try:
        # ── Step 1: Load ──────────────────────────────────────────────────
        docs = load_single_document(filepath)

        if not docs:
            raise ValueError("Loader returned no documents")

        # Extract metadata from first doc (enriched by loader)
        file_hash   = docs[0].metadata.get("file_hash", "")
        source_path = docs[0].metadata.get("source_path", filepath)
        page_count  = len(docs)

        # ── Deduplication check (before any LLM call) ─────────────────────
        if file_hash:
            existing = check_duplicate(file_hash)
            if existing:
                return ProcessingResult(
                    filepath=filepath,
                    doc_name=doc_name,
                    success=True,
                    skipped=True,
                    summary_id=str(existing.id),
                )

        # ── Step 2: Split ─────────────────────────────────────────────────
        chunks = split_documents(docs)

        # Free original docs from RAM — chunks are all we need now
        del docs
        docs = None
        gc.collect()

        if not chunks:
            raise ValueError("Splitting produced no chunks")

        chunk_count = len(chunks)

        # ── Steps 3 + 4: Map → Reduce (LLM summarisation) ────────────────
        raw_result = summarise_document(chunks, doc_name=doc_name)

        # Free chunks from RAM — summary text is all we need now
        del chunks
        chunks = None
        gc.collect()

        # ── Steps 5 + 6: Pydantic validation → Structured object ─────────
        metadata = {
            "doc_name":    doc_name,
            "source_path": source_path,
            "page_count":  page_count,
        }
        validated = validate_summary(raw_result, metadata)

        # ── Step 7: Save to PostgreSQL ────────────────────────────────────
        record = save_summary(validated, file_hash, chunk_count)

        logger.info(
            "Done: '%s' | chunks=%d | %.2fs | id=%s",
            doc_name, chunk_count, raw_result["elapsed_sec"], record.id,
        )

        return ProcessingResult(
            filepath    = filepath,
            doc_name    = doc_name,
            success     = True,
            summary_id  = str(record.id),
            chunk_count = chunk_count,
            elapsed_sec = raw_result["elapsed_sec"],
        )

    except Exception as e:
        # Always clean up RAM even on failure
        if docs is not None:
            del docs
        if chunks is not None:
            del chunks
        gc.collect()

        logger.error(" Failed: '%s' — %s", doc_name, e, exc_info=True)
        return ProcessingResult(
            filepath = filepath,
            doc_name = doc_name,
            success  = False,
            error    = str(e),
        )


def process_directory(directory: str = None) -> BatchReport:
    """
    Process every supported document in a directory.
    Lazy-loads files one at a time — safe for folders with hundreds of PDFs.
    One document failure does not stop the rest.
    """
    directory = directory or settings.DOCUMENTS_DIR
    report = BatchReport()

    # Ensure DB tables exist before we start
    init_db()

    logger.info("Starting batch processing for directory: %s", directory)

    for filepath, _ in load_directory(directory):
        report.total += 1
        result = process_single_document(filepath)
        report.results.append(result)

        if result.skipped:
            report.skipped += 1
        elif result.success:
            report.succeeded += 1
        else:
            report.failed += 1

    report.log()
    return report
