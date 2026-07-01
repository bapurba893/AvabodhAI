"""
api/routes/web.py
─────────────────────────────────────────────────────────────────
Single unified web scraping endpoint for AvabodhAI.

  POST /web/scrape

Three modes — controlled by the request body:
  - Single page  : 1 URL,  full_site=False  → scrape that page
  - Batch pages  : 2-10 URLs, full_site=False → scrape each page
  - Full website : 1 URL,  full_site=True   → BFS crawl whole site
"""

from typing import List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from api.schemas.web import WebScrapeRequest, WebScrapeResponse, PageResult, PageData
from db.database import get_db_session_fastapi
from pipeline.scraper import scrape_url_async, scrape_website_async
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from pipeline.embedder import store_chunk_embeddings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Internal helpers (pipeline is unchanged after Step 1)
# ─────────────────────────────────────────────────────────────

async def _process_single_url(
    url: str,
    wait_for_selector: str | None,
    extra_wait_ms: int,
    db: Session,
) -> dict:
    """Scrape one URL and run it through the full pipeline. Returns a plain dict."""
    try:
        docs = await scrape_url_async(
            url=url,
            wait_for_selector=wait_for_selector,
            extra_wait_ms=extra_wait_ms,
        )
        file_hash  = docs[0].metadata.get("file_hash", "")
        page_title = docs[0].metadata.get("title") or url
        page_count = len(docs)
        logger.info("Scraped '%s' — %d doc(s)", url, page_count)
    except Exception as e:
        logger.error("Scrape failed for %s: %s", url, e)
        return {"url": url, "status": "error", "error": f"Scrape failed: {e}"}

    if file_hash:
        existing = check_duplicate(file_hash)
        if existing:
            logger.info("Duplicate detected: %s", url)
            if existing.embedding_status == "pending":
                try:
                    chunks = split_documents(docs)
                    store_chunk_embeddings(
                        chunks=chunks, summary_id=existing.id,
                        doc_hash=file_hash, doc_name=existing.doc_name, source_path=url,
                    )
                except Exception as e:
                    logger.warning("Embedding retry failed: %s", e)
            return {
                "url": url,
                "status": "duplicate",
                "result": PageData(
                    id=existing.id, doc_name=existing.doc_name, url=url,
                    summary_text=existing.summary_text,
                    key_topics=existing.key_topics if isinstance(existing.key_topics, list) else None,
                    page_count=existing.page_count, chunk_count=existing.chunk_count,
                    source_path=existing.source_path, language=existing.language,
                    model_used=existing.model_used, doc_hash=existing.doc_hash,
                    created_at=existing.created_at, updated_at=existing.updated_at,
                    elapsed_sec=0.0, message="Duplicate — returning cached summary",
                ),
            }

    try:
        chunks = split_documents(docs)
        del docs
        if not chunks:
            return {"url": url, "status": "error", "error": "Page produced 0 chunks after splitting"}
        chunk_count = len(chunks)
        logger.info("Split into %d chunks", chunk_count)
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Splitting failed: {e}"}

    try:
        raw_result = summarise_document(chunks, doc_name=page_title)
        chunk_metadata    = raw_result.get("chunk_metadata", [])
        document_metadata = raw_result.get("document_metadata")
        logger.info("Summarisation complete in %.2fs", raw_result["elapsed_sec"])
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Summarisation failed: {e}"}

    try:
        validated = validate_summary(
            raw_result,
            {"doc_name": page_title, "source_path": url, "page_count": page_count},
        )
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Validation failed: {e}"}

    try:
        record = save_summary(validated, file_hash, chunk_count,document_metadata=document_metadata,
        )
        logger.info("Saved to DB — ID: %s", record.id)
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Database save failed: {e}"}

    try:
        store_chunk_embeddings(
            chunks=chunks, summary_id=record.id,
            doc_hash=file_hash, doc_name=page_title, source_path=url,chunk_metadata=chunk_metadata,
        )
        logger.info("Embeddings stored for '%s'", url)
    except Exception as e:
        logger.warning("Embedding storage failed (non-fatal): %s", e)

    return {
        "url": url,
        "status": "success",
        "result": PageData(
            id=record.id, doc_name=record.doc_name, url=url,
            summary_text=record.summary_text,
            key_topics=record.key_topics if isinstance(record.key_topics, list) else None,
            page_count=record.page_count, chunk_count=record.chunk_count,
            source_path=record.source_path, language=record.language,
            model_used=record.model_used, doc_hash=record.doc_hash,
            created_at=record.created_at, updated_at=record.updated_at,
            elapsed_sec=raw_result.get("elapsed_sec"),
        ),
    }


async def _process_crawled_doc(doc, db: Session) -> PageResult:
    """Run a single already-scraped Document through the pipeline (used in full_site mode)."""
    url        = doc.metadata.get("source", "unknown")
    file_hash  = doc.metadata.get("file_hash", "")
    page_title = doc.metadata.get("title") or url

    if file_hash:
        existing = check_duplicate(file_hash)
        if existing:
            logger.info("Duplicate skipped: %s", url)
            return PageResult(url=url, status="duplicate", detail="Already in knowledge base")

    try:
        chunks = split_documents([doc])
        if not chunks:
            return PageResult(url=url, status="error", detail="0 chunks after splitting")
        chunk_count = len(chunks)
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"Split failed: {e}")

    try:
        raw_result = summarise_document(chunks, doc_name=page_title)
        chunk_metadata    = raw_result.get("chunk_metadata", [])
        document_metadata = raw_result.get("document_metadata")
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"Summarisation failed: {e}")

    try:
        validated = validate_summary(
            raw_result,
            {"doc_name": page_title, "source_path": url, "page_count": 1},
        )
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"Validation failed: {e}")

    try:
        record = save_summary(validated, file_hash, chunk_count,document_metadata=document_metadata,
        )
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"DB save failed: {e}")

    try:
        store_chunk_embeddings(
            chunks=chunks, summary_id=record.id,
            doc_hash=file_hash, doc_name=page_title, source_path=url,chunk_metadata=chunk_metadata,
        )
    except Exception as e:
        logger.warning("Embedding storage failed (non-fatal): %s", e)

    return PageResult(
        url=url,
        status="success",
        data=PageData(
            id=record.id, doc_name=record.doc_name, url=url,
            summary_text=record.summary_text,
            key_topics=record.key_topics if isinstance(record.key_topics, list) else None,
            page_count=record.page_count, chunk_count=record.chunk_count,
            source_path=record.source_path, language=record.language,
            model_used=record.model_used, doc_hash=record.doc_hash,
            created_at=record.created_at, updated_at=record.updated_at,
            elapsed_sec=raw_result.get("elapsed_sec"),
        ),
    )


# ─────────────────────────────────────────────────────────────
# THE ONE ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.post(
    "/scrape",
    response_model=WebScrapeResponse,
    status_code=207,
    summary="Scrape URLs or crawl a full website",
    description="""
Single endpoint for all web scraping modes:

| Scenario | Request |
|---|---|
| Scrape 1 specific page | `urls: ["https://..."]` |
| Scrape 2–10 specific pages | `urls: ["https://...", "https://..."]` |
| Crawl entire website | `urls: ["https://..."], full_site: true` |

When `full_site=true`, exactly **one** URL is required. Use `max_pages` to set the
crawl depth limit (default 50, max 200). The mode is returned in the response
so you always know how it was handled.
    """,
)
async def scrape(
    body: WebScrapeRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    results: List[PageResult] = []

    # ── FULL SITE CRAWL ────────────────────────────────────────
    if body.full_site:
        url = str(body.urls[0])
        logger.info("Full site crawl: %s (max_pages=%d)", url, body.max_pages)

        try:
            docs = await scrape_website_async(
                url=url,
                max_pages=body.max_pages,
                same_domain_only=body.same_domain_only,
                wait_for_selector=body.wait_for_selector,
                extra_wait_ms=body.extra_wait_ms,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Crawl failed: {e}")

        if not docs:
            raise HTTPException(status_code=404, detail="No pages found at the given URL.")

        for doc in docs:
            item = await _process_crawled_doc(doc, db)
            results.append(item)

        mode = "full_site"

    # ── SINGLE PAGE or BATCH (specific URLs) ──────────────────
    else:
        for url_obj in body.urls:
            url = str(url_obj)
            logger.info("Scraping: %s", url)
            outcome = await _process_single_url(
                url, body.wait_for_selector, body.extra_wait_ms, db
            )
            results.append(PageResult(
                url=outcome["url"],
                status=outcome["status"],
                detail=outcome.get("error") if outcome["status"] == "error" else None,
                data=outcome.get("result") if outcome["status"] != "error" else None,
            ))

        mode = "single" if len(body.urls) == 1 else "batch"

    succeeded  = sum(1 for r in results if r.status == "success")
    duplicates = sum(1 for r in results if r.status == "duplicate")
    failed     = sum(1 for r in results if r.status == "error")

    return WebScrapeResponse(
        mode=mode,
        summary={
            "total":      len(results),
            "succeeded":  succeeded,
            "duplicates": duplicates,
            "failed":     failed,
        },
        results=results,
    )