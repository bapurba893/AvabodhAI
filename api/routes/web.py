"""
api/routes/web.py
─────────────────────────────────────────────────────────────────
Single unified web scraping endpoint for AvabodhAI.

  POST /web/scrape

Three modes — controlled by the request body:
  - Single page  : 1 URL,  full_site=False  → scrape that page
  - Batch pages  : 2-10 URLs, full_site=False → scrape each page
  - Full website : 1 URL,  full_site=True   → BFS crawl whole site

After text pipeline, images from each page are extracted, captioned
with GPT-4o Vision, and embedded into pgvector.

Multi-tenancy + department isolation: the endpoint takes tenant_id AND
org_unit_id from the X-Tenant-ID / X-Org-Unit-ID headers and threads
both through every save/store/dedup call for every page — scraped
content is exactly as isolated as an uploaded document.

Client-supplied fields (category, effective_from/to, is_ground_truth)
apply the same way as document upload — the same values are stamped
onto every page scraped in one request, since they describe the request
as a whole (a "scrape this policy site" request is one category, one
ground-truth decision, applied to everything it finds).

Preview link: for scraped pages this is trivially just the source URL
(no signed token needed — there's no local file to protect access to,
the page is already public at that URL).
"""

from datetime import date, datetime, timezone
from typing import List, Optional

from bs4 import BeautifulSoup

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_tenant_id, get_org_unit_id
from api.schemas.web import WebScrapeRequest, WebScrapeResponse, PageResult, PageData
from db.database import get_db_session_fastapi
from pipeline.scraper import scrape_url_async, scrape_website_async
from pipeline.splitter import split_documents
from pipeline.summariser import summarise_document
from pipeline.storage import validate_summary, check_duplicate, save_summary
from pipeline.embedder import store_chunk_embeddings, embed_and_store_images, ImageEmbeddingInput, check_image_hash_exists
from pipeline.image_processor import extract_images_from_soup
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _to_utc_datetime(d: Optional[date]) -> Optional[datetime]:
    """Convert a plain date (from the request body) to a tz-aware datetime for storage."""
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────
# Helper — extract and embed images from a scraped page
# ─────────────────────────────────────────────────────────────

def _wire_web_images(
    html:       str,
    page_url:   str,
    page_text:  str,
    summary_id,
    doc_hash:   str,
    doc_name:   str,
    tenant_id:  str,
    org_unit_id: str,
    is_ground_truth: bool,
) -> None:
    """
    Parse HTML, extract images, caption with GPT-4o Vision, embed.
    Non-fatal — errors are logged and swallowed.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        image_dicts = extract_images_from_soup(
            soup        = soup,
            base_url    = page_url,
            doc_name    = doc_name,
            summary_id  = summary_id,
            doc_hash    = doc_hash,
            source_path = page_url,
            page_text   = page_text,
            # Wrap the tenant+org-scoped dedup check as a plain (hash) -> bool
            # callable so image_processor.py stays decoupled from embedder.py.
            hash_exists_fn = lambda h: check_image_hash_exists(h, tenant_id, org_unit_id),
        )
        if image_dicts:
            image_inputs = [ImageEmbeddingInput(**d) for d in image_dicts]
            embed_and_store_images(
                images     = image_inputs,
                summary_id = summary_id,
                doc_name   = doc_name,
                tenant_id  = tenant_id,
                org_unit_id = org_unit_id,
                is_ground_truth = is_ground_truth,
            )
            logger.info("Web images embedded: %d from '%s'", len(image_inputs), page_url[:60])
    except Exception as e:
        logger.warning("Web image pipeline failed for %s (non-fatal): %s", page_url[:60], e)


# ─────────────────────────────────────────────────────────────
# Internal helpers (text pipeline)
# ─────────────────────────────────────────────────────────────

async def _process_single_url(
    url: str,
    wait_for_selector: str | None,
    extra_wait_ms: int,
    tenant_id: str,
    org_unit_id: str,
    category: Optional[str],
    effective_from: Optional[datetime],
    effective_to: Optional[datetime],
    is_ground_truth: bool,
    db: Session,
) -> dict:
    try:
        docs = await scrape_url_async(
            url=url,
            wait_for_selector=wait_for_selector,
            extra_wait_ms=extra_wait_ms,
        )
        file_hash  = docs[0].metadata.get("file_hash", "")
        page_title = docs[0].metadata.get("title") or url
        page_count = len(docs)
        # Store raw HTML for image extraction (scraper stores it in metadata)
        raw_html   = docs[0].metadata.get("raw_html", "")
        page_text  = docs[0].page_content
        logger.info("Scraped '%s' — %d doc(s)", url, page_count)
    except Exception as e:
        logger.error("Scrape failed for %s: %s", url, e)
        return {"url": url, "status": "error", "error": f"Scrape failed: {e}"}

    if file_hash:
        existing = check_duplicate(file_hash, tenant_id, org_unit_id)
        if existing:
            logger.info("Duplicate detected: %s (tenant=%s org_unit=%s)", url, tenant_id, org_unit_id)
            if existing.embedding_status == "pending":
                try:
                    chunks = split_documents(docs)
                    store_chunk_embeddings(
                        chunks=chunks, summary_id=existing.id,
                        doc_hash=file_hash, doc_name=existing.doc_name,
                        tenant_id=tenant_id, org_unit_id=org_unit_id,
                        is_ground_truth=existing.is_ground_truth,
                        source_path=url,
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
                    tenant_id=existing.tenant_id, org_unit_id=existing.org_unit_id,
                    category=existing.category, effective_from=existing.effective_from,
                    effective_to=existing.effective_to, is_ground_truth=existing.is_ground_truth,
                    preview_link=url,
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
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Splitting failed: {e}"}

    try:
        raw_result = summarise_document(chunks, doc_name=page_title)
        chunk_metadata    = raw_result.get("chunk_metadata", [])
        document_metadata = raw_result.get("document_metadata")
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
        record = save_summary(
            validated, file_hash, chunk_count,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            document_metadata=document_metadata,
            category=category, effective_from=effective_from,
            effective_to=effective_to, is_ground_truth=is_ground_truth,
        )
    except Exception as e:
        return {"url": url, "status": "error", "error": f"Database save failed: {e}"}

    try:
        store_chunk_embeddings(
            chunks=chunks, summary_id=record.id,
            doc_hash=file_hash, doc_name=page_title,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            is_ground_truth=is_ground_truth,
            source_path=url, chunk_metadata=chunk_metadata,
        )
    except Exception as e:
        logger.warning("Text embedding storage failed (non-fatal): %s", e)

    # ── Extract and embed images from web page ─────────────────────────────
    if raw_html:
        _wire_web_images(
            html       = raw_html,
            page_url   = url,
            page_text  = page_text,
            summary_id = record.id,
            doc_hash   = file_hash,
            doc_name   = page_title,
            tenant_id  = tenant_id,
            org_unit_id = org_unit_id,
            is_ground_truth = is_ground_truth,
        )

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
            tenant_id=record.tenant_id, org_unit_id=record.org_unit_id,
            category=record.category, effective_from=record.effective_from,
            effective_to=record.effective_to, is_ground_truth=record.is_ground_truth,
            preview_link=url,
            created_at=record.created_at, updated_at=record.updated_at,
            elapsed_sec=raw_result.get("elapsed_sec"),
        ),
    }


async def _process_crawled_doc(
    doc, tenant_id: str, org_unit_id: str,
    category: Optional[str], effective_from: Optional[datetime],
    effective_to: Optional[datetime], is_ground_truth: bool,
    db: Session,
) -> PageResult:
    url        = doc.metadata.get("source", "unknown")
    file_hash  = doc.metadata.get("file_hash", "")
    page_title = doc.metadata.get("title") or url
    raw_html   = doc.metadata.get("raw_html", "")
    page_text  = doc.page_content

    if file_hash:
        existing = check_duplicate(file_hash, tenant_id, org_unit_id)
        if existing:
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
        validated = validate_summary(raw_result, {"doc_name": page_title, "source_path": url, "page_count": 1})
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"Validation failed: {e}")

    try:
        record = save_summary(
            validated, file_hash, chunk_count,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            document_metadata=document_metadata,
            category=category, effective_from=effective_from,
            effective_to=effective_to, is_ground_truth=is_ground_truth,
        )
    except Exception as e:
        return PageResult(url=url, status="error", detail=f"DB save failed: {e}")

    try:
        store_chunk_embeddings(
            chunks=chunks, summary_id=record.id,
            doc_hash=file_hash, doc_name=page_title,
            tenant_id=tenant_id, org_unit_id=org_unit_id,
            is_ground_truth=is_ground_truth,
            source_path=url, chunk_metadata=chunk_metadata,
        )
    except Exception as e:
        logger.warning("Text embedding storage failed (non-fatal): %s", e)

    # ── Extract and embed images ────────────────────────────────────────────
    if raw_html:
        _wire_web_images(
            html       = raw_html,
            page_url   = url,
            page_text  = page_text,
            summary_id = record.id,
            doc_hash   = file_hash,
            doc_name   = page_title,
            tenant_id  = tenant_id,
            org_unit_id = org_unit_id,
            is_ground_truth = is_ground_truth,
        )

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
            tenant_id=record.tenant_id, org_unit_id=record.org_unit_id,
            category=record.category, effective_from=record.effective_from,
            effective_to=record.effective_to, is_ground_truth=record.is_ground_truth,
            preview_link=url,
            created_at=record.created_at, updated_at=record.updated_at,
            elapsed_sec=raw_result.get("elapsed_sec"),
        ),
    )


# ─────────────────────────────────────────────────────────────
# THE ONE ENDPOINT (unchanged contract, now tenant+org-scoped)
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
    """,
)
async def scrape(
    body: WebScrapeRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    results: List[PageResult] = []
    effective_from = _to_utc_datetime(body.effective_from)
    effective_to   = _to_utc_datetime(body.effective_to)

    if body.full_site:
        url = str(body.urls[0])
        logger.info("Full site crawl: %s (max_pages=%d, tenant=%s org_unit=%s)",
                   url, body.max_pages, tenant_id, org_unit_id)

        try:
            docs = await scrape_website_async(
                url=url, max_pages=body.max_pages,
                same_domain_only=body.same_domain_only,
                wait_for_selector=body.wait_for_selector,
                extra_wait_ms=body.extra_wait_ms,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Crawl failed: {e}")

        if not docs:
            raise HTTPException(status_code=404, detail="No pages found at the given URL.")

        for doc in docs:
            item = await _process_crawled_doc(
                doc, tenant_id, org_unit_id,
                body.category, effective_from, effective_to, body.is_ground_truth,
                db,
            )
            results.append(item)

        mode = "full_site"

    else:
        for url_obj in body.urls:
            url = str(url_obj)
            outcome = await _process_single_url(
                url, body.wait_for_selector, body.extra_wait_ms,
                tenant_id, org_unit_id,
                body.category, effective_from, effective_to, body.is_ground_truth,
                db,
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
        summary={"total": len(results), "succeeded": succeeded, "duplicates": duplicates, "failed": failed},
        results=results,
    )