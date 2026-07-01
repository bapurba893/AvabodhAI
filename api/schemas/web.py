"""
api/schemas/web.py
──────────────────
Unified Pydantic schemas for POST /web/scrape.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, HttpUrl, Field, model_validator


# ── Per-page data (returned on success/duplicate) ──────────────

class PageData(BaseModel):
    """Summary data for one successfully scraped/crawled page."""
    id:           UUID
    doc_name:     str
    url:          str
    summary_text: str
    key_topics:   Optional[List[str]]
    page_count:   int
    chunk_count:  int
    source_path:  Optional[str]
    language:     Optional[str]
    model_used:   Optional[str]
    doc_hash:     Optional[str]
    created_at:   datetime
    updated_at:   Optional[datetime]
    elapsed_sec:  Optional[float]
    message:      Optional[str] = None

    model_config = {"from_attributes": True}


class PageResult(BaseModel):
    """Result entry for a single URL in any mode."""
    url:    str
    status: str = Field(..., description="'success' | 'duplicate' | 'error'")
    detail: Optional[str] = None
    data:   Optional[PageData] = None


# ── Unified request ────────────────────────────────────────────

class WebScrapeRequest(BaseModel):
    """
    Unified body for POST /web/scrape.

    Three modes — all through the same endpoint:

      Single page  → urls has 1 item,    full_site=False  (default)
      Batch pages  → urls has 2–10 items, full_site=False  (default)
      Full website → urls has 1 item,    full_site=True
    """
    urls: List[HttpUrl] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "URL(s) to scrape. "
            "Supply 1–10 URLs for single/batch mode. "
            "Exactly 1 URL when full_site=True."
        ),
    )
    full_site: bool = Field(
        default=False,
        description=(
            "Set True to crawl the entire website starting from urls[0]. "
            "urls must contain exactly one entry."
        ),
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Max pages to crawl. Only relevant when full_site=True.",
    )
    same_domain_only: bool = Field(
        default=True,
        description="Stay within the same domain. Only relevant when full_site=True.",
    )
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="CSS selector to wait for before scraping (useful for JS-heavy SPAs).",
        examples=["article", "#mw-content-text", "main"],
    )
    extra_wait_ms: int = Field(
        default=1500,
        ge=0,
        le=10_000,
        description="Extra ms to wait after page load for lazy-loaded content.",
    )

    @model_validator(mode="after")
    def validate_full_site_single_url(self):
        if self.full_site and len(self.urls) != 1:
            raise ValueError(
                f"full_site=True requires exactly 1 URL in 'urls'. Got {len(self.urls)}."
            )
        return self

    # ── FIXED: single flat "example" (not "examples" dropdown) ─────
    # This makes Swagger's "Example Value" box show the real,
    # ready-to-send JSON body directly — no summary/value wrapper
    # for the user to accidentally copy.
    model_config = {"json_schema_extra": {
        "example": {
            "urls": ["https://example.com"],
            "full_site": True,
            "max_pages": 30,
            "same_domain_only": True,
            "extra_wait_ms": 1500,
        }
    }}


# ── Unified response ───────────────────────────────────────────

class WebScrapeResponse(BaseModel):
    """Unified response for all three scrape modes."""
    mode:    str  = Field(..., description="'single' | 'batch' | 'full_site'")
    summary: dict = Field(..., description="Counts: total, succeeded, duplicates, failed")
    results: List[PageResult]