"""
pipeline/image_processor.py
───────────────────────────
Core image understanding pipeline for AvabodhAI.

All three sources (PDF upload, web scraping, chat image upload)
call this single module. It:

  1. Filters noise (too small, UI icons, duplicates)
  2. Resizes to max 1024px for GPT-4o Vision (saves tokens)
  3. Calls GPT-4o Vision with structured Pydantic output
  4. Returns ImageCaptionOutput ready for embedding + storage

GPT-4o Vision SEES the actual pixels — not just alt text.
It reads chart values, understands diagrams, transcribes
embedded text in infographics, and describes photos.
"""

import base64
import hashlib
import io
import re
from typing import Optional

import httpx
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────

# Images smaller than this are noise (icons, tracking pixels, spacers)
_MIN_SIZE_BYTES  = 5_000       # 5KB
_MIN_DIMENSION   = 50          # 50px width or height

# Resize large images before Vision API call — saves tokens
# GPT-4o detail="auto": tiles ≤2048px; detail="low": fixed 512px
_MAX_DIMENSION   = 1024        # resize to this if larger

# CSS class / ID patterns that indicate UI chrome to skip
_NOISE_PATTERNS  = re.compile(
    r"logo|icon|avatar|emoji|badge|favicon|spinner|loader"
    r"|social|share|arrow|bullet|checkmark|star|rating|thumbnail",
    re.IGNORECASE,
)

VISION_MODEL = "gpt-4o"   # sees actual pixels, reads charts, diagrams, text


# ── Pydantic output schema ───────────────────────────────────────────────────

class ImageCaptionOutput(BaseModel):
    """
    Structured output from GPT-4o Vision.
    Every field is populated from the model's understanding of the image —
    not from alt text or filename.
    """
    caption:          str        = Field(description="Detailed description of what the image shows")
    image_type:       str        = Field(description="chart|table|diagram|photo|screenshot|infographic|other")
    contains_chart:   bool       = Field(default=False)
    contains_table:   bool       = Field(default=False)
    contains_text:    bool       = Field(default=False, description="True if image has embedded text (infographic etc)")
    key_elements:     list[str]  = Field(default_factory=list, description="Key visual elements: axis labels, column headers, objects")
    suggested_alt_text: str      = Field(default="", description="Short accessible description max 125 chars")
    confidence:       float      = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("image_type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"chart", "table", "diagram", "photo", "screenshot", "infographic", "other"}
        v = str(v).strip().lower()
        return v if v in allowed else "other"

    @field_validator("caption", "suggested_alt_text", mode="before")
    @classmethod
    def clean_str(cls, v) -> str:
        return str(v).strip() if v else ""


# ── Noise filter ─────────────────────────────────────────────────────────────

def should_skip_image(
    image_bytes: bytes,
    width:       int,
    height:      int,
    src:         str = "",
    css_classes: str = "",
    image_format: str = "",
) -> tuple[bool, str]:
    """
    Returns (True, reason) if image should be skipped, (False, "") otherwise.
    """
    size = len(image_bytes)

    if size < _MIN_SIZE_BYTES:
        return True, f"too small ({size} bytes < {_MIN_SIZE_BYTES})"

    if width < _MIN_DIMENSION or height < _MIN_DIMENSION:
        return True, f"too small dimensions ({width}x{height})"

    if src.startswith("data:image"):
        return True, "inline data URI — UI element"

    if image_format.lower() == "svg" and size < 10_000:
        return True, "small SVG — likely UI icon"

    combined = f"{src} {css_classes}".lower()
    if _NOISE_PATTERNS.search(combined):
        return True, f"noise pattern detected in src/class: {combined[:60]}"

    return False, ""


def compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 of raw image bytes — for dedup."""
    return hashlib.sha256(image_bytes).hexdigest()


# ── Image preparation ────────────────────────────────────────────────────────

def prepare_image_for_vision(image_bytes: bytes) -> tuple[str, str, int, int, str]:
    """
    Resize if needed, convert to JPEG for consistent encoding.
    Returns (base64_string, format, width, height, media_type).
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB — Vision API doesn't like RGBA/P mode JPEGs
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size

    # Resize if larger than max dimension (preserves aspect ratio)
    if max(width, height) > _MAX_DIMENSION:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        width, height = img.size

    # Encode as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    jpeg_bytes = buffer.read()

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return b64, "jpeg", width, height, "image/jpeg"


# ── GPT-4o Vision call ───────────────────────────────────────────────────────

_VISION_PROMPT = """Analyze this image carefully and respond with a JSON object.

You must return ONLY valid JSON with these exact fields:
{
  "caption": "<detailed description — if chart: describe axes, values, trends; if table: describe columns and key data; if diagram: describe components and flow; if photo: describe scene and objects>",
  "image_type": "<one of: chart, table, diagram, photo, screenshot, infographic, other>",
  "contains_chart": <true or false>,
  "contains_table": <true or false>,
  "contains_text": <true if image contains readable embedded text>,
  "key_elements": ["<element1>", "<element2>", ...],
  "suggested_alt_text": "<max 125 chars accessible description>",
  "confidence": <0.0 to 1.0 — your confidence in this analysis>
}

Important:
- Read chart axes and approximate values if visible
- Transcribe any text embedded in the image
- For diagrams, describe the relationships and flow
- key_elements should be specific (e.g. "Y-axis: Revenue in millions", "Q4 bar reaching 85")
- Be precise and factual, not generic"""


def caption_image_with_vision(
    image_bytes:    bytes,
    surrounding_text: str = "",
) -> Optional[ImageCaptionOutput]:
    """
    Call GPT-4o Vision and return structured ImageCaptionOutput.
    Returns None if Vision call fails (non-fatal — image skipped).
    """
    try:
        b64, fmt, width, height, media_type = prepare_image_for_vision(image_bytes)
    except Exception as e:
        logger.warning("Image preparation failed: %s", e)
        return None

    # Build context hint if surrounding text provided
    context_hint = ""
    if surrounding_text and len(surrounding_text.strip()) > 10:
        context_hint = f"\n\nContext from surrounding document text:\n{surrounding_text[:500]}"

    prompt = _VISION_PROMPT + context_hint

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=800,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type":  "image_url",
                            "image_url": {
                                "url":    f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw_json = response.choices[0].message.content
        import json
        data = json.loads(raw_json)
        result = ImageCaptionOutput(**data)

        logger.info(
            "Vision captioned image: type=%s confidence=%.2f elements=%d",
            result.image_type, result.confidence, len(result.key_elements),
        )
        return result

    except Exception as e:
        logger.warning("GPT-4o Vision call failed: %s", e)
        return None


# ── Download image from URL (web scraping mode) ──────────────────────────────

def download_image(url: str, timeout: int = 10) -> Optional[bytes]:
    """
    Download image bytes from a URL.
    Returns None on failure (non-fatal).
    """
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        if resp.status_code == 200:
            return resp.content
        logger.warning("Image download failed %s: HTTP %d", url, resp.status_code)
        return None
    except Exception as e:
        logger.warning("Image download error %s: %s", url, e)
        return None


# ── Build embedding text from caption ────────────────────────────────────────

def build_image_embedding_text(
    caption:      ImageCaptionOutput,
    doc_name:     str,
    surrounding_text: str = "",
) -> str:
    """
    Construct the composite text that gets embedded into pgvector.
    Rich signal = better semantic search results.
    The [IMAGE] prefix distinguishes image chunks from text chunks.
    """
    parts = [
        "[IMAGE]",
        f"Caption: {caption.caption}",
        f"Type: {caption.image_type}",
    ]
    if caption.key_elements:
        parts.append(f"Key elements: {', '.join(caption.key_elements)}")
    if caption.suggested_alt_text:
        parts.append(f"Alt text: {caption.suggested_alt_text}")
    if surrounding_text and len(surrounding_text.strip()) > 10:
        parts.append(f"Context: {surrounding_text[:400]}")
    parts.append(f"Source document: {doc_name}")

    return "\n".join(parts)


# ── Extract images from HTML soup (web scraping mode) ─────────────────────────

def extract_images_from_soup(
    soup,
    base_url:   str,
    doc_name:   str,
    summary_id,
    doc_hash:   str,
    source_path: str = "",
    page_text:  str = "",
) -> list:
    """
    Extract, download, caption, and prepare images found in BeautifulSoup HTML.
    Called from scraper.py after page is fetched.
    Returns list[ImageEmbeddingInput] ready for embed_and_store_images().

    Import ImageEmbeddingInput from pipeline.embedder where needed to avoid
    circular imports — this function returns plain dicts that the caller
    converts to ImageEmbeddingInput.
    """
    from urllib.parse import urljoin, urlparse
    results = []
    image_index = 0

    img_tags = soup.find_all("img")
    logger.info("Found %d <img> tags in '%s'", len(img_tags), doc_name)

    for img_tag in img_tags:
        try:
            src = img_tag.get("src", "").strip()
            if not src:
                continue

            # Make absolute URL
            if not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)

            # Skip data URIs
            if src.startswith("data:image"):
                continue

            alt_text   = img_tag.get("alt", "")
            css_class  = " ".join(img_tag.get("class", []))

            # Get surrounding context — parent figure/section text
            parent = img_tag.find_parent(["figure", "section", "article", "div", "p"])
            surrounding = parent.get_text(" ", strip=True)[:600] if parent else page_text[:400]

            # Download image bytes
            image_bytes = download_image(src)
            if image_bytes is None:
                continue

            # Detect format + get dimensions
            fmt = _detect_image_format_bytes(image_bytes)
            try:
                from PIL import Image as PILImage
                import io as _io
                pil_img = PILImage.open(_io.BytesIO(image_bytes))
                width, height = pil_img.size
            except Exception:
                width, height = 0, 0

            size = len(image_bytes)

            # Noise filter
            skip, reason = should_skip_image(
                image_bytes=image_bytes,
                width=width,
                height=height,
                src=src,
                css_classes=css_class,
                image_format=fmt,
            )
            if skip:
                logger.debug("Skipping web image %s: %s", src[:60], reason)
                continue

            img_hash = compute_image_hash(image_bytes)

            # Caption with GPT-4o Vision
            caption = caption_image_with_vision(
                image_bytes=image_bytes,
                surrounding_text=surrounding,
            )
            if caption is None:
                continue

            if caption.image_type == "other" and caption.confidence < 0.5:
                continue

            embedding_text = build_image_embedding_text(
                caption=caption,
                doc_name=doc_name,
                surrounding_text=surrounding,
            )

            results.append({
                "summary_id":        summary_id,
                "doc_hash":          doc_hash,
                "doc_name":          doc_name,
                "source_path":       source_path,
                "page_number":       None,
                "image_bytes_hash":  img_hash,
                "image_url":         src,
                "image_format":      fmt,
                "image_width":       width,
                "image_height":      height,
                "image_size_bytes":  size,
                "image_caption":     caption.caption,
                "image_type":        caption.image_type,
                "image_alt_text":    alt_text or caption.suggested_alt_text,
                "image_context":     surrounding[:500],
                "contains_chart":    caption.contains_chart,
                "contains_table":    caption.contains_table,
                "contains_text_img": caption.contains_text,
                "key_elements":      caption.key_elements,
                "vision_model_used": "gpt-4o",
                "vision_confidence": caption.confidence,
                "embedding_text":    embedding_text,
                "chunk_index":       image_index,
                "total_chunks":      1,
            })
            image_index += 1
            logger.info(
                "Web image %d captured from %s: type=%s",
                image_index, src[:50], caption.image_type,
            )

        except Exception as e:
            logger.warning("Failed to process web image: %s", e)
            continue

    # Update total_chunks
    total = len(results)
    for i, r in enumerate(results):
        r["chunk_index"]  = i
        r["total_chunks"] = total

    logger.info("Extracted %d web images from '%s'", total, doc_name)
    return results


def _detect_image_format_bytes(image_bytes: bytes) -> str:
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if image_bytes[:4] in (b"GIF8", b"GIF9"):
        return "gif"
    if image_bytes[:4] == b"RIFF" and len(image_bytes) > 12 and image_bytes[8:12] == b"WEBP":
        return "webp"
    try:
        import io as _io
        from PIL import Image as PILImage
        img = PILImage.open(_io.BytesIO(image_bytes))
        return (img.format or "jpeg").lower()
    except Exception:
        return "jpeg"