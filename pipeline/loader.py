"""
pipeline/loader.py
------------------
Step 1 — Document upload and loading.

Handles:
- Multiple file types (.pdf, .txt, .docx, .csv)
- Lazy loading for large folders (memory efficient)
- SHA-256 hash per file (for deduplication)
- Metadata enrichment on each LangChain Document
- NEW: PDF image extraction via pdfplumber
"""

import hashlib
import io
import os
from pathlib import Path
from typing import Generator, Optional
from uuid import UUID

import pdfplumber
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from PIL import Image

from config.settings import get_settings
from pipeline.image_processor import (
    should_skip_image,
    compute_image_hash,
    caption_image_with_vision,
    build_image_embedding_text,
    ImageCaptionOutput,
)
from pipeline.embedder import ImageEmbeddingInput, check_image_hash_exists
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

LOADER_MAP = {
    ".pdf":  PyMuPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv":  CSVLoader,
}


def _compute_file_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _enrich_metadata(doc: Document, filepath: str, file_hash: str) -> Document:
    doc.metadata.update({
        "source_path": os.path.abspath(filepath),
        "doc_name":    os.path.basename(filepath),
        "file_hash":   file_hash,
        "file_size":   os.path.getsize(filepath),
    })
    return doc


def load_single_document(filepath: str) -> list[Document]:
    """Load one file and return LangChain Documents."""
    ext = Path(filepath).suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(LOADER_MAP.keys())}"
        )

    loader_cls = LOADER_MAP[ext]

    try:
        loader = loader_cls(filepath)
        docs = loader.load()
        file_hash = _compute_file_hash(filepath)
        enriched = [_enrich_metadata(d, filepath, file_hash) for d in docs]
        logger.info(
            "Loaded '%s' -> %d doc object(s) | hash=%s",
            os.path.basename(filepath), len(enriched), file_hash[:12],
        )
        return enriched
    except Exception as e:
        logger.error("Failed to load '%s': %s", filepath, e)
        raise


def load_directory(directory: str) -> Generator[tuple[str, list[Document]], None, None]:
    """Lazy-load every supported file in a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    supported_files = [
        f for f in directory.rglob("*")
        if f.suffix.lower() in LOADER_MAP and f.is_file()
    ]

    if not supported_files:
        logger.warning("No supported files found in '%s'", directory)
        return

    logger.info("Found %d supported file(s) in '%s'", len(supported_files), directory)

    for filepath in supported_files:
        try:
            docs = load_single_document(str(filepath))
            yield str(filepath), docs
        except Exception as e:
            logger.error("Skipping '%s' due to error: %s", filepath, e)
            continue


# ─────────────────────────────────────────────────────────────────────────────
# NEW — PDF image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(
    filepath:   str,
    summary_id: UUID,
    doc_hash:   str,
    doc_name:   str,
    tenant_id:  str,
    source_path: str = "",
) -> list[ImageEmbeddingInput]:
    """
    Extract all images from a PDF, caption them with GPT-4o Vision,
    and return a list of ImageEmbeddingInput ready for embed_and_store_images().

    Uses pdfplumber for page-level image extraction.
    Surrounding context = text on the same page near the image.

    tenant_id scopes the early dedup check (check_image_hash_exists) so a
    letterhead/watermark image shared across another tenant's documents
    doesn't cause this tenant's copy to be silently skipped.

    Returns empty list if PDF has no images or pdfplumber fails.
    """
    results: list[ImageEmbeddingInput] = []
    image_index = 0

    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            logger.info("Extracting images from '%s' (%d pages)", doc_name, total_pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # Get page text for surrounding context
                page_text = page.extract_text() or ""

                page_images = page.images
                if not page_images:
                    continue

                for img_obj in page_images:
                    try:
                        # pdfplumber image object has 'stream' key with raw bytes
                        raw_bytes = img_obj.get("stream")
                        if raw_bytes is None:
                            continue

                        # Convert stream to bytes if needed
                        if hasattr(raw_bytes, "get_data"):
                            image_bytes = raw_bytes.get_data()
                        elif isinstance(raw_bytes, (bytes, bytearray)):
                            image_bytes = bytes(raw_bytes)
                        else:
                            continue

                        width  = int(img_obj.get("width",  0))
                        height = int(img_obj.get("height", 0))
                        size   = len(image_bytes)

                        # Detect format from bytes magic number
                        image_format = _detect_image_format(image_bytes)

                        # Noise filter
                        skip, reason = should_skip_image(
                            image_bytes=image_bytes,
                            width=width,
                            height=height,
                            image_format=image_format,
                        )
                        if skip:
                            logger.debug("Skipping PDF image page=%d: %s", page_num, reason)
                            continue

                        # Dedup by hash — skip the Vision API call entirely
                        # if this exact image is already stored (e.g. a
                        # repeated letterhead/watermark across pages)
                        img_hash = compute_image_hash(image_bytes)
                        try:
                            if check_image_hash_exists(img_hash, tenant_id):
                                logger.debug(
                                    "Skipping already-embedded PDF image on page %d (hash=%s)",
                                    page_num, img_hash[:12],
                                )
                                continue
                        except Exception as e:
                            logger.debug("Dedup check failed, continuing anyway: %s", e)

                        # Caption with GPT-4o Vision
                        caption = caption_image_with_vision(
                            image_bytes=image_bytes,
                            surrounding_text=page_text[:800],
                        )
                        if caption is None:
                            logger.warning("Vision captioning failed for PDF image on page %d", page_num)
                            continue

                        # Skip low-confidence "other" images
                        if caption.image_type == "other" and caption.confidence < 0.5:
                            logger.debug("Skipping low-confidence image on page %d", page_num)
                            continue

                        embedding_text = build_image_embedding_text(
                            caption=caption,
                            doc_name=doc_name,
                            surrounding_text=page_text[:400],
                        )

                        results.append(ImageEmbeddingInput(
                            summary_id        = summary_id,
                            doc_hash          = doc_hash,
                            doc_name          = doc_name,
                            source_path       = source_path,
                            page_number       = page_num,
                            image_bytes_hash  = img_hash,
                            image_url         = None,   # PDF-embedded, no URL
                            image_format      = image_format,
                            image_width       = width,
                            image_height      = height,
                            image_size_bytes  = size,
                            image_caption     = caption.caption,
                            image_type        = caption.image_type,
                            image_alt_text    = caption.suggested_alt_text,
                            image_context     = page_text[:500],
                            contains_chart    = caption.contains_chart,
                            contains_table    = caption.contains_table,
                            contains_text_img = caption.contains_text,
                            key_elements      = caption.key_elements,
                            vision_model_used = "gpt-4o",
                            vision_confidence = caption.confidence,
                            embedding_text    = embedding_text,
                            chunk_index       = image_index,
                            total_chunks      = 1,   # updated later
                        ))

                        image_index += 1
                        logger.info(
                            "PDF image %d captured: page=%d type=%s confidence=%.2f",
                            image_index, page_num, caption.image_type, caption.confidence,
                        )

                    except Exception as e:
                        logger.warning("Failed to process PDF image on page %d: %s", page_num, e)
                        continue

    except Exception as e:
        logger.warning("pdfplumber image extraction failed for '%s': %s", doc_name, e)
        return []

    # Update total_chunks on all records now that we know the total
    total = len(results)
    for i, r in enumerate(results):
        r.chunk_index  = i
        r.total_chunks = total

    logger.info("Extracted %d images from PDF '%s'", total, doc_name)
    return results


def _detect_image_format(image_bytes: bytes) -> str:
    """Detect image format from magic bytes."""
    if image_bytes[:4] == b"\x89PNG":
        return "png"
    if image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    if image_bytes[:4] in (b"GIF8", b"GIF9"):
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    # Try PIL as fallback
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return (img.format or "jpeg").lower()
    except Exception:
        return "jpeg"