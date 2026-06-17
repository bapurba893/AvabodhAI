"""
pipeline/loader.py
------------------
Step 1 — Document upload and loading.

Handles:
- Multiple file types (.pdf, .txt, .docx, .csv)
- Lazy loading for large folders (memory efficient)
- SHA-256 hash per file (used later for deduplication — don't re-summarise
  a file that was already processed)
- Metadata enrichment on each LangChain Document object
"""

import hashlib
import os
from pathlib import Path
from typing import Generator

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Map file extension → LangChain loader class
LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt":  TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv":  CSVLoader,
}


def _compute_file_hash(filepath: str) -> str:
    """SHA-256 hash of file contents — used to detect duplicate uploads."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _enrich_metadata(doc: Document, filepath: str, file_hash: str) -> Document:
    """
    Add extra metadata fields to every Document object.
    These travel with each chunk through the entire pipeline.
    """
    doc.metadata.update({
        "source_path": os.path.abspath(filepath),
        "doc_name":    os.path.basename(filepath),
        "file_hash":   file_hash,
        "file_size":   os.path.getsize(filepath),
    })
    return doc


def load_single_document(filepath: str) -> list[Document]:
    """
    Load one file and return a list of LangChain Document objects.
    Raises ValueError if the file type is not supported.
    """
    ext = Path(filepath).suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(LOADER_MAP.keys())}"
        )

    loader_cls = LOADER_MAP[ext]

    try:
        # PDF: use mode="single" so entire PDF = one Document (page metadata added later)
        loader = loader_cls(filepath)

        docs = loader.load()
        file_hash = _compute_file_hash(filepath)

        enriched = [_enrich_metadata(d, filepath, file_hash) for d in docs]
        logger.info(
            "Loaded '%s' -> %d document object(s) | hash=%s",
            os.path.basename(filepath), len(enriched), file_hash[:12],
        )
        return enriched

    except Exception as e:
        logger.error("Failed to load '%s': %s", filepath, e)
        raise


def load_directory(directory: str) -> Generator[tuple[str, list[Document]], None, None]:
    """
    Lazy-load every supported file in a directory one at a time.
    Yields (filepath, documents) tuples — never holds the whole folder in RAM.

    Why lazy?  If the folder has 500 PDFs, loading all at once would exhaust
    RAM.  Yielding one file at a time means we process → summarise → save →
    free memory before loading the next file.
    """
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

    logger.info(
        "Found %d supported file(s) in '%s'", len(supported_files), directory
    )

    for filepath in supported_files:
        try:
            docs = load_single_document(str(filepath))
            yield str(filepath), docs
        except Exception as e:
            # Skip broken files — log and continue with others
            logger.error("Skipping '%s' due to error: %s", filepath, e)
            continue
