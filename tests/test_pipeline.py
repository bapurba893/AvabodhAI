"""
tests/test_pipeline.py
----------------------
Unit tests for each pipeline step.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from db.models import DocumentSummaryOutput
from pipeline.storage import validate_summary
from pipeline.splitter import split_documents


# ── Pydantic validation tests ─────────────────────────────────────────────────

def test_validate_summary_happy_path():
    raw = {
        "summary_text": "This document covers quarterly revenue targets.",
        "chunk_count":  42,
        "reduce_model": "gpt-4o-mini",
    }
    meta = {
        "doc_name":    "report.pdf",
        "source_path": "/docs/report.pdf",
        "page_count":  10,
    }
    result = validate_summary(raw, meta)
    assert isinstance(result, DocumentSummaryOutput)
    assert result.doc_name == "report.pdf"
    assert len(result.summary_text) > 0


def test_validate_summary_cleans_whitespace():
    raw = {
        "summary_text": "   This   has   extra   spaces.   ",
        "chunk_count": 1,
        "reduce_model": "gpt-4o-mini",
    }
    meta = {"doc_name": "test.pdf", "source_path": "", "page_count": 1}
    result = validate_summary(raw, meta)
    assert "  " not in result.summary_text


def test_validate_summary_empty_fails():
    raw = {"summary_text": "", "chunk_count": 1, "reduce_model": "gpt-4o-mini"}
    meta = {"doc_name": "test.pdf", "source_path": "", "page_count": 1}
    with pytest.raises(ValueError):
        validate_summary(raw, meta)


def test_validate_summary_doc_name_strips_path():
    raw = {
        "summary_text": "Valid summary content here.",
        "chunk_count":  5,
        "reduce_model": "gpt-4o-mini",
    }
    meta = {"doc_name": "/deep/nested/path/file.pdf", "source_path": "", "page_count": 1}
    result = validate_summary(raw, meta)
    assert result.doc_name == "file.pdf"


# ── Chunk filtering tests ─────────────────────────────────────────────────────

@patch("pipeline.splitter.SemanticChunker")
@patch("pipeline.splitter.OpenAIEmbeddings")
def test_small_chunks_discarded(mock_embeddings, mock_chunker):
    tiny_doc  = Document(page_content="Hi", metadata={})
    large_doc = Document(
        page_content="This is a properly sized chunk with enough content to pass the minimum size filter.",
        metadata={},
    )
    mock_instance = MagicMock()
    mock_instance.split_documents.return_value = [tiny_doc, large_doc]
    mock_chunker.return_value = mock_instance

    docs = [Document(page_content="test", metadata={})]
    chunks = split_documents(docs)

    # tiny_doc should be filtered out
    for c in chunks:
        assert len(c.page_content) >= 100


def test_chunk_index_added():
    """Chunk metadata must include chunk_index for traceability."""
    with patch("pipeline.splitter.SemanticChunker") as mock_chunker, \
         patch("pipeline.splitter.OpenAIEmbeddings"):
        docs_out = [
            Document(
                page_content="A" * 150,
                metadata={"source": "test"},
            )
            for _ in range(3)
        ]
        mock_instance = MagicMock()
        mock_instance.split_documents.return_value = docs_out
        mock_chunker.return_value = mock_instance

        chunks = split_documents([Document(page_content="test", metadata={})])
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_index") == i
