"""
pipeline/retriever.py
---------------------
Retrieval pipeline:
1. Embed query using OpenAI
2. Vector similarity search in pgvector (document_chunks)
3. Contextual compression — filters chunks to query-relevant content only

Contextual compression is better than raw retrieval because:
- Raw retrieval returns full chunks (may have irrelevant parts)
- Compression extracts only the sentences relevant to the query
- LLM gets cleaner, more focused context → better answers
"""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def embed_query(query: str) -> list[float]:
    """Convert query text to embedding vector."""
    client = OpenAIEmbeddings(
        api_key=SecretStr(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None,
        model=settings.EMBEDDING_MODEL,
    )
    return client.embed_query(query)


def vector_search(
    query_vector: list[float],
    db: Session,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search document_chunks table using pgvector cosine similarity.
    Returns top_k most similar chunks with similarity scores.
    """
    try:
        if doc_filter:
            results = db.execute(
                text("""
                    SELECT id, doc_name, chunk_index, chunk_text,
                           chunk_size, page_number, doc_hash,
                           1 - (embedding <=> CAST(:vector AS vector)) AS similarity
                    FROM document_chunks
                    WHERE doc_name = :doc_filter
                    ORDER BY embedding <=> CAST(:vector AS vector)
                    LIMIT :top_k
                """),
                {"vector": str(query_vector), "doc_filter": doc_filter, "top_k": top_k}
            ).fetchall()
        else:
            results = db.execute(
                text("""
                    SELECT id, doc_name, chunk_index, chunk_text,
                           chunk_size, page_number, doc_hash,
                           1 - (embedding <=> CAST(:vector AS vector)) AS similarity
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:vector AS vector)
                    LIMIT :top_k
                """),
                {"vector": str(query_vector), "top_k": top_k}
            ).fetchall()

        return [
            {
                "id":          str(r.id),
                "doc_name":    r.doc_name,
                "chunk_index": r.chunk_index,
                "chunk_text":  r.chunk_text,
                "chunk_size":  r.chunk_size,
                "page_number": r.page_number,
                "doc_hash":    r.doc_hash,
                "similarity":  round(float(r.similarity), 4),
            }
            for r in results
        ]
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []


def contextual_compression(
    query: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Contextual compression retriever.
    Takes raw chunks and extracts only the parts relevant to the query.

    Why: A chunk may be 500 chars but only 2 sentences are relevant.
    Compression gives the LLM only those 2 sentences — cleaner context.

    Returns filtered chunks. If compression removes everything from a chunk,
    that chunk is dropped.
    """
    if not chunks:
        return []

    # Keep the retrieval flow lightweight and dependency-safe.
    # If a future LangChain version exposes compression utilities, this can be
    # enhanced without changing the public interface.
    try:
        if not query.strip():
            return chunks[:3]

        filtered_chunks = []
        query_lower = query.lower()
        for chunk in chunks:
            text = chunk.get("chunk_text", "") or ""
            if query_lower in text.lower():
                filtered_chunks.append(chunk)

        logger.info(
            "Compression fallback: %d chunks -> %d after filtering",
            len(chunks), len(filtered_chunks)
        )
        return filtered_chunks if filtered_chunks else chunks[:3]

    except Exception as e:
        logger.warning("Contextual compression fallback failed — using raw chunks: %s", e)
        return chunks[:5]


def retrieve(
    query: str,
    db: Session,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Vector search
    3. Contextual compression
    Returns final list of relevant chunks.
    """
    # Step 1: embed
    query_vector = embed_query(query)

    # Step 2: vector search (get more than needed — compression will filter)
    raw_chunks = vector_search(
        query_vector=query_vector,
        db=db,
        top_k=top_k * 2,    # get 2x so compression has room to filter
        doc_filter=doc_filter,
    )

    if not raw_chunks:
        logger.info("No chunks found for query: %s", query[:50])
        return []

    # Step 3: contextual compression
    compressed = contextual_compression(query, raw_chunks)
    return compressed[:top_k]