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

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def embed_query(query: str) -> list[float]:
    """Convert query text to embedding vector."""
    client = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
    )
    return client.embed_query(query)


def vector_search(
    query_vector: list[float],
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search document_chunks table using pgvector cosine similarity.
    Returns top_k most similar chunks with similarity scores.

    tenant_id and org_unit_id are BOTH required (no defaults) and always
    applied together to the WHERE clause — org_unit_id is a hard boundary
    WITHIN a tenant (department-level), so a department code that happens
    to collide across two different tenants must never leak across
    companies. Never filter on org_unit_id alone.

    is_ground_truth = true is always applied too, unconditionally — there
    is no query-time toggle for this by design (confirmed: only documents
    explicitly marked ground truth are ever retrievable).

    role_filter: 'text' | 'image' | None (None = search both — this is how
    a text-only query still surfaces relevant image captions automatically,
    since image captions live in the same table with the same embedding model).

    Each result dict now also carries role/image_type/image_caption/
    section_heading/topic/chunk_type so callers (memory.py prompt building,
    chat.py source attribution) can tell text chunks and image chunks apart
    and render them differently.
    """
    where_clauses = [
        "tenant_id = :tenant_id",
        "org_unit_id = :org_unit_id",
        "is_ground_truth = true",
    ]
    params: dict = {
        "vector": str(query_vector), "top_k": top_k,
        "tenant_id": tenant_id, "org_unit_id": org_unit_id,
    }

    if doc_filter:
        where_clauses.append("doc_name = :doc_filter")
        params["doc_filter"] = doc_filter

    if role_filter:
        where_clauses.append("role = :role_filter")
        params["role_filter"] = role_filter

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = text(f"""
        SELECT id, doc_name, chunk_index, chunk_text,
               chunk_size, page_number, doc_hash,
               role, image_type, image_caption, image_url,
               section_heading, topic, chunk_type,
               1 - (embedding <=> CAST(:vector AS vector)) AS similarity
        FROM document_chunks
        {where_sql}
        ORDER BY embedding <=> CAST(:vector AS vector)
        LIMIT :top_k
    """)

    try:
        results = db.execute(query, params).fetchall()

        return [
            {
                "id":              str(r.id),
                "doc_name":        r.doc_name,
                "chunk_index":     r.chunk_index,
                "chunk_text":      r.chunk_text,
                "chunk_size":      r.chunk_size,
                "page_number":     r.page_number,
                "doc_hash":        r.doc_hash,
                "role":            r.role or "text",
                "image_type":      r.image_type,
                "image_caption":   r.image_caption,
                "image_url":       r.image_url,
                "section_heading": r.section_heading,
                "topic":           r.topic,
                "chunk_type":      r.chunk_type,
                "similarity":      round(float(r.similarity), 4),
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

    # Image chunks are already a dense, purpose-built caption — running them
    # through an LLM sentence-extractor tends to mangle or drop the caption,
    # so only text chunks go through compression. Image chunks pass straight
    # through untouched, keeping their full caption intact.
    text_chunks  = [c for c in chunks if c.get("role", "text") == "text"]
    image_chunks = [c for c in chunks if c.get("role") == "image"]

    if not text_chunks:
        return image_chunks

    # Convert to LangChain Documents for compression
    docs = [
        Document(
            page_content=c["chunk_text"],
            metadata={
                "doc_name":    c["doc_name"],
                "chunk_index": c["chunk_index"],
                "similarity":  c["similarity"],
                "id":          c["id"],
                "role":        c.get("role", "text"),
            }
        )
        for c in text_chunks
    ]

    try:
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.MAP_MODEL,
            temperature=0.0,
        )
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = compressor.compress_documents(docs, query)

        # Map compressed docs back to original chunk metadata.
        # Match by id (unique per row) — chunk_index alone isn't safe because
        # text and image chunks are each independently 0-indexed per document,
        # so a text chunk_index=0 and an image chunk_index=0 can collide.
        by_id = {c["id"]: c for c in text_chunks}
        compressed_chunks = []
        for doc in compressed_docs:
            if not doc.page_content.strip():
                continue
            original = by_id.get(doc.metadata.get("id"))
            if original:
                compressed_chunks.append({
                    **original,
                    "chunk_text": doc.page_content,  # compressed text
                    "compressed": True,
                })

        if not compressed_chunks:
            compressed_chunks = text_chunks[:3]

        # Merge image chunks back in untouched, re-sort by similarity
        merged = compressed_chunks + image_chunks
        merged.sort(key=lambda c: c.get("similarity", 0), reverse=True)

        logger.info(
            "Compression: %d chunks -> %d text (+%d image) after filtering",
            len(text_chunks), len(compressed_chunks), len(image_chunks),
        )
        return merged

    except Exception as e:
        logger.warning("Contextual compression failed — using raw chunks: %s", e)
        return (text_chunks[:5] + image_chunks)


def retrieve(
    query: str,
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Vector search (both text and image chunks, unless role_filter narrows it) —
       always scoped to tenant_id + org_unit_id, always ground-truth-only
    3. Contextual compression
    Returns final list of relevant chunks.
    """
    # Step 1: embed
    query_vector = embed_query(query)

    # Step 2: vector search (get more than needed — compression will filter)
    raw_chunks = vector_search(
        query_vector=query_vector,
        db=db,
        tenant_id=tenant_id,
        org_unit_id=org_unit_id,
        top_k=top_k * 2,    # get 2x so compression has room to filter
        doc_filter=doc_filter,
        role_filter=role_filter,
    )

    if not raw_chunks:
        logger.info("No chunks found for query: %s", query[:50])
        return []

    # Step 3: contextual compression
    compressed = contextual_compression(query, raw_chunks)
    return compressed[:top_k]