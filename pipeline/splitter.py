"""
pipeline/splitter.py
--------------------
Step 2 — Text splitting using SemanticChunker.

SemanticChunker splits on meaning, not fixed character count.
Two sentences that talk about the same topic stay in the same chunk.
Two sentences about different topics get split — even if they are adjacent.

Extra guards added for production:
- Chunks below MIN_CHUNK_SIZE are discarded (noise)
- Chunks above MAX_CHUNK_SIZE are hard-split (prevents token overflow in LLM)
- Chunk index added to metadata for traceability

HuggingFace change:
- OpenAIEmbeddings replaced with HuggingFaceEmbeddings
- Runs LOCALLY — no API call, no cost for this step
- Model downloaded once and cached (~90MB)
"""

from langchain_experimental.text_splitter import SemanticChunker
#from langchain_ollama import OllamaEmbeddings   # was: OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _hard_split_large_chunk(doc: Document) -> list[Document]:
    """
    If SemanticChunker produces a chunk larger than MAX_CHUNK_SIZE,
    fall back to character-based splitting on that chunk only.
    Prevents token overflow when sending to LLM.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=50,
    )
    sub_docs = splitter.create_documents(
        [doc.page_content],
        metadatas=[doc.metadata],
    )
    return sub_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of Documents into semantic chunks.
    Returns cleaned, indexed chunks ready for the Map step.

    HuggingFace embeddings run LOCALLY via sentence-transformers.
    No API call, no cost, no internet needed after first download.
    Model is cached in ~/.cache/huggingface after first run.
    """
    """
    embeddings = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    )
    """
    embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    model="text-embedding-3-small",
    )

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.CHUNK_BREAKPOINT_THRESHOLD,
    )

    try:
        raw_chunks = chunker.split_documents(docs)
    except Exception as e:
        logger.error("SemanticChunker failed: %s — falling back to character split", e)
        fallback = RecursiveCharacterTextSplitter(
            chunk_size=settings.MAX_CHUNK_SIZE, chunk_overlap=50
        )
        raw_chunks = fallback.split_documents(docs)

    # ── Clean and validate chunks ──────────────────────────────────────────
    cleaned: list[Document] = []
    discarded = 0

    for chunk in raw_chunks:
        text = chunk.page_content.strip()

        # 1. Discard empty or too-small chunks
        if len(text) < settings.MIN_CHUNK_SIZE:
            discarded += 1
            continue

        # 2. Hard-split chunks that are too large for the LLM context
        if len(text) > settings.MAX_CHUNK_SIZE:
            sub_chunks = _hard_split_large_chunk(chunk)
            cleaned.extend(sub_chunks)
            continue

        chunk.page_content = text
        cleaned.append(chunk)

    # 3. Add chunk index to metadata for traceability
    for i, chunk in enumerate(cleaned):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_count"] = len(cleaned)
        # Ensure page number is carried through for embedder
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata.get("page_number", None)

    logger.info(
        "Splitting complete — %d chunks produced, %d discarded (too small)",
        len(cleaned), discarded,
    )
    return cleaned