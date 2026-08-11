"""
pipeline/summariser.py
----------------------
Steps 3 + 4 - Map + Reduce summarisation using OpenAI.

NEW: Metadata extraction added at both levels —
- Chunk-level metadata extracted in the SAME Map call (no extra cost)
- Document-level metadata extracted in ONE extra call after Reduce

Map step:    ThreadPoolExecutor — all chunks processed simultaneously,
             each call now returns summary + structured chunk metadata
Reduce step: Single LLM call combines all chunk summaries into one,
             followed by one structured-output call for document metadata
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from db.models import ChunkMetadataOutput, DocumentMetadataOutput
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"]   = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT

# ── Existing plain-text summary prompt (kept for reduce input) ────────────────
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a professional document analyst.
Read the following section and write a concise summary.
Focus on: key facts, important decisions, main arguments, critical data.
Keep it under 150 words. Be specific.

SECTION:
{text}

CONCISE SUMMARY:""",
)

# ── NEW — Structured prompt for chunk-level metadata + summary together ───────
CHUNK_METADATA_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional document analyst extracting structured metadata.
Read the following section and return:
- A concise summary (under 150 words, specific, factual)
- The section/heading this content likely belongs to (e.g. "Introduction", "Methodology"), or null if unclear
- The chunk type: paragraph, table, list, heading, or code
- A 2-3 word topic label for this specific section
- Named entities mentioned (people, organizations, locations) — empty list if none
- Whether this section contains numerical data, statistics, or tabular data
- Your confidence (0.0-1.0) in this extraction

SECTION:
{text}"""
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a senior document analyst.
Below are summaries of individual sections of a document.
Create a structured summary using EXACTLY this format:

DOCUMENT OVERVIEW
Write 3-4 sentences covering the big picture of the entire document.
KEY FINDINGS
- Write each key finding as a bullet point
- Each bullet should be one clear sentence
- Include 5-7 most important findings only
MAIN TOPICS COVERED
- List each major topic discussed in the document
- Keep each topic to 3-5 words
- Include 4-6 topics
CONCLUSION
Write one powerful sentence that captures the overall takeaway.
Rules:
- Use exactly the headings shown above
- Do not add extra sections
- Do not write paragraphs under KEY FINDINGS or MAIN TOPICS
- Be specific, not vague
- Maximum 400 words total
SECTION SUMMARIES:
{text}

STRUCTURED SUMMARY:""",
)

# ── NEW — Document-level metadata extraction prompt ────────────────────────────
DOCUMENT_METADATA_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior document analyst extracting document-level metadata.
Based on the document summary and section summaries below, extract:
- title: the real document title (not a filename)
- author: if mentioned anywhere, else null
- document_type: one of resume, research_paper, contract, report, invoice, manual, article, other
- domain: one of legal, technical, financial, academic, medical, general
- detected_language: the language of the document
- key_entities: organizations, people, locations mentioned (max 10)
- mentioned_dates: any dates referenced (max 10)
- target_audience: one of technical, general, executive
- sentiment: one of positive, negative, neutral, critical
- confidentiality_level: one of public, internal, confidential — infer from content tone

DOCUMENT SUMMARY:
{summary}

SECTION SUMMARIES:
{sections}"""
)


def _build_llm(max_tokens: int) -> ChatOpenAI:
    if settings.use_ollama:
        if ChatOllama is not None:
            return ChatOllama(
                model=settings.OLLAMA_CHAT_MODEL,
                base_url=settings.ollama_url,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=max_tokens,
            )
        return ChatOpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_url.rstrip('/')}/v1",
            model=settings.OLLAMA_CHAT_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=max_tokens,
        )
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MAP_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )


def _summarise_chunk(args: tuple) -> tuple:
    """
    Summarise a single chunk — designed to run in a thread.
    Returns (index, summary_text) tuple.
    Kept separate from metadata extraction — used only as a fallback
    if structured extraction fails for a chunk.
    """
    index, text, llm = args
    try:
        chain = MAP_PROMPT | llm | StrOutputParser()
        summary = chain.invoke({"text": text[:3000]})
        logger.info("Chunk %d summarised", index + 1)
        return index, summary.strip()
    except Exception as e:
        logger.warning("Chunk %d failed: %s — using truncated text", index + 1, e)
        return index, text[:200]


def _extract_chunk_metadata(args: tuple) -> tuple:
    """
    NEW — Extract summary + structured metadata in ONE LLM call per chunk.
    Uses with_structured_output so Pydantic validates the shape directly.
    Runs in a thread alongside other chunks — same parallelism as before.

    Returns (index, ChunkMetadataOutput | None)
    """
    index, text, structured_llm = args
    try:
        chain = CHUNK_METADATA_PROMPT | structured_llm
        result: ChunkMetadataOutput = chain.invoke({"text": text[:3000]})
        logger.info("Chunk %d metadata extracted (topic=%s, type=%s)",
                   index + 1, result.topic, result.chunk_type)
        return index, result
    except Exception as e:
        logger.warning("Chunk %d metadata extraction failed: %s", index + 1, e)
        return index, None


@traceable(run_type="chain", name="MapReduce Summarisation - OpenAI Parallel")
def summarise_document(chunks: list[Document], doc_name: str = "document") -> dict:
    """
    Parallel Map-Reduce summarisation + metadata extraction.

    Map step:    All chunks sent to OpenAI simultaneously.
                 Each call now returns BOTH summary text AND structured
                 chunk metadata (section_heading, chunk_type, topic, entities)
                 in the SAME call — no extra API cost vs the old version.

    Reduce step: Single LLM call combines all chunk summaries into the
                 final structured summary, followed by ONE additional
                 structured-output call that extracts document-level
                 metadata (title, author, document_type, domain, etc).

    Returns dict with summary_text + document metadata + per-chunk metadata
    (chunk_metadata list, same order as input chunks, None for failed ones).
    """
    if not chunks:
        raise ValueError(f"No chunks provided for '{doc_name}'")

    logger.info(
        "Starting PARALLEL summarisation + metadata extraction for '%s' - %d chunks | model=%s",
        doc_name, len(chunks), settings.MAP_MODEL,
    )

    start = time.time()
    base_llm = _build_llm(settings.MAP_MAX_TOKENS)
    # Structured output LLM — forces Pydantic-validated ChunkMetadataOutput shape
    structured_llm = base_llm.with_structured_output(ChunkMetadataOutput)

    # ── Step 3: MAP — parallel processing with metadata ────────────────────
    chunk_args = [
        (i, chunk.page_content.strip(), structured_llm)
        for i, chunk in enumerate(chunks)
        if chunk.page_content.strip()
    ]

    chunk_summaries = [""] * len(chunk_args)
    chunk_metadata: list = [None] * len(chunk_args)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_extract_chunk_metadata, args): args[0]
            for args in chunk_args
        }
        completed = 0
        for future in as_completed(futures):
            index, metadata_result = future.result()
            if metadata_result is not None:
                chunk_summaries[index] = metadata_result.summary
                chunk_metadata[index] = metadata_result
            else:
                # Fallback to plain summarisation if structured extraction failed
                _, text, _ = chunk_args[index]
                fallback_idx, fallback_summary = _summarise_chunk(
                    (index, text, base_llm)
                )
                chunk_summaries[fallback_idx] = fallback_summary
            completed += 1
            logger.info("Map progress: %d/%d chunks done", completed, len(chunk_args))

    # Remove empty summaries (keep metadata aligned by filtering both together)
    paired = [(s, m) for s, m in zip(chunk_summaries, chunk_metadata) if s]
    chunk_summaries = [s for s, _ in paired]
    chunk_metadata  = [m for _, m in paired]

    if not chunk_summaries:
        raise RuntimeError(f"All chunks failed for '{doc_name}'")

    map_time = round(time.time() - start, 2)
    logger.info(
        "Map step complete in %.2fs — %d chunk summaries + metadata ready",
        map_time, len(chunk_summaries),
    )

    # ── Step 4: REDUCE — single call ─────────────────────────────────────
    reduce_llm = _build_llm(settings.REDUCE_MAX_TOKENS)
    combined = "\n\n".join(chunk_summaries)

    if len(combined) > 12000:
        logger.info("Combined text too long (%d chars) - truncating", len(combined))
        combined = combined[:12000]

    reduce_chain = COMBINE_PROMPT | reduce_llm | StrOutputParser()

    try:
        final_summary = reduce_chain.invoke({"text": combined}).strip()
    except Exception as e:
        logger.error("Reduce step failed: %s", e)
        raise RuntimeError(f"Reduce step failed: {e}")

    # ── NEW — Step 5: Document-level metadata extraction (ONE extra call) ──
    document_metadata = None
    try:
        metadata_llm = base_llm.with_structured_output(DocumentMetadataOutput)
        metadata_chain = DOCUMENT_METADATA_PROMPT | metadata_llm
        document_metadata = metadata_chain.invoke({
            "summary":  final_summary,
            "sections": combined[:8000],   # cap to control token cost
        })
        logger.info(
            "Document metadata extracted: type=%s domain=%s title=%s",
            document_metadata.document_type, document_metadata.domain,
            document_metadata.title[:50] if document_metadata.title else "(none)",
        )
    except Exception as e:
        logger.warning("Document metadata extraction failed (non-fatal): %s", e)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Summarisation + metadata complete for '%s' | chunks=%d | map=%.2fs | total=%.2fs",
        doc_name, len(chunks), map_time, elapsed,
    )

    return {
        "summary_text":      final_summary,
        "map_model":         settings.MAP_MODEL,
        "reduce_model":      settings.REDUCE_MODEL,
        "chunk_count":       len(chunks),
        "elapsed_sec":       elapsed,
        "chunk_metadata":    chunk_metadata,     # list aligned with final chunk order
        "document_metadata": document_metadata,  # DocumentMetadataOutput | None
    }
