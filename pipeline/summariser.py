"""
pipeline/summariser.py
----------------------
Steps 3 + 4 - Map + Reduce summarisation using OpenAI.

Uses parallel processing for Map step — all chunks sent to OpenAI
simultaneously instead of one by one. This gives ~10x speed improvement.

Map step:    RunnableParallel — all chunks processed at same time
Reduce step: Single LLM call — combines all chunk summaries into one
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"]   = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]    = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT

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


def _build_llm(max_tokens: int) -> ChatOpenAI:
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


@traceable(run_type="chain", name="MapReduce Summarisation - OpenAI Parallel")
def summarise_document(chunks: list[Document], doc_name: str = "document") -> dict:
    """
    Parallel Map-Reduce summarisation.

    Map step:    All chunks sent to OpenAI simultaneously via ThreadPoolExecutor
                 Instead of 40 sequential calls, all 40 run at the same time
    Reduce step: Single LLM call combines all chunk summaries

    Speed: ~10x faster than sequential processing
    """
    if not chunks:
        raise ValueError(f"No chunks provided for '{doc_name}'")

    logger.info(
        "Starting PARALLEL summarisation for '%s' - %d chunks | model=%s",
        doc_name, len(chunks), settings.MAP_MODEL,
    )

    start = time.time()
    llm = _build_llm(settings.MAP_MAX_TOKENS)

    # ── Step 3: MAP — parallel processing ────────────────────────────────
    # Prepare args for each chunk
    chunk_args = [
        (i, chunk.page_content.strip(), llm)
        for i, chunk in enumerate(chunks)
        if chunk.page_content.strip()
    ]

    # ThreadPoolExecutor sends all chunks simultaneously
    # max_workers=10 means 10 chunks processed at same time
    # OpenAI rate limit is 500 RPM on most plans — 10 parallel is safe
    chunk_summaries = [""] * len(chunk_args)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_summarise_chunk, args): args[0]
            for args in chunk_args
        }
        completed = 0
        for future in as_completed(futures):
            index, summary = future.result()
            chunk_summaries[index] = summary
            completed += 1
            logger.info("Map progress: %d/%d chunks done", completed, len(chunk_args))

    # Remove empty summaries
    chunk_summaries = [s for s in chunk_summaries if s]

    if not chunk_summaries:
        raise RuntimeError(f"All chunks failed for '{doc_name}'")

    map_time = round(time.time() - start, 2)
    logger.info(
        "Map step complete in %.2fs — %d chunk summaries ready",
        map_time, len(chunk_summaries),
    )

    # ── Step 4: REDUCE — single call ─────────────────────────────────────
    reduce_llm = _build_llm(settings.REDUCE_MAX_TOKENS)
    combined = "\n\n".join(chunk_summaries)

    # If combined text too long, truncate to fit context window
    if len(combined) > 12000:
        logger.info("Combined text too long (%d chars) - truncating", len(combined))
        combined = combined[:12000]

    reduce_chain = COMBINE_PROMPT | reduce_llm | StrOutputParser()

    try:
        final_summary = reduce_chain.invoke({"text": combined}).strip()
    except Exception as e:
        logger.error("Reduce step failed: %s", e)
        raise RuntimeError(f"Reduce step failed: {e}")

    elapsed = round(time.time() - start, 2)
    logger.info(
        "Summarisation complete for '%s' | chunks=%d | map=%.2fs | total=%.2fs",
        doc_name, len(chunks), map_time, elapsed,
    )

    return {
        "summary_text": final_summary,
        "map_model":    settings.MAP_MODEL,
        "reduce_model": settings.REDUCE_MODEL,
        "chunk_count":  len(chunks),
        "elapsed_sec":  elapsed,
    }