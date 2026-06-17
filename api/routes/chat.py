"""
api/routes/chat.py
------------------
All chat endpoints:

POST   /chat/message          — send message, get full JSON response
GET    /chat/stream            — send message, get SSE token stream
POST   /chat/threads           — create thread manually
GET    /chat/threads           — list all threads
GET    /chat/threads/{id}      — get one thread
PATCH  /chat/threads/{id}      — update thread title
DELETE /chat/threads/{id}      — delete thread + all messages
GET    /chat/threads/{id}/messages — get all messages in thread
"""

import uuid
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SourceReference,
    ThreadCreateRequest,
    ThreadUpdateRequest,
    ThreadResponse,
    ThreadListResponse,
    MessageListResponse,
    DeleteResponse,
)
from db.database import get_db_session_fastapi
from db.models import ChatThread, ChatMessage
from pipeline.retriever import retrieve
from pipeline.memory import load_memory_from_db, build_prompt_with_history
from pipeline.chat import chat_complete, chat_stream, generate_thread_title
from pipeline.chat_storage import (
    create_thread, update_thread_title, increment_message_count,
    save_human_message, save_ai_message,
    get_thread, get_thread_messages,
)
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — shared logic for both streaming and non-streaming
# ─────────────────────────────────────────────────────────────────────────────

async def _prepare_chat(
    request: ChatMessageRequest,
    db: Session,
) -> tuple:
    """
    Steps 1-5 shared between /message and /stream:
    1. Thread create/load
    2. Load memory
    3. Retrieve chunks
    4. Build prompt
    Returns (thread_id, is_new_thread, memory, chunks, prompt)
    """
    # ── Thread handling ───────────────────────────────────────────────────
    is_new_thread = False
    if request.thread_id is None:
        thread = create_thread(doc_filter=request.doc_filter)
        thread_id = str(thread.id)
        is_new_thread = True
        logger.info("New thread created: %s", thread_id[:8])
    else:
        thread_id = str(request.thread_id)
        thread = get_thread(thread_id, db)
        if not thread:
            raise HTTPException(status_code=404,
                                detail=f"Thread '{thread_id}' not found")

    # ── Load memory from DB ───────────────────────────────────────────────
    memory = load_memory_from_db(thread_id, db)

    # ── Retrieve relevant chunks ──────────────────────────────────────────
    chunks = retrieve(
        query      = request.query,
        db         = db,
        top_k      = request.top_k,
        doc_filter = request.doc_filter,
    )

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = build_prompt_with_history(
        query         = request.query,
        memory        = memory,
        context_chunks= chunks,
        doc_filter    = request.doc_filter,
    )

    return thread_id, is_new_thread, memory, chunks, prompt


def _build_sources(chunks: list[dict]) -> list[dict]:
    """Build source reference list from retrieved chunks."""
    return [
        {
            "doc_name":    c["doc_name"],
            "chunk_index": c["chunk_index"],
            "chunk_text":  c["chunk_text"][:200],  # preview only
            "similarity":  c.get("similarity"),
        }
        for c in chunks
    ]


async def _save_turn(
    thread_id: str,
    is_new_thread: bool,
    query: str,
    answer: str,
    sources: list,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Save both human and AI messages to DB.
    Generate thread title if new thread.
    Returns thread title (if generated).
    """
    # Save human message (own row)
    save_human_message(thread_id=thread_id, content=query)

    # Save AI message (own row)
    save_ai_message(
        thread_id         = thread_id,
        content           = answer,
        sources           = sources,
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
    )

    # Increment message count
    increment_message_count(thread_id)

    # Auto-generate title for new thread
    thread_title = None
    if is_new_thread:
        thread_title = generate_thread_title(query)
        update_thread_title(thread_id, thread_title)
        logger.info("Thread title set: %s", thread_title)

    return thread_title


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/message — full JSON response
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=200,
    summary="Send a message and get full JSON response",
)
async def send_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    """
    Send a message and receive the complete AI response as JSON.
    Use this for Postman testing or programmatic access.
    Use /chat/stream for real-time word-by-word streaming.
    """
    thread_id, is_new_thread, memory, chunks, prompt = await _prepare_chat(request, db)

    # Handle fallback when no chunks found
    if not chunks:
        fallback = "I don't have relevant information in the documents to answer your question."
        await _save_turn(thread_id, is_new_thread, request.query, fallback, [])
        return ChatMessageResponse(
            message_id   = uuid.uuid4(),
            thread_id    = uuid.UUID(thread_id),
            role         = "ai",
            content      = fallback,
            sources      = [],
            thread_title = generate_thread_title(request.query) if is_new_thread else None,
            created_at   = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )

    # LLM call — non-streaming
    result = chat_complete(prompt)
    answer = result["content"]

    # Pydantic output validation
    sources = _build_sources(chunks)
    thread_title = await _save_turn(
        thread_id         = thread_id,
        is_new_thread     = is_new_thread,
        query             = request.query,
        answer            = answer,
        sources           = sources,
        prompt_tokens     = result.get("prompt_tokens"),
        completion_tokens = result.get("completion_tokens"),
    )

    # Build validated response
    return ChatMessageResponse(
        message_id   = uuid.uuid4(),
        thread_id    = uuid.UUID(thread_id),
        role         = "ai",
        content      = answer,
        sources      = [SourceReference(**s) for s in sources],
        thread_title = thread_title,
        created_at   = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thread CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/threads", response_model=ThreadResponse, status_code=201,
             summary="Create a new thread")
async def create_thread_endpoint(
    request: ThreadCreateRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    thread = create_thread(
        title      = request.title,
        user_id    = request.user_id,
        doc_filter = request.doc_filter,
    )
    return ThreadResponse(
        id            = thread.id,
        title         = thread.title,
        user_id       = thread.user_id,
        doc_filter    = thread.doc_filter,
        message_count = thread.message_count or 0,
        created_at    = thread.created_at,
        updated_at    = thread.updated_at,
    )


@router.get("/threads", response_model=ThreadListResponse, summary="List all threads")
async def list_threads(
    page:     int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    total = db.query(ChatThread).count()
    threads = (
        db.query(ChatThread)
        .order_by(ChatThread.updated_at.desc())
        .offset(offset).limit(per_page).all()
    )
    return ThreadListResponse(
        total=total,
        threads=[
            ThreadResponse(
                id=t.id, title=t.title, user_id=t.user_id,
                doc_filter=t.doc_filter,
                message_count=t.message_count or 0,
                created_at=t.created_at, updated_at=t.updated_at,
            )
            for t in threads
        ]
    )


@router.get("/threads/{thread_id}", response_model=ThreadResponse,
            summary="Get thread by ID")
async def get_thread_endpoint(
    thread_id: uuid.UUID,
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    return ThreadResponse(
        id=thread.id, title=thread.title, user_id=thread.user_id,
        doc_filter=thread.doc_filter,
        message_count=thread.message_count or 0,
        created_at=thread.created_at, updated_at=thread.updated_at,
    )


@router.patch("/threads/{thread_id}", response_model=ThreadResponse,
              summary="Update thread title")
async def update_thread_endpoint(
    thread_id: uuid.UUID,
    request: ThreadUpdateRequest,
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    thread.title = request.title
    db.commit()
    db.refresh(thread)
    return ThreadResponse(
        id=thread.id, title=thread.title, user_id=thread.user_id,
        doc_filter=thread.doc_filter,
        message_count=thread.message_count or 0,
        created_at=thread.created_at, updated_at=thread.updated_at,
    )


@router.delete("/threads/{thread_id}", response_model=DeleteResponse,
               summary="Delete thread and all its messages")
async def delete_thread_endpoint(
    thread_id: uuid.UUID,
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    db.delete(thread)
    db.commit()
    return DeleteResponse(id=thread_id)


@router.get("/threads/{thread_id}/messages", summary="Get all messages in a thread")
async def get_messages_endpoint(
    thread_id: uuid.UUID,
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return {
        "thread_id":    str(thread_id),
        "thread_title": thread.title,
        "total":        len(messages),
        "messages": [
            {
                "id":         str(m.id),
                "role":       m.role,
                "content":    m.content,
                "sources":    m.sources or [],
                "created_at": str(m.created_at),
            }
            for m in messages
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /chat/search — semantic search across chat history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/search", summary="Semantic search across chat history")
async def search_chat_history(
    query:     str = Query(..., min_length=1, description="Search query"),
    top_k:     int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db_session_fastapi),
):
    """
    Find similar messages from chat history using message embeddings.
    Useful for finding previously asked questions.
    """
    from pipeline.retriever import embed_query
    from sqlalchemy import text

    query_vector = embed_query(query)

    try:
        results = db.execute(
            text("""
                SELECT m.id, m.role, m.content, m.created_at,
                       t.title as thread_title, t.id as thread_id,
                       1 - (m.embedding <=> CAST(:vector AS vector)) AS similarity
                FROM chat_messages m
                JOIN chat_threads t ON m.thread_id = t.id
                WHERE m.embedding IS NOT NULL
                ORDER BY m.embedding <=> CAST(:vector AS vector)
                LIMIT :top_k
            """),
            {"vector": str(query_vector), "top_k": top_k}
        ).fetchall()

        return {
            "query": query,
            "total": len(results),
            "results": [
                {
                    "id":           str(r.id),
                    "role":         r.role,
                    "content":      r.content[:300],
                    "thread_id":    str(r.thread_id),
                    "thread_title": r.thread_title,
                    "similarity":   round(float(r.similarity), 4),
                    "created_at":   str(r.created_at),
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")