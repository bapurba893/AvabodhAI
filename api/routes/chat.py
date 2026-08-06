"""
api/routes/chat.py
------------------
All chat endpoints.

Multimodal chat:
  POST /chat/message accepts an optional image (base64) alongside the query.

  When image is attached — TWO parallel paths:
    Path 1: GPT-4o Vision captions the image → caption embedded →
            pgvector similarity search finds related document chunks
    Path 2: Final LLM call uses GPT-4o with BOTH the image bytes AND
            the retrieved context — GPT-4o literally sees the pixels
            and answers using document knowledge simultaneously

Multi-tenancy + department isolation: every endpoint takes tenant_id AND
org_unit_id from the X-Tenant-ID / X-Org-Unit-ID headers. Thread
creation/lookup, message storage, and document retrieval are all scoped
by BOTH together — a chat can never surface another tenant's, or another
department's, documents or thread history.
"""

import base64
import uuid
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from openai import OpenAI

from api.dependencies import get_tenant_id, get_org_unit_id
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
from pipeline.retriever import retrieve, embed_query, vector_search
from pipeline.memory import load_memory_from_db, build_prompt_with_history
from pipeline.chat import chat_complete, chat_stream, generate_thread_title
from pipeline.chat_storage import (
    create_thread, update_thread_title, increment_message_count,
    save_human_message, save_ai_message,
    get_thread, get_thread_messages,
)
from pipeline.image_processor import (
    caption_image_with_vision,
    build_image_embedding_text,
    should_skip_image,
)
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

VISION_MODEL = "gpt-4o"


# ─────────────────────────────────────────────────────────────────────────────
# Image-aware retrieval helper
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve_with_image(
    query:         str,
    image_base64:  str,
    image_media_type: str,
    db:            Session,
    tenant_id:     str,
    org_unit_id:   str,
    top_k:         int = 5,
    doc_filter:    Optional[str] = None,
) -> tuple[list[dict], Optional[str]]:
    """
    When user attaches an image:
    1. Caption it with GPT-4o Vision to understand what it shows
    2. Build composite embedding text from caption
    3. Use that text to search pgvector (finds both text AND image chunks) —
       scoped to tenant_id + org_unit_id like every other retrieval path
    Returns (chunks, image_caption_text)
    """
    try:
        # Decode base64 to bytes for Vision API
        image_bytes = base64.b64decode(image_base64)

        # Caption the user's image
        caption = caption_image_with_vision(
            image_bytes=image_bytes,
            surrounding_text=query,
        )

        if caption is None:
            logger.warning("Vision captioning of user image failed — falling back to text search")
            return retrieve(query=query, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
                           top_k=top_k, doc_filter=doc_filter), None

        # Build rich search text from caption
        caption_search_text = build_image_embedding_text(
            caption=caption,
            doc_name="user_query",
            surrounding_text=query,
        )

        # Embed caption text and search
        query_vector = embed_query(caption_search_text)
        chunks = vector_search(
            query_vector=query_vector,
            db=db,
            tenant_id=tenant_id,
            org_unit_id=org_unit_id,
            top_k=top_k * 2,
            doc_filter=doc_filter,
        )

        # Also search with the original text query and merge results
        text_chunks = retrieve(query=query, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
                              top_k=top_k, doc_filter=doc_filter)

        # Merge — deduplicate by chunk id, keep highest similarity
        seen_ids = set()
        merged = []
        for chunk in chunks + text_chunks:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

        # Sort by similarity and take top_k
        merged.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        logger.info(
            "Image-aware retrieval: caption type=%s, found %d merged chunks",
            caption.image_type, len(merged[:top_k]),
        )

        return merged[:top_k], caption.caption

    except Exception as e:
        logger.warning("Image-aware retrieval failed — falling back to text: %s", e)
        return retrieve(query=query, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
                       top_k=top_k, doc_filter=doc_filter), None


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal LLM call — GPT-4o sees image + retrieved context
# ─────────────────────────────────────────────────────────────────────────────

def _chat_complete_with_image(
    prompt:           str,
    image_base64:     str,
    image_media_type: str,
) -> dict:
    """
    Call GPT-4o with both the image bytes AND the text prompt.
    GPT-4o literally sees the image pixels alongside the document context.
    Returns {"content": str, "prompt_tokens": int, "completion_tokens": int}
    """
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1500,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:{image_media_type};base64,{image_base64}",
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

        return {
            "content":           response.choices[0].message.content,
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        logger.error("Multimodal GPT-4o call failed: %s", e)
        raise RuntimeError(f"Multimodal LLM call failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared chat preparation
# ─────────────────────────────────────────────────────────────────────────────

async def _prepare_chat(request: ChatMessageRequest, tenant_id: str, org_unit_id: str, db: Session) -> tuple:
    is_new_thread = False
    if request.thread_id is None:
        thread = create_thread(tenant_id=tenant_id, org_unit_id=org_unit_id, doc_filter=request.doc_filter)
        thread_id = str(thread.id)
        is_new_thread = True
    else:
        thread_id = str(request.thread_id)
        # Scoped by tenant_id + org_unit_id — a thread_id belonging to
        # another tenant OR another department returns None here exactly
        # like a nonexistent one.
        thread = get_thread(thread_id, tenant_id, org_unit_id, db)
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")

    memory = load_memory_from_db(thread_id, tenant_id, org_unit_id, db)

    # ── Retrieval — image-aware if image attached, always scoped ───────────
    if request.image_base64:
        chunks, image_caption = _retrieve_with_image(
            query            = request.query,
            image_base64     = request.image_base64,
            image_media_type = request.image_media_type or "image/jpeg",
            db               = db,
            tenant_id        = tenant_id,
            org_unit_id      = org_unit_id,
            top_k            = request.top_k,
            doc_filter       = request.doc_filter,
        )
    else:
        chunks = retrieve(
            query=request.query, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
            top_k=request.top_k, doc_filter=request.doc_filter,
        )
        image_caption = None

    prompt = build_prompt_with_history(
        query          = request.query,
        memory         = memory,
        context_chunks = chunks,
        doc_filter     = request.doc_filter,
    )

    return thread_id, is_new_thread, memory, chunks, prompt, image_caption


def _build_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_name":    c["doc_name"],
            "chunk_index": c["chunk_index"],
            "chunk_text":  c["chunk_text"][:200],
            "similarity":  c.get("similarity"),
            "role":        c.get("role", "text"),
            "image_type":  c.get("image_type"),
            "image_url":   c.get("image_url"),
        }
        for c in chunks
    ]


async def _save_turn(
    thread_id: str, tenant_id: str, org_unit_id: str, is_new_thread: bool,
    query: str, answer: str, sources: list,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    image_caption: Optional[str] = None,
) -> Optional[str]:
    save_human_message(thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id, content=query)
    save_ai_message(
        thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id,
        content=answer, sources=sources,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        has_image=image_caption is not None, image_caption=image_caption,
    )
    increment_message_count(thread_id, tenant_id, org_unit_id)

    thread_title = None
    if is_new_thread:
        thread_title = generate_thread_title(query)
        update_thread_title(thread_id, tenant_id, org_unit_id, thread_title)
    return thread_title


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/message — full JSON response (multimodal)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    response_model=ChatMessageResponse,
    status_code=200,
    summary="Send a message and get full JSON response",
    description=(
        "Send a text query, optionally with an image (base64). "
        "When image is provided, GPT-4o Vision sees the actual image pixels "
        "alongside retrieved document context — dual-path: "
        "similarity search finds related knowledge, GPT-4o answers about the image."
    ),
)
async def send_message(
    request: ChatMessageRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread_id, is_new_thread, memory, chunks, prompt, image_caption = await _prepare_chat(
        request, tenant_id, org_unit_id, db
    )

    # Fallback when no chunks found
    if not chunks:
        fallback_msg = (
            "I can see the image you've shared, but I don't have relevant information "
            "in the knowledge base to answer your question about it."
            if request.image_base64
            else "I don't have relevant information in the documents to answer your question."
        )
        await _save_turn(
            thread_id, tenant_id, org_unit_id, is_new_thread, request.query, fallback_msg, [],
            image_caption=image_caption,
        )
        return ChatMessageResponse(
            message_id   = uuid.uuid4(),
            thread_id    = uuid.UUID(thread_id),
            role         = "ai",
            content      = fallback_msg,
            sources      = [],
            thread_title = generate_thread_title(request.query) if is_new_thread else None,
            created_at   = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            image_understood = bool(request.image_base64),
            image_caption = image_caption,
        )

    # ── LLM call — multimodal if image attached, text-only otherwise ──────
    if request.image_base64:
        logger.info("Multimodal chat: image + text query, thread=%s", thread_id[:8])
        result = _chat_complete_with_image(
            prompt           = prompt,
            image_base64     = request.image_base64,
            image_media_type = request.image_media_type or "image/jpeg",
        )
        image_understood = True
    else:
        result = chat_complete(prompt)
        image_understood = False

    answer  = result["content"]
    sources = _build_sources(chunks)

    thread_title = await _save_turn(
        thread_id=thread_id, tenant_id=tenant_id, org_unit_id=org_unit_id, is_new_thread=is_new_thread,
        query=request.query, answer=answer, sources=sources,
        prompt_tokens=result.get("prompt_tokens"),
        completion_tokens=result.get("completion_tokens"),
        image_caption=image_caption,
    )

    return ChatMessageResponse(
        message_id       = uuid.uuid4(),
        thread_id        = uuid.UUID(thread_id),
        role             = "ai",
        content          = answer,
        sources          = [SourceReference(**s) for s in sources],
        thread_title     = thread_title,
        created_at       = __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        image_understood = image_understood,
        image_caption    = image_caption,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thread CRUD — all tenant + org-unit scoped
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/threads", response_model=ThreadResponse, status_code=201, summary="Create a new thread")
async def create_thread_endpoint(
    request: ThreadCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = create_thread(
        tenant_id=tenant_id, org_unit_id=org_unit_id, title=request.title,
        user_id=request.user_id, doc_filter=request.doc_filter,
    )
    return ThreadResponse(
        id=thread.id, title=thread.title, user_id=thread.user_id,
        doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
        created_at=thread.created_at, updated_at=thread.updated_at,
        tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id,
    )


@router.get("/threads", response_model=ThreadListResponse, summary="List all threads")
async def list_threads(
    page: int = Query(default=1, ge=1), per_page: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    offset = (page - 1) * per_page
    base_query = db.query(ChatThread).filter(
        ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    )
    total  = base_query.count()
    threads = base_query.order_by(ChatThread.updated_at.desc()).offset(offset).limit(per_page).all()
    return ThreadListResponse(total=total, threads=[
        ThreadResponse(id=t.id, title=t.title, user_id=t.user_id, doc_filter=t.doc_filter,
                       message_count=t.message_count or 0, created_at=t.created_at,
                       updated_at=t.updated_at, tenant_id=t.tenant_id, org_unit_id=t.org_unit_id)
        for t in threads
    ])


@router.get("/threads/{thread_id}", response_model=ThreadResponse, summary="Get thread by ID")
async def get_thread_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    return ThreadResponse(id=thread.id, title=thread.title, user_id=thread.user_id,
                          doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
                          created_at=thread.created_at, updated_at=thread.updated_at,
                          tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id)


@router.patch("/threads/{thread_id}", response_model=ThreadResponse, summary="Update thread title")
async def update_thread_endpoint(
    thread_id: uuid.UUID, request: ThreadUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    thread.title = request.title
    db.commit()
    db.refresh(thread)
    return ThreadResponse(id=thread.id, title=thread.title, user_id=thread.user_id,
                          doc_filter=thread.doc_filter, message_count=thread.message_count or 0,
                          created_at=thread.created_at, updated_at=thread.updated_at,
                          tenant_id=thread.tenant_id, org_unit_id=thread.org_unit_id)


@router.delete("/threads/{thread_id}", response_model=DeleteResponse, summary="Delete thread and all its messages")
async def delete_thread_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    # chat_messages cascade-deletes automatically via ON DELETE CASCADE.
    db.delete(thread)
    db.commit()
    return DeleteResponse(id=thread_id)


@router.get("/threads/{thread_id}/messages", summary="Get all messages in a thread")
async def get_messages_endpoint(
    thread_id: uuid.UUID,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    thread = db.query(ChatThread).filter(
        ChatThread.id == thread_id, ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.thread_id == thread_id,
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.org_unit_id == org_unit_id,
        )
        .order_by(ChatMessage.created_at.asc()).all()
    )
    return {
        "thread_id": str(thread_id), "thread_title": thread.title, "total": len(messages),
        "messages": [
            {"id": str(m.id), "role": m.role, "content": m.content,
             "sources": m.sources or [], "created_at": str(m.created_at)}
            for m in messages
        ]
    }


@router.get("/search", summary="Semantic search across chat history")
async def search_chat_history(
    query: str = Query(..., min_length=1), top_k: int = Query(default=5, ge=1, le=20),
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
    db: Session = Depends(get_db_session_fastapi),
):
    query_vector = embed_query(query)
    try:
        results = db.execute(
            sql_text("""
                SELECT m.id, m.role, m.content, m.created_at,
                       t.title as thread_title, t.id as thread_id,
                       1 - (m.embedding <=> CAST(:vector AS vector)) AS similarity
                FROM chat_messages m
                JOIN chat_threads t ON m.thread_id = t.id
                WHERE m.embedding IS NOT NULL
                  AND m.tenant_id = :tenant_id
                  AND m.org_unit_id = :org_unit_id
                ORDER BY m.embedding <=> CAST(:vector AS vector)
                LIMIT :top_k
            """),
            {"vector": str(query_vector), "top_k": top_k, "tenant_id": tenant_id, "org_unit_id": org_unit_id}
        ).fetchall()
        return {
            "query": query, "total": len(results),
            "results": [
                {"id": str(r.id), "role": r.role, "content": r.content[:300],
                 "thread_id": str(r.thread_id), "thread_title": r.thread_title,
                 "similarity": round(float(r.similarity), 4), "created_at": str(r.created_at)}
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")