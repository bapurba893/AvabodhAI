"""
pipeline/chat_storage.py
------------------------
Saves chat messages and threads to PostgreSQL.
Both human and AI messages saved as separate rows.
Message embeddings generated and stored for semantic search.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm import make_transient
from langchain_openai import OpenAIEmbeddings

from db.models import ChatThread, ChatMessage
from db.database import get_db_session_context
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _embed_text(text: str) -> Optional[list[float]]:
    """Generate embedding for a message."""
    try:
        client = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
        return client.embed_query(text)
    except Exception as e:
        logger.warning("Message embedding failed: %s", e)
        return None


def create_thread(
    title: Optional[str] = None,
    user_id: Optional[str] = None,
    doc_filter: Optional[str] = None,
) -> ChatThread:
    """Create a new chat thread and return it."""
    with get_db_session_context() as session:
        thread = ChatThread(
            title     = title,
            user_id   = user_id,
            doc_filter= doc_filter,
        )
        session.add(thread)
        session.flush()
        session.expunge(thread)
        make_transient(thread)
        logger.info("Created thread: %s", thread.id)
        return thread


def update_thread_title(thread_id: str, title: str) -> None:
    """Update thread title — called after first message."""
    with get_db_session_context() as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id)
        ).first()
        if thread:
            thread.title = title
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)
            logger.info("Updated thread title: %s -> %s", thread_id[:8], title)


def increment_message_count(thread_id: str) -> None:
    """Increment message counter on thread."""
    with get_db_session_context() as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id)
        ).first()
        if thread:
            thread.message_count = (thread.message_count or 0) + 1
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)


def save_human_message(
    thread_id: str,
    content: str,
) -> ChatMessage:
    """
    Save human message as its own row.
    Generates and stores embedding for semantic search.
    """
    embedding = _embed_text(content)

    with get_db_session_context() as session:
        msg = ChatMessage(
            thread_id       = uuid.UUID(thread_id),
            role            = "human",
            content         = content,
            embedding       = embedding,
            embedding_model = settings.EMBEDDING_MODEL if embedding else None,
        )
        session.add(msg)
        session.flush()
        session.expunge(msg)
        make_transient(msg)
        logger.info("Saved human message to thread %s", thread_id[:8])
        return msg


def save_ai_message(
    thread_id: str,
    content: str,
    sources: Optional[list] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> ChatMessage:
    """
    Save AI message as its own row — separate from human message.
    Stores sources (which chunks were used) and token usage.
    Generates and stores embedding.
    """
    embedding = _embed_text(content)

    with get_db_session_context() as session:
        msg = ChatMessage(
            thread_id         = uuid.UUID(thread_id),
            role              = "ai",
            content           = content,
            embedding         = embedding,
            embedding_model   = settings.EMBEDDING_MODEL if embedding else None,
            sources           = sources or [],
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
        )
        session.add(msg)
        session.flush()
        session.expunge(msg)
        make_transient(msg)
        logger.info("Saved AI message to thread %s | sources=%d",
                   thread_id[:8], len(sources or []))
        return msg


def get_thread(thread_id: str, db: Session) -> Optional[ChatThread]:
    """Get thread by ID."""
    return db.query(ChatThread).filter(
        ChatThread.id == uuid.UUID(thread_id)
    ).first()


def get_thread_messages(thread_id: str, db: Session) -> list[ChatMessage]:
    """Get all messages for a thread ordered by time."""
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.thread_id == uuid.UUID(thread_id))
        .order_by(ChatMessage.created_at.asc())
        .all()
    )