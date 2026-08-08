"""
pipeline/memory.py
------------------
Chat memory management using ConversationBufferWindowMemory.

Loads past N turns from chat_messages table and injects them into prompt.
This is how the chatbot remembers conversation history.

Window = last N turns only (not entire history) to avoid token overflow.
"""
from typing import Optional
from sqlalchemy.orm import Session

from db.models import ChatMessage
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# How many past turns to include in context
# 1 turn = 1 human + 1 AI message
MEMORY_WINDOW_SIZE = 5


class ConversationBufferWindowMemory:
    """Lightweight compatibility wrapper for conversation history."""

    def __init__(self, k: int = 5, return_messages: bool = True, memory_key: str = "chat_history", input_key: str = "query", output_key: str = "answer") -> None:
        self.k = k
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.chat_memory = type("ChatMemory", (), {"messages": []})()

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.chat_memory.messages.append({"type": "human", "content": inputs.get(self.input_key, "")})
        self.chat_memory.messages.append({"type": "ai", "content": outputs.get(self.output_key, "")})

    def load_memory_variables(self, _inputs: dict) -> dict:
        return {self.memory_key: self.format_messages()}

    def format_messages(self) -> str:
        if not self.chat_memory.messages:
            return ""
        parts = []
        for msg in self.chat_memory.messages[-(self.k * 2):]:
            role = "Human" if msg["type"] == "human" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)


def load_memory_from_db(thread_id: str, db: Session) -> ConversationBufferWindowMemory:
    """
    Load last N turns from chat_messages table.
    Returns a ConversationBufferWindowMemory with history injected.

    This is called on every request — memory is rebuilt from DB each time.
    This ensures consistency even if server restarts.
    """
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_SIZE,
        return_messages=True,
        memory_key="chat_history",
        input_key="query",
        output_key="answer",
    )

    try:
        # Load last N*2 messages (N turns = N human + N AI)
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(MEMORY_WINDOW_SIZE * 2)
            .all()
        )

        # Inject messages into memory in pairs (human, ai)
        i = 0
        while i < len(messages) - 1:
            human_msg = messages[i]
            ai_msg    = messages[i + 1]

            if human_msg.role == "human" and ai_msg.role == "ai":
                memory.save_context(
                    {"query": human_msg.content},
                    {"answer": ai_msg.content},
                )
                i += 2
            else:
                i += 1

        logger.info(
            "Loaded %d messages from thread %s into memory",
            len(messages), str(thread_id)[:8],
        )

    except Exception as e:
        logger.warning("Memory load failed for thread %s: %s", thread_id, e)

    return memory


def build_prompt_with_history(
    query: str,
    memory: ConversationBufferWindowMemory,
    context_chunks: list[dict],
    doc_filter: Optional[str] = None,
) -> str:
    """
    Build the full prompt string:
    - System instructions
    - Retrieved document context
    - Conversation history
    - Current query
    """
    from typing import Optional

    # System message
    system = """You are Avabodh, an intelligent document assistant.
Answer questions based strictly on the provided document context.
If the answer is not in the context, say clearly: "I don't have enough information in the documents to answer this."
Always be specific, cite which document your answer comes from.
Keep answers concise but complete."""

    # Document context from retrieved chunks
    if context_chunks:
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(
                f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}]\n"
                f"{chunk['chunk_text']}"
            )
        context_str = "\n\n".join(context_parts)
    else:
        context_str = "No relevant document context found."

    # Conversation history from memory
    history_messages = getattr(memory.chat_memory, "messages", [])
    history_str = ""
    if history_messages:
        history_parts = []
        for msg in history_messages:
            if isinstance(msg, dict):
                role = "Human" if msg.get("type") == "human" else "Assistant"
                history_parts.append(f"{role}: {msg.get('content', '')}")
        history_str = "\n".join(history_parts)

    # Build full prompt
    prompt = f"""{system}

DOCUMENT CONTEXT:
{context_str}

CONVERSATION HISTORY:
{history_str if history_str else "No previous conversation."}

CURRENT QUESTION:
{query}

ANSWER:"""

    return prompt