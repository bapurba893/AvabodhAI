import os
import uuid
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:  # Raised with a clear message only if local mode is used.
    ChatOllama = None
    OllamaEmbeddings = None
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ConversationBufferWindowMemory:
    """Lightweight compatibility wrapper for chat history in KB flows."""

    def __init__(self, k: int = 5, memory_key: str = "chat_history", return_messages: bool = False) -> None:
        self.k = k
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = type("ChatMemory", (), {"messages": []})()

    def add_user_message(self, message: str) -> None:
        self.chat_memory.messages.append({"type": "human", "content": message})

    def add_ai_message(self, message: str) -> None:
        self.chat_memory.messages.append({"type": "ai", "content": message})

    def load_memory_variables(self, _inputs: dict) -> dict:
        messages = self.chat_memory.messages[-(self.k * 2):]
        if self.return_messages:
            return {self.memory_key: messages}
        return {self.memory_key: "\n".join(
            f"Human: {msg['content']}" if msg["type"] == "human" else f"Assistant: {msg['content']}"
            for msg in messages
        )}

from config.settings import get_settings
from db.models import KBDocument, KBChatHistory
from pipeline.loader import load_single_document

logger = logging.getLogger(__name__)
settings = get_settings()


def _embeddings_client():
    """Return OpenAI embeddings in production or a local Ollama client in demo mode."""
    if settings.use_ollama:
        if OllamaEmbeddings is None:
            raise RuntimeError("Ollama mode requires langchain-ollama. Install requirements.txt.")
        return OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.ollama_url,
        )
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)


def _chat_model(temperature: float = 0.0):
    """Select Ollama automatically when OPENAI_API_KEY is absent."""
    if settings.use_ollama:
        if ChatOllama is None:
            raise RuntimeError("Ollama mode requires langchain-ollama. Install requirements.txt.")
        return ChatOllama(
            model=settings.OLLAMA_CHAT_MODEL,
            base_url=settings.ollama_url,
            temperature=temperature,
        )
    return ChatOpenAI(model=settings.MAP_MODEL, temperature=temperature, api_key=settings.OPENAI_API_KEY)


def _vector_size() -> int:
    return settings.OLLAMA_EMBEDDING_DIMENSIONS if settings.use_ollama else settings.EMBEDDING_DIMENSIONS


async def ingest_document(file_path: str, owner_id: str, document_id: str, db: AsyncSession) -> None:
    """
    Ingests an uploaded document into the PGVector store.
    Splits using SemanticChunker and updates DB status to ready.
    """
    try:
        # Load the document
        docs = load_single_document(file_path)
        if not docs:
            raise ValueError("No text extracted from document.")

        # Initialize embeddings & chunker
        embeddings = _embeddings_client()
        text_splitter = SemanticChunker(embeddings)

        # Split documents
        chunks = text_splitter.split_documents(docs)

        # Inject metadata for tenant isolation
        for chunk in chunks:
            chunk.metadata["owner_id"] = owner_id
            chunk.metadata["document_id"] = document_id

        # Setup PGEngine and PGVectorStore
        engine = PGEngine.from_connection_string(settings.async_db_url)
        try:
            await engine.ainit_vectorstore_table(table_name="kb_vectors", vector_size=_vector_size())
        except Exception:
            # Table already exists on subsequent runs — safe to continue
            pass

        vector_store = await PGVectorStore.create(
            engine=engine,
            embedding_service=embeddings,
            table_name="kb_vectors"
        )

        # Add to vector store
        await vector_store.aadd_documents(chunks)

        # Update document state to ready
        stmt = select(KBDocument).filter_by(id=uuid.UUID(document_id))
        result = await db.execute(stmt)
        kb_doc = result.scalar_one_or_none()
        if kb_doc:
            kb_doc.status = "ready"
            await db.commit()

        logger.info("Ingestion completed successfully for document: %s", document_id)

    except Exception as e:
        logger.exception("Failed to ingest document %s", document_id)
        # Update document state to failed
        stmt = select(KBDocument).filter_by(id=uuid.UUID(document_id))
        result = await db.execute(stmt)
        kb_doc = result.scalar_one_or_none()
        if kb_doc:
            kb_doc.status = "failed"
            kb_doc.error_message = str(e)
            await db.commit()
        raise


def get_compression_retriever(owner_id: str):
    """
    Builds the ContextualCompressionRetriever with tenant isolation.
    """
    embeddings = _embeddings_client()
    engine = PGEngine.from_connection_string(settings.async_db_url)
    
    # Initialize base retriever
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name="kb_vectors"
    )
    
    base_retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"owner_id": owner_id}}
    )
    
    # Avoid a second model call merely to compress retrieved chunks.  This
    # keeps the endpoint provider-neutral and works with local Ollama.
    return base_retriever


async def _retrieve_documents(retriever, query: str):
    """Support both modern LangChain retrievers and older compatibility APIs."""
    if hasattr(retriever, "aget_relevant_documents"):
        return await retriever.aget_relevant_documents(query)
    return await retriever.ainvoke(query)


async def retrieve_relevant_context(owner_id: str, query: str) -> List[str]:
    """
    Retrieves compressed relevant context chunks for a query.
    """
    retriever = get_compression_retriever(owner_id)
    docs = await _retrieve_documents(retriever, query)
    return [doc.page_content for doc in docs]


async def chat_with_kb(
    owner_id: str,
    message: str,
    conversation_id: str | None,
    db: AsyncSession
) -> str:
    """
    Executes the conversational retrieval pipeline with history.
    """
    conversation_id = conversation_id or str(uuid.uuid4())

    # 1. Fetch last 5 turns (10 messages) of chat history, scoped to owner
    stmt = (
        select(KBChatHistory)
        .where(
            KBChatHistory.conversation_id == conversation_id,
            KBChatHistory.owner_id == owner_id,
        )
        .order_by(KBChatHistory.created_at.desc())
        .limit(10)
    )
    result = await db.execute(stmt)
    history_records = list(result.scalars().all())
    history_records.reverse()

    # 2. Populate conversation window memory
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=False
    )
    for rec in history_records:
        if rec.role == "human":
            memory.chat_memory.add_user_message(rec.content)
        else:
            memory.chat_memory.add_ai_message(rec.content)

    # 3. Setup retrieval and compression
    retriever = get_compression_retriever(owner_id)

    # 4. Setup LLM & custom chain
    llm = _chat_model(temperature=0.0)

    template = """You are a helpful assistant. Use the following context and chat history to answer the user's question. If you don't know the answer, say you don't know.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Load context and run LCEL chain
    retrieved_docs = await _retrieve_documents(retriever, message)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chat_history_str = memory.load_memory_variables({})["chat_history"]

    chain = prompt | llm | StrOutputParser()
    
    response = await chain.ainvoke({
        "context": context,
        "chat_history": chat_history_str,
        "question": message
    })

    # 5. Save human query and AI response to history
    human_msg = KBChatHistory(
        conversation_id=conversation_id,
        owner_id=owner_id,
        role="human",
        content=message,
        created_at=datetime.now(timezone.utc)
    )
    ai_msg = KBChatHistory(
        conversation_id=conversation_id,
        owner_id=owner_id,
        role="ai",
        content=response,
        created_at=datetime.now(timezone.utc)
    )
    db.add(human_msg)
    db.add(ai_msg)
    await db.commit()

    return response
