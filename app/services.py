"""
Backend Services — the business logic layer that powers both the API and the agent.

Architecture principle:
  API routes  ──>  Services  ──>  Database / Vector Store / File System
  Agent       ──>  Services  ──>  Database / Vector Store / File System

Services are pure business logic with no HTTP or agent concerns.
They are called by both the FastAPI routes AND the agent's tools.
This is the "deep integration" between agent and backend.
"""
import os
import logging
from typing import Optional

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.database import DocumentService, DocumentRecord

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Service: Ingestion (document loading → chunking → embedding → storage)
# ═══════════════════════════════════════════════════════════════════════════════

class IngestionService:
    """
    Handles the document ingestion pipeline:
      File → Load → Chunk → Embed → Store in Vector DB

    This service manages the lifecycle from raw file to searchable vectors.
    """

    def __init__(self):
        self._embedding_key = settings.embedding_api_key
        self._embedding_model_name = settings.embedding_model
        self._embedding_base_url = settings.embedding_base_url

        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        # ── Lazy-initialized attributes ──────────────────────────────────────
        # The embedding client and vector store are created on first use.
        # This avoids crashes when no API key is configured at import time.
        self._embeddings = None
        self._vector_store = None

        # Text splitter — tuned for Chinese + English text (no API key needed)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    # ── Lazy Initializers ─────────────────────────────────────────────────────

    @property
    def embeddings(self):
        if self._embeddings is None:
            kw = {"model": self._embedding_model_name, "api_key": self._embedding_key}
            if self._embedding_base_url:
                kw["base_url"] = self._embedding_base_url
            self._embeddings = OpenAIEmbeddings(**kw)
        return self._embeddings

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name="mini_rag_docs",
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
        return self._vector_store

    def is_available(self) -> bool:
        """Check if the service can initialize (API key is set)."""
        return bool(self._embedding_key)

    # ── Document Loading ──────────────────────────────────────────────────────

    def load_document(self, file_path: str) -> list[Document]:
        """Load a document from disk. Supports .txt, .md, .pdf."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, file_path: str) -> dict:
        """
        Full ingestion: load → chunk → embed → store.
        Returns stats: documents_loaded, chunks_created.
        """
        raw_docs = self.load_document(file_path)
        logger.info("Loaded %d page(s) from %s", len(raw_docs), file_path)

        chunks = self.text_splitter.split_documents(raw_docs)
        logger.info("Split into %d chunks", len(chunks))

        filename = os.path.basename(file_path)
        for c in chunks:
            c.metadata.setdefault("source_file", filename)
            c.metadata.setdefault("source_path", file_path)

        ids = self.vector_store.add_documents(chunks)
        logger.info("Stored %d vectors", len(ids))

        return {"documents_loaded": len(raw_docs), "chunks_created": len(chunks)}

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> list[Document]:
        """Search the vector store for chunks relevant to the query."""
        if not self.is_available():
            return []
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        return retriever.invoke(query)

    # ── Management ────────────────────────────────────────────────────────────

    def delete_document_vectors(self, file_path: str):
        """Remove all vector entries associated with a source file."""
        if not self.is_available():
            return
        existing = self.vector_store.get(where={"source_path": file_path})
        if existing and existing.get("ids"):
            self.vector_store.delete(ids=existing["ids"])
            logger.info("Deleted %d vectors for %s", len(existing["ids"]), file_path)

    def get_chunk_count(self) -> int:
        """Return total number of stored chunks (0 if service not available)."""
        if not self.is_available():
            return 0
        try:
            data = self.vector_store.get()
            return len(data.get("ids", []))
        except Exception:
            return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Service: LLM (wraps the language model for both the agent and direct use)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    """Central LLM access point. Both the agent and the API use this."""

    def __init__(self):
        kw = {
            "model": settings.llm_model,
            "api_key": settings.llm_api_key,
            "temperature": 0.3,
        }
        if settings.llm_base_url:
            kw["base_url"] = settings.llm_base_url
        self.llm = ChatOpenAI(**kw)

    def invoke(self, prompt: str) -> str:
        """Simple prompt → answer."""
        return self.llm.invoke(prompt).content

    def invoke_with_usage(self, prompt: str) -> tuple[str, dict]:
        """
        Invoke the LLM and return (response_text, token_usage).

        token_usage format: {prompt_tokens, completion_tokens, total_tokens}
        Returns empty dict if token usage is not available.
        """
        response = self.llm.invoke(prompt)
        meta = response.response_metadata or {}
        usage = meta.get("token_usage", {})
        return response.content, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    @property
    def model(self) -> ChatOpenAI:
        return self.llm


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Accessors
# ═══════════════════════════════════════════════════════════════════════════════

_ingestion: Optional[IngestionService] = None
_llm: Optional[LLMService] = None


def get_ingestion_service() -> IngestionService:
    global _ingestion
    if _ingestion is None and settings.embedding_api_key:
        _ingestion = IngestionService()
    elif _ingestion is None:
        _ingestion = IngestionService()  # will be available but lazy-init will fail gracefully
    return _ingestion


def get_llm_service() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm
