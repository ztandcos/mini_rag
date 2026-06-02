"""
Mini RAG — Minimal Retrieval-Augmented Generation Local Knowledge Base.

A FastAPI application with:
  - Document upload & ingestion pipeline
  - Vector search (ChromaDB + OpenAI-compatible embeddings)
  - LLM-powered Q&A (OpenAI-compatible: DeepSeek, OpenAI, etc.)
  - LangGraph-inspired agent with function calling

Architecture:
  Client  ──>  API Routes  ──>  Services  ──>  Database / Vector Store
              Agent (LLM)  ──>  Tools  ──>  Services  ──>  Database / Vector Store
"""
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.database import init_db, DocumentService, get_db as get_db_session
from app.schemas import (
    DocumentResponse,
    DocumentListResponse,
    UploadResponse,
    IngestResponse,
    DeleteResponse,
    ChatRequest,
    ChatResponse,
    SourceInfo,
    HealthResponse,
)
from app.services import get_ingestion_service
from app.agent import RAGAgent

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (modern replacement for on_event) ───────────────────────────────

_agent: Optional[RAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _agent

    # ── Startup ─────────────────────────────────────────────────────────
    os.makedirs(settings.upload_dir, exist_ok=True)
    init_db()

    if settings.llm_api_key:
        _agent = RAGAgent()
        logger.info("Agent initialized with model: %s", settings.llm_model)
    else:
        logger.warning("No LLM API key configured — agent will be unavailable")

    logger.info("Mini RAG started")
    logger.info("  LLM: %s | %s", settings.llm_model, settings.llm_base_url)
    logger.info("  Embeddings: %s | %s", settings.embedding_model, settings.embedding_base_url)
    logger.info("  Vector Store: %s", settings.chroma_persist_dir)

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Mini RAG shutting down")


# ── App Initialization ───────────────────────────────────────────────────────

app = FastAPI(
    title="Mini RAG",
    description="基于 RAG 的本地知识库问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check(db: DBSession = Depends(get_db_session)):
    """Health check endpoint — verifies system is operational."""
    doc_svc = DocumentService(db)
    doc_count = doc_svc.count_by_status()

    # Only try to access vector store if API key is configured
    chunk_count = 0
    if settings.embedding_api_key:
        try:
            ing_svc = get_ingestion_service()
            chunk_count = ing_svc.get_chunk_count()
        except Exception:
            chunk_count = 0

    return HealthResponse(
        status="ok",
        documents_count=doc_count,
        chunks_count=chunk_count,
        llm_configured=bool(settings.llm_api_key),
    )


# ─── Document Management ─────────────────────────────────────────────────────

@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(db: DBSession = Depends(get_db_session)):
    """List all uploaded documents with their status."""
    doc_svc = DocumentService(db)
    docs = doc_svc.list_all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: DBSession = Depends(get_db_session)):
    """
    Upload a document file.

    Supported formats: .txt, .md, .pdf
    The document is saved to disk and registered in the database.
    Use the /ingest endpoint later to process it into the vector store.
    """
    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".txt", ".md", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持: .txt, .md, .pdf",
        )

    # Save file to upload directory with timestamp to avoid name collisions
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(settings.upload_dir, safe_filename)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        file_size = len(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # Register in database
    doc_svc = DocumentService(db)
    doc = doc_svc.create(
        filename=file.filename or safe_filename,
        file_path=file_path,
        file_type=ext.lstrip("."),
        file_size=file_size,
    )

    return UploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        message="文件上传成功，请使用 /ingest 导入到知识库",
    )


@app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: int, db: DBSession = Depends(get_db_session)):
    """
    Delete a document from the knowledge base.
    Removes: file, database record, and all vector entries.
    """
    doc_svc = DocumentService(db)
    doc = doc_svc.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")

    # Remove vectors if they exist (only if configured)
    if doc.file_path and settings.embedding_api_key:
        ing_svc = get_ingestion_service()
        ing_svc.delete_document_vectors(doc.file_path)

    # Remove DB record + physical file
    doc_svc.delete(document_id)

    return DeleteResponse(message=f"文档 '{doc.filename}' 已删除")


# ─── Ingestion ───────────────────────────────────────────────────────────────

@app.post("/api/documents/ingest", response_model=IngestResponse)
def ingest_documents(db: DBSession = Depends(get_db_session)):
    """
    Ingest all pending documents into the vector store.

    This is the pipeline: load → chunk → embed → store.
    After ingestion, documents become searchable via the agent.
    """
    if not settings.embedding_api_key:
        raise HTTPException(status_code=503, detail="未配置 Embeddings API Key，无法导入")

    doc_svc = DocumentService(db)
    pending = doc_svc.get_pending_ingestion()

    if not pending:
        raise HTTPException(status_code=400, detail="没有待导入的文档")

    ing_svc = get_ingestion_service()
    total_chunks = 0
    processed = 0

    for doc in pending:
        try:
            if not os.path.exists(doc.file_path):
                doc_svc.update_status(doc.id, "error", "文件不存在")
                continue

            stats = ing_svc.ingest(doc.file_path)
            doc_svc.update_status(doc.id, "ingested")
            total_chunks += stats["chunks_created"]
            processed += 1
            logger.info("Ingested document %d: %s", doc.id, doc.filename)

        except Exception as e:
            logger.exception("Failed to ingest document %d", doc.id)
            doc_svc.update_status(doc.id, "error", str(e))

    return IngestResponse(
        documents_processed=processed,
        chunks_created=total_chunks,
    )


# ─── Agent Chat ──────────────────────────────────────────────────────────────

@app.post("/api/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest, db: DBSession = Depends(get_db_session)):
    """
    Chat with the RAG agent.

    The agent uses function calling to:
      1. Search the knowledge base for relevant information
      2. Manage documents (list, view, delete)
      3. Generate answers with source citations

    This is the primary endpoint for interacting with the knowledge base.
    """
    if not _agent:
        # Provide a fallback if no LLM is configured
        raise HTTPException(
            status_code=503,
            detail="Agent 不可用：请配置 LLM API key（设置 LLM_API_KEY 环境变量）",
        )

    result = _agent.chat(req.message, db)

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result.get("sources", [])],
        agent_thought=result.get("agent_thought"),
    )


# ─── Root ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Mini RAG",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
