"""
Agent Tool Definitions — the "function calling" interface between the agent and backend services.

Each tool is a capability the agent can invoke, defined in OpenAI function-calling format:
  - name & description (for the LLM to decide when to use it)
  - parameters JSON Schema (for the LLM to fill in)
  - fn (the actual Python implementation that calls backend services)

Architecture:
  Agent (LLM decides which tool)  ──>  Tool (validates + executes)
       ──>  Backend Service (business logic)
            ──> Database / Vector Store / File System

This three-layer separation lets the agent focus on reasoning while
the services handle the actual work.
"""
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.database import DocumentService, DocumentRecord
from app.services import get_ingestion_service
from app.schemas import SourceInfo

logger = logging.getLogger(__name__)


# ─── Tool Definition Schema ──────────────────────────────────────────────────
# Each tool is a dict with the OpenAI function-calling format plus a callable `fn`.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索知识库，找到与用户问题最相关的文档片段。当你需要回答一个需要特定知识的问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，最好是和知识库内容相关的关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的相关片段数量，默认4个",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出知识库中所有已上传的文档及其状态（已上传/已导入/出错）。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_info",
            "description": "查看某个文档的详细信息，包括文件名、大小、状态等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "integer",
                        "description": "文档ID",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_document",
            "description": "从知识库中删除一个文档及其所有向量数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "integer",
                        "description": "要删除的文档ID",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_base_stats",
            "description": "查看知识库的统计信息，如文档总数、向量片段数等。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ─── Tool Implementations ────────────────────────────────────────────────────
# Each function below corresponds to one tool definition above.

def _search_kb(db: Session, query: str, top_k: int = 4) -> dict:
    """Execute a vector search against the knowledge base."""
    svc = get_ingestion_service()
    docs = svc.retrieve(query, top_k=top_k)

    sources = [
        SourceInfo(
            filename=d.metadata.get("source_file", "unknown"),
            content=d.page_content,
        )
        for d in docs
    ]
    return {
        "success": True,
        "results": [s.model_dump() for s in sources],
        "total": len(sources),
    }


def _list_docs(db: Session) -> dict:
    """List all documents in the knowledge base."""
    doc_svc = DocumentService(db)
    docs = doc_svc.list_all()
    return {
        "success": True,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "status": d.status,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ],
        "total": len(docs),
    }


def _get_doc_info(db: Session, document_id: int) -> dict:
    """Get detailed information about a specific document."""
    doc_svc = DocumentService(db)
    doc = doc_svc.get(document_id)
    if not doc:
        return {"success": False, "error": f"文档 {document_id} 不存在"}
    return {
        "success": True,
        "document": {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "status": doc.status,
            "error_message": doc.error_message,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        },
    }


def _delete_doc(db: Session, document_id: int) -> dict:
    """Delete a document and its vectors from the knowledge base."""
    doc_svc = DocumentService(db)
    doc = doc_svc.get(document_id)
    if not doc:
        return {"success": False, "error": f"文档 {document_id} 不存在"}

    # Remove vectors from ChromaDB
    if doc.file_path:
        svc = get_ingestion_service()
        svc.delete_document_vectors(doc.file_path)

    # Remove database record + physical file
    doc_svc.delete(document_id)
    return {"success": True, "message": f"文档 '{doc.filename}' 已删除"}


def _get_kb_stats(db: Session) -> dict:
    """Get knowledge base statistics."""
    doc_svc = DocumentService(db)
    svc = get_ingestion_service()
    total_docs = doc_svc.count_by_status()
    ingested_docs = doc_svc.count_by_status("ingested")
    chunks = svc.get_chunk_count()
    return {
        "success": True,
        "stats": {
            "total_documents": total_docs,
            "ingested_documents": ingested_docs,
            "total_chunks": chunks,
        },
    }


# ─── Tool Router ─────────────────────────────────────────────────────────────
# Maps tool names to their implementations. The agent calls this.

TOOL_IMPLEMENTATIONS = {
    "search_knowledge_base": _search_kb,
    "list_documents": _list_docs,
    "get_document_info": _get_doc_info,
    "delete_document": _delete_doc,
    "get_knowledge_base_stats": _get_kb_stats,
}


def execute_tool(db: Session, tool_name: str, arguments: dict) -> Any:
    """
    Execute a named tool with the given arguments.
    This is the single entry point for the agent to call any tool.
    """
    logger.info("Agent executing tool: %s(%s)", tool_name, arguments)

    fn = TOOL_IMPLEMENTATIONS.get(tool_name)
    if not fn:
        return {"success": False, "error": f"未知工具: {tool_name}"}

    try:
        return fn(db, **arguments)
    except Exception as e:
        logger.exception("Tool %s failed", tool_name)
        return {"success": False, "error": str(e)}
