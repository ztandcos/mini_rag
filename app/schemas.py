"""
Pydantic schemas for API request/response validation.
Clear separation between internal models (database) and API contracts.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─── Documents ───────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Public representation of a document."""
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str              # uploaded | ingested | error
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # enables mapping from SQLAlchemy model


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str = "uploaded"
    message: str


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    status: str = "success"


class DeleteResponse(BaseModel):
    message: str


# ─── Agent Chat ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")


class SourceInfo(BaseModel):
    """A source document chunk referenced in the answer."""
    filename: str
    content: str


class ChatResponse(BaseModel):
    """Agent's response with reasoning trace and sources."""
    answer: str
    sources: list[SourceInfo] = []
    agent_thought: Optional[str] = Field(
        None, description="Agent's internal reasoning trace"
    )


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    documents_count: int = 0
    chunks_count: int = 0
    llm_configured: bool = False
