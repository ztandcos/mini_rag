"""
Database layer — SQLAlchemy ORM with SQLite for document metadata tracking.

This service-layer module provides:
  - The DocumentRecord ORM model
  - DB initialization
  - CRUD operations used by both the API routes and the agent

The database tracks what documents have been uploaded, their ingestion
status, and error information — it is separate from the vector store
(ChromaDB) which holds the actual embedding vectors.
"""
import os
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

# ── Engine & Session ─────────────────────────────────────────────────────────

_DB_PATH = settings.database_url.replace("sqlite:///", "")
os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Model ────────────────────────────────────────────────────────────────

class DocumentRecord(Base):
    """Tracks each uploaded document's metadata and ingestion state."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)   # txt | md | pdf
    file_size = Column(Integer, default=0)
    status = Column(String(50), default="uploaded")  # uploaded | ingested | error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc),
                        default=lambda: datetime.now(timezone.utc))


# ── Initialization ───────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)


# ── CRUD Operations ──────────────────────────────────────────────────────────

class DocumentService:
    """Backend service for document metadata CRUD."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, filename: str, file_path: str, file_type: str,
               file_size: int) -> DocumentRecord:
        doc = DocumentRecord(
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            status="uploaded",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get(self, doc_id: int) -> Optional[DocumentRecord]:
        return self.db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()

    def list_all(self) -> list[DocumentRecord]:
        return self.db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()

    def update_status(self, doc_id: int, status: str,
                      error_message: Optional[str] = None) -> Optional[DocumentRecord]:
        doc = self.get(doc_id)
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            self.db.commit()
            self.db.refresh(doc)
        return doc

    def delete(self, doc_id: int) -> bool:
        doc = self.get(doc_id)
        if doc:
            # Also remove the physical file
            if os.path.exists(doc.file_path):
                os.remove(doc.file_path)
            self.db.delete(doc)
            self.db.commit()
            return True
        return False

    def count_by_status(self, status: Optional[str] = None) -> int:
        q = self.db.query(DocumentRecord)
        if status:
            q = q.filter(DocumentRecord.status == status)
        return q.count()

    def get_pending_ingestion(self) -> list[DocumentRecord]:
        """Get documents that haven't been ingested yet."""
        return self.db.query(DocumentRecord).filter(
            DocumentRecord.status == "uploaded"
        ).all()


# ── FastAPI Dependency ───────────────────────────────────────────────────────

def get_db():
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
