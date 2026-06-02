"""
Application configuration via environment variables.
Uses pydantic-settings for type-safe, .env-aware config management.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM (OpenAI-compatible: DeepSeek, OpenAI, etc.) ─────────────────
    llm_api_key: str = ""                     # e.g. sk-xxx (DeepSeek or OpenAI)
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # ── Embeddings (OpenAI-compatible) ──────────────────────────────────
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    # ── Vector Store ────────────────────────────────────────────────────
    chroma_persist_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db"
    )

    # ── Document Storage ────────────────────────────────────────────────
    upload_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "uploads"
    )

    # ── Database ────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/mini_rag.db"

    # ── Chunking ────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ── Retrieval ───────────────────────────────────────────────────────
    retrieval_top_k: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
