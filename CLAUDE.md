# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini RAG — a minimal RAG (Retrieval-Augmented Generation) local knowledge base built with FastAPI + LangChain + ChromaDB. Demonstrates deep integration between agent (function calling) and backend services.

## Architecture

Three-layer design pattern:
- **API Layer** (`main.py`) — FastAPI routes, HTTP request/response handling
- **Agent Layer** (`app/agent.py` + `app/tools.py`) — LLM reasoning via function calling, tools delegate to services
- **Service Layer** (`app/services.py` + `app/database.py`) — Pure business logic, shared between API and Agent

Key principle: Services are the single source of truth. Both API routes and Agent tools call the same services.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py
```

## Commands

```bash
# Start server
python main.py

# API docs (after starting)
# http://localhost:8000/docs

# Install/update deps
pip install -r requirements.txt

# Freeze current deps
pip freeze > requirements.txt
```

## Project Structure

```
mini_rag/
├── main.py              # FastAPI entry point + all routes
├── app/
│   ├── config.py        # pydantic-settings, reads from .env
│   ├── schemas.py       # Pydantic request/response models
│   ├── database.py      # SQLAlchemy ORM, DocumentRecord, DocumentService CRUD
│   ├── services.py      # IngestionService (load→chunk→embed→store), LLMService
│   ├── tools.py         # Agent tool definitions (OpenAI function-calling format)
│   └── agent.py         # RAGAgent — function-calling loop with tool execution
└── data/
    ├── uploads/         # Uploaded files
    └── chroma_db/       # ChromaDB persistence
```

## Key Files

- `main.py` — Read this first. All API routes are here. Startup initializes DB + Agent.
- `app/services.py` — Core business logic. `IngestionService` handles the full RAG pipeline.
- `app/tools.py` — Tool definitions + implementations. The bridge between agent and services.
- `app/agent.py` — `RAGAgent` class with tool-calling loop. Uses `llm.bind_tools()` pattern.
- `app/database.py` — `DocumentService` CRUD class. SQLite via SQLAlchemy.

## Configuration

All config via environment variables (`.env` file). Supports separate providers for LLM and embeddings:
- LLM: DeepSeek (default), OpenAI, or any OpenAI-compatible provider
- Embeddings: OpenAI (default), or any OpenAI-compatible provider
