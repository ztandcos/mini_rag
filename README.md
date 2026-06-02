# Mini RAG — 本地知识库问答系统

一个基于 **FastAPI + LangChain + LangGraph + ChromaDB** 的 RAG 本地知识库 MVP 项目，用于学习 **Agent 与后端的深度集成**。

## 架构概览

```
用户 ──> API 路由 ──> 服务层 ──> 数据库/向量存储
        Agent (LLM) ──> 工具 ──> 服务层 ──> 数据库/向量存储
```

**三层分离设计：**
- **API 层** (FastAPI routes) — 处理 HTTP 请求/响应
- **Agent 层** (function calling) — LLM 推理决策，调用工具
- **服务层** (Services) — 纯业务逻辑，可被 API 和 Agent 共同调用

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/ztandcos/mini_rag.git
cd mini_rag

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key (DeepSeek/OpenAI) | — |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | 模型名称 | `deepseek-chat` |
| `EMBEDDING_API_KEY` | 嵌入模型 API Key | — |
| `EMBEDDING_BASE_URL` | 嵌入模型 API 地址 | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-3-small` |

### 3. 启动服务

```bash
python main.py
```

访问 http://localhost:8000/docs 查看 Swagger API 文档。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/documents` | 列出所有文档 |
| `POST` | `/api/documents/upload` | 上传文档 (.txt/.md/.pdf) |
| `POST` | `/api/documents/ingest` | 导入文档到向量库 |
| `DELETE` | `/api/documents/{id}` | 删除文档 |
| `POST` | `/api/agent/chat` | 与 Agent 对话 |

### 使用示例

```bash
# 1. 上传文档
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@./example.txt"

# 2. 导入到向量库
curl -X POST http://localhost:8000/api/documents/ingest

# 3. 与 Agent 对话
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "文档中提到了什么内容？"}'
```

## 项目结构

```
mini_rag/
├── main.py              # FastAPI 入口 + API 路由
├── app/
│   ├── __init__.py
│   ├── config.py        # 环境变量配置 (pydantic-settings)
│   ├── schemas.py       # 请求/响应数据模型 (Pydantic)
│   ├── database.py      # SQLAlchemy ORM + 文档 CRUD 服务
│   ├── services.py      # 核心业务服务 (加载/切片/嵌入/检索/LLM)
│   ├── tools.py         # Agent function calling 工具定义
│   └── agent.py         # LangChain 函数调用 Agent
├── data/
│   ├── uploads/         # 上传文件存储
│   └── chroma_db/       # ChromaDB 持久化
├── requirements.txt
├── .env.example
└── .gitignore
```

## 学习要点

本项目旨在演示 **Agent 与后端如何深度集成**：

1. **服务层复用** — 同一套 `services.py` 同时被 API 路由和 Agent 工具调用
2. **Function Calling** — Agent 通过工具定义与后端交互，LLM 负责推理决策
3. **清晰分层** — API / Agent / Services / Database 各司其职
4. **生产风格** — 类型提示、错误处理、日志、配置管理

## 技术栈

- **FastAPI** — 异步 Web 框架
- **LangChain** — RAG 管道 (文档加载、文本分割、嵌入、检索)
- **LangGraph / Function Calling** — Agent 推理与工具编排
- **ChromaDB** — 向量数据库
- **SQLAlchemy + SQLite** — 文档元数据管理
- **OpenAI / DeepSeek** — LLM 和嵌入模型
