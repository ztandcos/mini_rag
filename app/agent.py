"""
RAG Agent — the "brain" that orchestrates backend services via function calling.

Architecture:
  User Query  ──>  Agent (LLM)  ──>  Decide which tool(s) to call
                          │
                          ▼
                   Tool Execution  ──>  Backend Services  ──>  DB / Vector Store
                          │
                          ▼
                   Generate answer with tool results
                          │
                          ▼
                   Return answer + sources

This demonstrates the core pattern of agent-backend integration:
  - The LLM handles reasoning & planning
  - Tools provide the interface to backend capabilities
  - Backend services handle the actual data operations
  - The agent never accesses the database or vector store directly
"""
import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.database import DocumentService, get_db
from app.services import get_llm_service
from app.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# System prompt that defines the agent's persona and behavior
SYSTEM_PROMPT = """你是一个智能知识库助手，你可以使用各种工具来帮助用户管理知识库和回答问题。

## 你的能力
1. **搜索知识库** — 当用户问需要特定知识的问题时，搜索知识库获取相关文档片段
2. **管理文档** — 列出、查看、删除知识库中的文档
3. **查看统计** — 查看知识库的整体状态

## 工作流程
1. 分析用户的问题，确定需要哪些信息
2. 选择合适的工具来获取信息
3. 基于工具返回的结果，给出完整的回答
4. 如果搜索结果为空，如实告知用户

请使用中文回答用户问题。"""


class RAGAgent:
    """
    A function-calling agent that uses LangChain's tool-calling support.

    Rather than LangGraph's graph paradigm, this agent uses the simpler
    OpenAI tool-use loop: LLM decides → execute → LLM decides → execute → ...
    until it has enough information to answer.

    This is the most common pattern used in production agent systems.
    """

    def __init__(self):
        self.llm = get_llm_service().model
        # Bind tool definitions to the LLM so it knows what tools are available
        self.llm_with_tools = self.llm.bind_tools(
            [t["function"] for t in TOOL_DEFINITIONS]
        )

    def chat(self, message: str, db: Session) -> dict:
        """
        Process a user message through the agent loop.

        Args:
            message: The user's input message
            db: Database session for tool execution

        Returns:
            dict with "answer", "sources", "agent_thought", "token_usage"
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        all_sources = []
        thought_steps = []
        max_turns = 5  # Prevent infinite loops

        # Track token usage across all turns
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for turn in range(max_turns):
            # ── Step 1: LLM decides next action ──────────────────────────
            response = self.llm_with_tools.invoke(messages)

            # Aggregate token usage from this response
            meta = response.response_metadata or {}
            usage = meta.get("token_usage", {})
            if usage:
                token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                token_usage["total_tokens"] += usage.get("total_tokens", 0)

            messages.append(response)

            # ── Step 2: Check if LLM wants to call tools ────────────────
            if not response.tool_calls:
                # No more tools needed — this is the final answer
                logger.info("Agent finished after %d turns", turn + 1)
                break

            # ── Step 3: Execute each requested tool ──────────────────────
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"].name if hasattr(tool_call["name"], 'name') else tool_call["name"]
                # Handle different LangChain versions - tool_call structure varies
                if hasattr(tool_call, 'name'):
                    tool_name = tool_call.name
                    args = tool_call.args
                    tool_id = tool_call.id
                elif isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                else:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    tool_id = tool_call["id"]

                thought_steps.append({
                    "turn": turn + 1,
                    "tool": tool_name,
                    "arguments": args,
                })
                logger.info("Agent turn %d: calling %s(%s)", turn + 1, tool_name, args)

                # Execute the tool against backend services
                result = execute_tool(db, tool_name, args)

                # Collect sources for the final response
                if tool_name == "search_knowledge_base" and result.get("success"):
                    all_sources.extend(result.get("results", []))

                # Feed result back to the LLM
                result_msg = json.dumps(result, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id if isinstance(tool_id, str) else tool_id.id,
                    "content": result_msg,
                })

        # ── Extract the final answer ─────────────────────────────────────
        answer = ""
        if messages and hasattr(messages[-1], "content"):
            answer = messages[-1].content
        elif messages and isinstance(messages[-1], dict):
            answer = messages[-1].get("content", "")

        logger.info("Agent response: %s... (tokens: %s)", answer[:80], token_usage)

        # If no tokens were tracked at all (provider didn't return usage), remove the zero dict
        if token_usage["total_tokens"] == 0:
            token_usage = {}

        return {
            "answer": answer,
            "sources": all_sources,
            "agent_thought": json.dumps(thought_steps, ensure_ascii=False)
            if thought_steps
            else None,
            "token_usage": token_usage,
        }
