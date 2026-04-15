"""LangGraph node factory functions and routing predicates.

Each ``make_*`` function returns an ``async`` callable with the signature
``(AgentState) -> dict`` expected by LangGraph.  Dependencies are closed over
via the factory parameter — this is the idiomatic alternative to
``functools.partial`` when the closure captures multiple objects.

Routing functions return a ``Literal`` string that LangGraph maps to an edge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from research_agent.agent.state import AgentState
from research_agent.llm.protocols import LLMClient
from research_agent.memory.protocols import MemoryService
from research_agent.models.research import (
    Citation,
    Message,
    ResearchReport,
    SearchResult,
    ToolResult,
)
from research_agent.retrieval.protocols import Reranker, Retriever
from research_agent.tools.protocols import ScrapeInput, SearchInput, Tool

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NodeFn = Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a deep research assistant.  Your goal is to produce a thorough, \
cited answer to the user's query.

To gather information, emit one or more tool calls using this exact format:
<tool_call>{"tool": "<tool_name>", "input": {<tool_input_json>}}</tool_call>

Available tools:
  - firecrawl_search: {"query": "<string>", "limit": <int, optional>}
  - firecrawl_scrape: {"url": "<string>", "only_main_content": <bool, optional>}

When you have enough information to answer, emit:
<final_answer>Your complete answer here.</final_answer>

Only emit <final_answer> when you are confident in your answer.
"""

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)

# ---------------------------------------------------------------------------
# make_memory_node
# ---------------------------------------------------------------------------


def make_memory_node(memory_service: MemoryService) -> NodeFn:
    """Return a node that fetches relevant memories for the current query."""

    async def memory_node(state: AgentState) -> dict[str, Any]:
        memories = await memory_service.search(state["session_id"], state["query"])
        return {"memories": memories}

    return memory_node


# ---------------------------------------------------------------------------
# make_retrieval_node
# ---------------------------------------------------------------------------


def make_retrieval_node(retriever: Retriever, reranker: Reranker) -> NodeFn:
    """Return a node that retrieves and reranks documents for the current query."""

    async def retrieval_node(state: AgentState) -> dict[str, Any]:
        raw: list[SearchResult] = await retriever.retrieve(state["query"], top_k=10)
        reranked: list[SearchResult] = await reranker.rerank(state["query"], raw)
        return {"search_results": reranked}

    return retrieval_node


# ---------------------------------------------------------------------------
# make_reason_node
# ---------------------------------------------------------------------------


def _build_context(state: AgentState) -> str:
    """Construct a context block injected into the system prompt."""
    parts: list[str] = []

    if state["memories"]:
        parts.append("## Prior memories\n" + "\n".join(f"- {m}" for m in state["memories"]))

    if state["search_results"]:
        snippets = "\n".join(
            f"[{i + 1}] ({r.url})\n{r.content[:500]}" for i, r in enumerate(state["search_results"])
        )
        parts.append(f"## Retrieved documents\n{snippets}")

    if state["tool_results"]:
        tool_outputs = "\n".join(
            f"- {'ERROR' if r.is_error else 'OK'}: {r.error or r.content}"
            for r in state["tool_results"]
        )
        parts.append(f"## Previous tool outputs\n{tool_outputs}")

    return "\n\n".join(parts)


def make_reason_node(llm: LLMClient) -> NodeFn:
    """Return a node that calls the LLM and parses tool calls or a final answer."""

    async def reason_node(state: AgentState) -> dict[str, Any]:
        context = _build_context(state)
        system_content = _SYSTEM_PROMPT
        if context:
            system_content = f"{_SYSTEM_PROMPT}\n\n{context}"

        system_msg = Message(role="system", content=system_content)
        user_msg = Message(role="user", content=state["query"])

        # Build the full message list: system + prior history + current user turn
        # (if this is the first iteration, state["messages"] is empty)
        prior = state["messages"]
        if not prior:
            messages_to_send = [system_msg, user_msg]
            new_messages: list[Message] = [user_msg]
        else:
            messages_to_send = [system_msg] + prior
            new_messages = []

        response: str = await llm.complete(messages_to_send)

        assistant_msg = Message(role="assistant", content=response)
        new_messages.append(assistant_msg)

        # Parse tool calls
        tool_calls: list[dict[str, Any]] = []
        for match in _TOOL_CALL_RE.finditer(response):
            try:
                parsed = json.loads(match.group(1).strip())
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                pass  # malformed tool call — skip silently

        return {
            "messages": new_messages,
            "tool_calls_pending": tool_calls,
            "iteration_count": state["iteration_count"] + 1,
        }

    return reason_node


# ---------------------------------------------------------------------------
# make_tool_node
# ---------------------------------------------------------------------------


def _build_tool_input(tool_name: str, raw_input: dict[str, Any]) -> Any:
    """Construct the typed ToolInput subclass from the parsed LLM output."""
    if tool_name == "firecrawl_search":
        return SearchInput(
            query=raw_input["query"],
            limit=int(raw_input.get("limit", 5)),
        )
    if tool_name == "firecrawl_scrape":
        return ScrapeInput(
            url=raw_input["url"],
            only_main_content=bool(raw_input.get("only_main_content", True)),
        )
    return None  # unknown tool — caller handles


def make_tool_node(tools: list[Tool]) -> NodeFn:
    """Return a node that dispatches all pending tool calls and collects results."""
    tool_map: dict[str, Tool] = {t.name: t for t in tools}

    async def tool_node(state: AgentState) -> dict[str, Any]:
        pending = state["tool_calls_pending"]
        new_results: list[ToolResult] = []
        new_messages: list[Message] = []

        for call in pending:
            tool_name: str = call.get("tool", "")
            raw_input: dict[str, Any] = call.get("input", {})

            if tool_name not in tool_map:
                err = ToolResult(
                    is_error=True,
                    error=f"Unknown tool: {tool_name!r}",
                )
                new_results.append(err)
                new_messages.append(
                    Message(role="tool", content=f"ERROR: Unknown tool {tool_name!r}")
                )
                continue

            tool_input = _build_tool_input(tool_name, raw_input)
            result: ToolResult = await tool_map[tool_name].execute(tool_input)
            new_results.append(result)
            content = result.error if result.is_error else (result.content or "")
            new_messages.append(Message(role="tool", content=f"[{tool_name}] {content}"))

        return {
            "tool_results": new_results,
            "tool_calls_pending": [],
            "messages": new_messages,
        }

    return tool_node


# ---------------------------------------------------------------------------
# make_synthesis_node
# ---------------------------------------------------------------------------


def make_synthesis_node(llm: LLMClient, memory_service: MemoryService) -> NodeFn:
    """Return a node that synthesises a final report and persists it to memory."""

    async def synthesis_node(state: AgentState) -> dict[str, Any]:
        context = _build_context(state)
        synthesis_prompt = (
            f"Based on all gathered information, write a comprehensive answer to: "
            f"{state['query']}\n\n{context}"
        )
        messages = [
            Message(role="system", content="You are a research synthesis assistant."),
            Message(role="user", content=synthesis_prompt),
        ]
        summary: str = await llm.complete(messages)

        citations = [
            Citation(
                url=r.url,
                content_snippet=r.content[:200],
                relevance_score=r.score,
            )
            for r in state["search_results"]
        ]

        report = ResearchReport(
            query=state["query"],
            summary=summary,
            citations=citations,
            session_id=state["session_id"],
        )

        await memory_service.add(state["session_id"], summary)

        return {"final_report": report}

    return synthesis_node


# ---------------------------------------------------------------------------
# Routing predicates
# ---------------------------------------------------------------------------


def should_continue(
    state: AgentState,
) -> Literal["retrieve", "synthesize"]:
    """Decide whether to run another ReAct iteration or move to synthesis."""
    if state["final_report"] is not None:
        return "synthesize"
    if state["iteration_count"] >= state["max_iterations"]:
        return "synthesize"
    return "retrieve"


def should_use_tools(
    state: AgentState,
) -> Literal["tools", "retrieve"]:
    """After reasoning, dispatch to tool node or loop back to retrieval."""
    if state["tool_calls_pending"]:
        return "tools"
    return "retrieve"
