"""Unit tests for AgentState TypedDict in research_agent.agent.state."""

from __future__ import annotations

import operator

import pytest

from research_agent.agent.state import AgentState
from research_agent.models.research import (
    Citation,
    Message,
    ResearchReport,
    SearchResult,
    ToolResult,
)


@pytest.mark.unit
class TestAgentState:
    """AgentState is a TypedDict — tests verify its structure and reducer behaviour."""

    def _minimal(self) -> AgentState:
        return AgentState(
            query="climate change",
            session_id="sess-1",
            max_iterations=5,
            iteration_count=0,
            search_results=[],
            memories=[],
            messages=[],
            tool_results=[],
            tool_calls_pending=[],
            final_report=None,
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def test_contains_query(self) -> None:
        s = self._minimal()
        assert s["query"] == "climate change"

    def test_contains_session_id(self) -> None:
        s = self._minimal()
        assert s["session_id"] == "sess-1"

    def test_contains_max_iterations(self) -> None:
        s = self._minimal()
        assert s["max_iterations"] == 5

    def test_contains_iteration_count(self) -> None:
        s = self._minimal()
        assert s["iteration_count"] == 0

    def test_contains_search_results(self) -> None:
        s = self._minimal()
        assert s["search_results"] == []

    def test_contains_memories(self) -> None:
        s = self._minimal()
        assert s["memories"] == []

    def test_contains_messages(self) -> None:
        s = self._minimal()
        assert s["messages"] == []

    def test_contains_tool_results(self) -> None:
        s = self._minimal()
        assert s["tool_results"] == []

    def test_contains_tool_calls_pending(self) -> None:
        s = self._minimal()
        assert s["tool_calls_pending"] == []

    def test_contains_final_report(self) -> None:
        s = self._minimal()
        assert s["final_report"] is None

    # ------------------------------------------------------------------
    # Correct types accepted
    # ------------------------------------------------------------------

    def test_accepts_message_list(self) -> None:
        msg = Message(role="user", content="hello")
        s = self._minimal()
        s["messages"] = [msg]
        assert s["messages"][0] is msg

    def test_accepts_tool_result_list(self) -> None:
        tr = ToolResult(is_error=False, content="ok")
        s = self._minimal()
        s["tool_results"] = [tr]
        assert s["tool_results"][0] is tr

    def test_accepts_search_result_list(self) -> None:
        sr = SearchResult(content="text", url="https://x.com", score=0.9, metadata={})
        s = self._minimal()
        s["search_results"] = [sr]
        assert s["search_results"][0] is sr

    def test_accepts_final_report(self) -> None:
        citations = [Citation(url="https://x.com", content_snippet="s", relevance_score=0.9)]
        report = ResearchReport(query="q", summary="s", citations=citations, session_id="sess-1")
        s = self._minimal()
        s["final_report"] = report
        assert s["final_report"] is report

    def test_accepts_tool_calls_pending_dicts(self) -> None:
        s = self._minimal()
        s["tool_calls_pending"] = [{"tool": "firecrawl_search", "input": {"query": "q"}}]
        assert len(s["tool_calls_pending"]) == 1

    def test_accepts_memories_strings(self) -> None:
        s = self._minimal()
        s["memories"] = ["memory one", "memory two"]
        assert s["memories"] == ["memory one", "memory two"]

    # ------------------------------------------------------------------
    # Annotated reducer semantics (operator.add accumulates lists)
    # ------------------------------------------------------------------

    def test_messages_reducer_accumulates(self) -> None:
        """LangGraph uses operator.add to merge messages across node updates."""
        existing = [Message(role="user", content="first")]
        new = [Message(role="assistant", content="second")]
        merged = operator.add(existing, new)
        assert len(merged) == 2
        assert merged[0].content == "first"
        assert merged[1].content == "second"

    def test_tool_results_reducer_accumulates(self) -> None:
        a = [ToolResult(is_error=False, content="a")]
        b = [ToolResult(is_error=False, content="b")]
        merged = operator.add(a, b)
        assert len(merged) == 2
