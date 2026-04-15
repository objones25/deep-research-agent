"""Unit tests for agent node functions in research_agent.agent.nodes.

All external dependencies (LLMClient, Retriever, Reranker, MemoryService, Tool)
are replaced with AsyncMock / MagicMock at the protocol boundary.  No network
I/O or file I/O occurs in these tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from research_agent.agent.nodes import (
    make_memory_node,
    make_reason_node,
    make_retrieval_node,
    make_synthesis_node,
    make_tool_node,
    should_continue,
    should_use_tools,
)
from research_agent.agent.state import AgentState
from research_agent.models.research import (
    Message,
    ResearchReport,
    SearchResult,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides: Any) -> AgentState:
    state: AgentState = {
        "query": "What is quantum entanglement?",
        "session_id": "sess-test",
        "max_iterations": 5,
        "iteration_count": 0,
        "search_results": [],
        "memories": [],
        "messages": [],
        "tool_results": [],
        "tool_calls_pending": [],
        "final_report": None,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _search_result(n: int = 1) -> SearchResult:
    return SearchResult(
        content=f"content {n}",
        url=f"https://example.com/{n}",
        score=0.9 - n * 0.05,
        metadata={},
    )


# ---------------------------------------------------------------------------
# make_memory_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryNode:
    @pytest.mark.asyncio
    async def test_returns_memories_from_service(self) -> None:
        memory_service = AsyncMock()
        memory_service.search.return_value = ["memory A", "memory B"]

        node = make_memory_node(memory_service)
        result = await node(_base_state())

        assert result["memories"] == ["memory A", "memory B"]

    @pytest.mark.asyncio
    async def test_calls_search_with_query_and_session(self) -> None:
        memory_service = AsyncMock()
        memory_service.search.return_value = []

        node = make_memory_node(memory_service)
        state = _base_state(query="test query", session_id="sess-42")
        await node(state)

        memory_service.search.assert_awaited_once_with("sess-42", "test query")

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_memories(self) -> None:
        memory_service = AsyncMock()
        memory_service.search.return_value = []

        node = make_memory_node(memory_service)
        result = await node(_base_state())

        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_does_not_return_extra_keys(self) -> None:
        memory_service = AsyncMock()
        memory_service.search.return_value = []

        node = make_memory_node(memory_service)
        result = await node(_base_state())

        assert set(result.keys()) == {"memories"}


# ---------------------------------------------------------------------------
# make_retrieval_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrievalNode:
    @pytest.mark.asyncio
    async def test_returns_reranked_results(self) -> None:
        results = [_search_result(1), _search_result(2)]
        reranked = [_search_result(2), _search_result(1)]

        retriever = AsyncMock()
        retriever.retrieve.return_value = results
        reranker = AsyncMock()
        reranker.rerank.return_value = reranked

        node = make_retrieval_node(retriever, reranker)
        result = await node(_base_state())

        assert result["search_results"] == reranked

    @pytest.mark.asyncio
    async def test_calls_retriever_with_query(self) -> None:
        retriever = AsyncMock()
        retriever.retrieve.return_value = []
        reranker = AsyncMock()
        reranker.rerank.return_value = []

        node = make_retrieval_node(retriever, reranker)
        await node(_base_state(query="quantum entanglement"))

        retriever.retrieve.assert_awaited_once()
        call_args = retriever.retrieve.call_args
        assert (
            call_args[0][0] == "quantum entanglement"
            or call_args[1].get("query") == "quantum entanglement"
        )

    @pytest.mark.asyncio
    async def test_calls_reranker_with_query_and_results(self) -> None:
        results = [_search_result(1)]
        retriever = AsyncMock()
        retriever.retrieve.return_value = results
        reranker = AsyncMock()
        reranker.rerank.return_value = results

        node = make_retrieval_node(retriever, reranker)
        await node(_base_state(query="q"))

        reranker.rerank.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_only_returns_search_results_key(self) -> None:
        retriever = AsyncMock()
        retriever.retrieve.return_value = []
        reranker = AsyncMock()
        reranker.rerank.return_value = []

        node = make_retrieval_node(retriever, reranker)
        result = await node(_base_state())

        assert set(result.keys()) == {"search_results"}


# ---------------------------------------------------------------------------
# make_reason_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReasonNode:
    @pytest.mark.asyncio
    async def test_increments_iteration_count(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>Done.</final_answer>"

        node = make_reason_node(llm)
        result = await node(_base_state(iteration_count=2))

        assert result["iteration_count"] == 3

    @pytest.mark.asyncio
    async def test_parses_tool_call_from_response(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = (
            "<tool_call>"
            '{"tool": "firecrawl_search", "input": {"query": "quantum"}}'
            "</tool_call>"
        )

        node = make_reason_node(llm)
        result = await node(_base_state())

        assert result["tool_calls_pending"] == [
            {"tool": "firecrawl_search", "input": {"query": "quantum"}}
        ]

    @pytest.mark.asyncio
    async def test_no_tool_calls_when_final_answer(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>Quantum entanglement is...</final_answer>"

        node = make_reason_node(llm)
        result = await node(_base_state())

        assert result["tool_calls_pending"] == []

    @pytest.mark.asyncio
    async def test_appends_user_and_assistant_messages(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>answer</final_answer>"

        node = make_reason_node(llm)
        result = await node(_base_state())

        # messages is an append-only list — the node returns new messages to add
        new_messages: list[Message] = result["messages"]
        roles = [m.role for m in new_messages]
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_calls_llm_with_messages(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>x</final_answer>"
        existing_msg = Message(role="user", content="prior")

        node = make_reason_node(llm)
        await node(_base_state(messages=[existing_msg]))

        llm.complete.assert_awaited_once()
        call_args = llm.complete.call_args[0][0]
        assert any(m.role == "user" for m in call_args)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parsed(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = (
            "<tool_call>"
            '{"tool": "firecrawl_search", "input": {"query": "a"}}'
            "</tool_call>"
            "<tool_call>"
            '{"tool": "firecrawl_scrape", "input": {"url": "https://x.com"}}'
            "</tool_call>"
        )

        node = make_reason_node(llm)
        result = await node(_base_state())

        assert len(result["tool_calls_pending"]) == 2

    @pytest.mark.asyncio
    async def test_empty_tool_calls_when_no_markers(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "I need to think more about this."

        node = make_reason_node(llm)
        result = await node(_base_state())

        assert result["tool_calls_pending"] == []

    @pytest.mark.asyncio
    async def test_context_injected_when_memories_present(self) -> None:
        """Covers _build_context memories branch and system_content context branch."""
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>done</final_answer>"

        node = make_reason_node(llm)
        await node(_base_state(memories=["prior fact A", "prior fact B"]))

        call_args = llm.complete.call_args[0][0]
        system_msg = call_args[0]
        assert "Prior memories" in system_msg.content

    @pytest.mark.asyncio
    async def test_context_injected_when_tool_results_present(self) -> None:
        """Covers _build_context tool_results branch."""
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>done</final_answer>"

        node = make_reason_node(llm)
        tr = ToolResult(is_error=False, content="tool output content")
        await node(_base_state(tool_results=[tr]))

        call_args = llm.complete.call_args[0][0]
        system_msg = call_args[0]
        assert "Previous tool outputs" in system_msg.content

    @pytest.mark.asyncio
    async def test_prior_messages_used_in_subsequent_turn(self) -> None:
        """Covers the else branch when prior messages already exist."""
        llm = AsyncMock()
        llm.complete.return_value = "<final_answer>continued</final_answer>"

        node = make_reason_node(llm)
        prior = [
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
        ]
        result = await node(_base_state(messages=prior))

        # New messages list should not re-add the user query
        new_msgs: list[Message] = result["messages"]
        assert all(m.role == "assistant" for m in new_msgs)

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json_is_skipped(self) -> None:
        """Covers the except json.JSONDecodeError pass branch."""
        llm = AsyncMock()
        llm.complete.return_value = "<tool_call>NOT VALID JSON {{</tool_call>"

        node = make_reason_node(llm)
        result = await node(_base_state())

        # Malformed JSON should be skipped — no tool calls pending
        assert result["tool_calls_pending"] == []


# ---------------------------------------------------------------------------
# make_tool_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolNode:
    def _make_tool(self, name: str, result: ToolResult) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.execute = AsyncMock(return_value=result)
        return tool

    @pytest.mark.asyncio
    async def test_executes_pending_tool_calls(self) -> None:
        tool = self._make_tool("firecrawl_search", ToolResult(is_error=False, content="result"))
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[{"tool": "firecrawl_search", "input": {"query": "quantum"}}]
        )
        result = await node(state)

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].content == "result"

    @pytest.mark.asyncio
    async def test_clears_pending_calls_after_execution(self) -> None:
        tool = self._make_tool("firecrawl_search", ToolResult(is_error=False, content="ok"))
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[{"tool": "firecrawl_search", "input": {"query": "q"}}]
        )
        result = await node(state)

        assert result["tool_calls_pending"] == []

    @pytest.mark.asyncio
    async def test_executes_multiple_tools(self) -> None:
        search_tool = self._make_tool(
            "firecrawl_search", ToolResult(is_error=False, content="search result")
        )
        scrape_tool = self._make_tool(
            "firecrawl_scrape", ToolResult(is_error=False, content="scraped content")
        )
        node = make_tool_node([search_tool, scrape_tool])
        state = _base_state(
            tool_calls_pending=[
                {"tool": "firecrawl_search", "input": {"query": "q"}},
                {"tool": "firecrawl_scrape", "input": {"url": "https://x.com"}},
            ]
        )
        result = await node(state)

        assert len(result["tool_results"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self) -> None:
        node = make_tool_node([])
        state = _base_state(tool_calls_pending=[{"tool": "nonexistent_tool", "input": {}}])
        result = await node(state)

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0].is_error is True

    @pytest.mark.asyncio
    async def test_appends_tool_result_messages(self) -> None:
        tool = self._make_tool(
            "firecrawl_search", ToolResult(is_error=False, content="result content")
        )
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[{"tool": "firecrawl_search", "input": {"query": "q"}}]
        )
        result = await node(state)

        assert "messages" in result
        assert any(m.role == "tool" for m in result["messages"])

    @pytest.mark.asyncio
    async def test_no_pending_calls_returns_empty(self) -> None:
        node = make_tool_node([])
        result = await node(_base_state(tool_calls_pending=[]))

        assert result["tool_results"] == []
        assert result["tool_calls_pending"] == []

    @pytest.mark.asyncio
    async def test_search_input_constructed_correctly(self) -> None:
        from research_agent.tools.protocols import SearchInput

        tool = self._make_tool("firecrawl_search", ToolResult(is_error=False, content="ok"))
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[
                {"tool": "firecrawl_search", "input": {"query": "entanglement", "limit": 3}}
            ]
        )
        await node(state)

        call_args = tool.execute.call_args[0][0]
        assert isinstance(call_args, SearchInput)
        assert call_args.query == "entanglement"
        assert call_args.limit == 3

    @pytest.mark.asyncio
    async def test_scrape_input_constructed_correctly(self) -> None:
        from research_agent.tools.protocols import ScrapeInput

        tool = self._make_tool("firecrawl_scrape", ToolResult(is_error=False, content="ok"))
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[
                {"tool": "firecrawl_scrape", "input": {"url": "https://example.com"}}
            ]
        )
        await node(state)

        call_args = tool.execute.call_args[0][0]
        assert isinstance(call_args, ScrapeInput)
        assert call_args.url == "https://example.com"


# ---------------------------------------------------------------------------
# make_synthesis_node
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSynthesisNode:
    @pytest.mark.asyncio
    async def test_produces_final_report(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "Quantum entanglement is a phenomenon..."
        memory_service = AsyncMock()

        node = make_synthesis_node(llm, memory_service)
        state = _base_state(
            query="What is quantum entanglement?",
            session_id="sess-1",
            search_results=[_search_result(1)],
        )
        result = await node(state)

        assert result["final_report"] is not None
        report: ResearchReport = result["final_report"]
        assert report.query == "What is quantum entanglement?"
        assert report.session_id == "sess-1"
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    @pytest.mark.asyncio
    async def test_saves_summary_to_memory(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "Summary of findings."
        memory_service = AsyncMock()

        node = make_synthesis_node(llm, memory_service)
        state = _base_state(session_id="sess-1", search_results=[])
        await node(state)

        memory_service.add.assert_awaited_once()
        call_args = memory_service.add.call_args
        assert call_args[0][0] == "sess-1" or call_args[1].get("session_id") == "sess-1"

    @pytest.mark.asyncio
    async def test_citations_built_from_search_results(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "The answer."
        memory_service = AsyncMock()

        node = make_synthesis_node(llm, memory_service)
        results = [_search_result(1), _search_result(2)]
        state = _base_state(search_results=results)
        result = await node(state)

        report: ResearchReport = result["final_report"]  # type: ignore[assignment]
        assert len(report.citations) == len(results)
        assert report.citations[0].url == results[0].url

    @pytest.mark.asyncio
    async def test_empty_search_results_yields_empty_citations(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "No sources found."
        memory_service = AsyncMock()

        node = make_synthesis_node(llm, memory_service)
        result = await node(_base_state(search_results=[]))

        report: ResearchReport = result["final_report"]  # type: ignore[assignment]
        assert report.citations == []


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoutingFunctions:
    def test_should_continue_returns_continue_below_limit(self) -> None:
        state = _base_state(iteration_count=2, max_iterations=5, tool_calls_pending=[])
        assert should_continue(state) == "retrieve"

    def test_should_continue_returns_end_at_limit(self) -> None:
        state = _base_state(iteration_count=5, max_iterations=5)
        assert should_continue(state) == "synthesize"

    def test_should_continue_returns_end_when_final_report_set(self) -> None:
        report = ResearchReport(query="q", summary="s", citations=[], session_id="sess")
        state = _base_state(iteration_count=1, max_iterations=5, final_report=report)
        assert should_continue(state) == "synthesize"

    def test_should_use_tools_returns_tools_when_pending(self) -> None:
        state = _base_state(
            tool_calls_pending=[{"tool": "firecrawl_search", "input": {"query": "q"}}]
        )
        assert should_use_tools(state) == "tools"

    def test_should_use_tools_returns_retrieve_when_no_pending(self) -> None:
        state = _base_state(tool_calls_pending=[])
        assert should_use_tools(state) == "retrieve"


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrievalNodeLogging:
    @pytest.mark.asyncio
    async def test_logs_retrieval_executed_with_counts_and_latency(self) -> None:
        raw_results = [_search_result(1), _search_result(2)]
        reranked_results = [_search_result(1)]
        retriever = AsyncMock()
        retriever.retrieve.return_value = raw_results
        reranker = AsyncMock()
        reranker.rerank.return_value = reranked_results

        node = make_retrieval_node(retriever, reranker)
        with capture_logs() as cap:
            await node(_base_state())

        events = [e["event"] for e in cap]
        assert "retrieval_executed" in events
        entry = next(e for e in cap if e["event"] == "retrieval_executed")
        assert entry["log_level"] == "info"
        assert entry["num_retrieved"] == 2
        assert entry["num_reranked"] == 1
        assert "latency_ms" in entry


@pytest.mark.unit
class TestReasonNodeLogging:
    @pytest.mark.asyncio
    async def test_logs_warning_on_malformed_tool_call_json(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "<tool_call>NOT VALID JSON {{</tool_call>"

        node = make_reason_node(llm)
        with capture_logs() as cap:
            await node(_base_state())

        events = [e["event"] for e in cap]
        assert "tool_call_parse_failed" in events
        entry = next(e for e in cap if e["event"] == "tool_call_parse_failed")
        assert entry["log_level"] == "warning"

    @pytest.mark.asyncio
    async def test_no_parse_warning_on_valid_tool_call(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = (
            "<tool_call>"
            '{"tool": "firecrawl_search", "input": {"query": "quantum"}}'
            "</tool_call>"
        )

        node = make_reason_node(llm)
        with capture_logs() as cap:
            await node(_base_state())

        parse_failures = [e for e in cap if e["event"] == "tool_call_parse_failed"]
        assert parse_failures == []


@pytest.mark.unit
class TestToolNodeLogging:
    def _make_tool(self, name: str, result: ToolResult) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        tool.execute = AsyncMock(return_value=result)
        return tool

    @pytest.mark.asyncio
    async def test_logs_tool_unknown_warning(self) -> None:
        node = make_tool_node([])
        state = _base_state(tool_calls_pending=[{"tool": "nonexistent_tool", "input": {}}])

        with capture_logs() as cap:
            await node(state)

        events = [e["event"] for e in cap]
        assert "tool_unknown" in events
        entry = next(e for e in cap if e["event"] == "tool_unknown")
        assert entry["log_level"] == "warning"
        assert entry["tool_name"] == "nonexistent_tool"

    @pytest.mark.asyncio
    async def test_logs_tool_dispatched_on_success(self) -> None:
        tool = self._make_tool("firecrawl_search", ToolResult(is_error=False, content="ok"))
        node = make_tool_node([tool])
        state = _base_state(
            tool_calls_pending=[{"tool": "firecrawl_search", "input": {"query": "q"}}]
        )

        with capture_logs() as cap:
            await node(state)

        events = [e["event"] for e in cap]
        assert "tool_dispatched" in events
        entry = next(e for e in cap if e["event"] == "tool_dispatched")
        assert entry["log_level"] == "info"
        assert entry["tool_name"] == "firecrawl_search"

    @pytest.mark.asyncio
    async def test_no_tool_dispatched_log_for_unknown_tool(self) -> None:
        node = make_tool_node([])
        state = _base_state(tool_calls_pending=[{"tool": "bad_tool", "input": {}}])

        with capture_logs() as cap:
            await node(state)

        dispatched = [e for e in cap if e["event"] == "tool_dispatched"]
        assert dispatched == []


@pytest.mark.unit
class TestSynthesisNodeLogging:
    @pytest.mark.asyncio
    async def test_logs_synthesis_started(self) -> None:
        llm = AsyncMock()
        llm.complete.return_value = "Summary."
        memory_service = AsyncMock()

        node = make_synthesis_node(llm, memory_service)
        with capture_logs() as cap:
            await node(_base_state(session_id="sess-log-test", search_results=[]))

        events = [e["event"] for e in cap]
        assert "synthesis_started" in events
        entry = next(e for e in cap if e["event"] == "synthesis_started")
        assert entry["log_level"] == "info"
        assert entry["session_id"] == "sess-log-test"
