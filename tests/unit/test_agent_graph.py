"""Unit tests for the LangGraph agent graph in research_agent.agent.graph.

Verifies that create_graph() wires up all nodes correctly and that the
compiled graph can execute a simple research workflow end-to-end using mocks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agent.agent.graph import create_graph
from research_agent.models.research import ResearchQuery, SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_retriever(results: list[SearchResult] | None = None) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=results or [])
    return retriever


def _mock_reranker(results: list[SearchResult] | None = None) -> MagicMock:
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=results or [])
    return reranker


def _mock_memory() -> MagicMock:
    memory = MagicMock()
    memory.search = AsyncMock(return_value=[])
    memory.add = AsyncMock(return_value=None)
    return memory


def _mock_llm(response: str = "<final_answer>Test answer.</final_answer>") -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


def _mock_tool(name: str = "firecrawl_search") -> MagicMock:
    from research_agent.models.research import ToolResult

    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(return_value=ToolResult(is_error=False, content="tool result"))
    return tool


# ---------------------------------------------------------------------------
# create_graph
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateGraph:
    def test_returns_compiled_graph(self) -> None:
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=_mock_llm(),
            tools=[],
        )
        # LangGraph compiled graphs expose ainvoke
        assert hasattr(graph, "ainvoke")

    def test_returns_graph_with_astream(self) -> None:
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=_mock_llm(),
            tools=[],
        )
        assert hasattr(graph, "astream")

    def test_accepts_empty_tools_list(self) -> None:
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=_mock_llm(),
            tools=[],
        )
        assert graph is not None

    def test_accepts_multiple_tools(self) -> None:
        tools = [_mock_tool("firecrawl_search"), _mock_tool("firecrawl_scrape")]
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=_mock_llm(),
            tools=tools,
        )
        assert graph is not None

    def test_invalid_retriever_raises(self) -> None:
        with pytest.raises(TypeError, match="retriever"):
            create_graph(
                retriever="not_a_retriever",  # type: ignore[arg-type]
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                llm_client=_mock_llm(),
                tools=[],
            )

    def test_invalid_reranker_raises(self) -> None:
        with pytest.raises(TypeError, match="reranker"):
            create_graph(
                retriever=_mock_retriever(),
                reranker="not_a_reranker",  # type: ignore[arg-type]
                memory_service=_mock_memory(),
                llm_client=_mock_llm(),
                tools=[],
            )

    def test_invalid_memory_service_raises(self) -> None:
        with pytest.raises(TypeError, match="memory_service"):
            create_graph(
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service="not_memory",  # type: ignore[arg-type]
                llm_client=_mock_llm(),
                tools=[],
            )

    def test_invalid_llm_client_raises(self) -> None:
        with pytest.raises(TypeError, match="llm_client"):
            create_graph(
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                llm_client="not_llm",  # type: ignore[arg-type]
                tools=[],
            )


# ---------------------------------------------------------------------------
# End-to-end execution (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGraphExecution:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_final_report(self) -> None:
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=_mock_llm("<final_answer>Quantum entanglement answer.</final_answer>"),
            tools=[],
        )
        query = ResearchQuery(query="What is quantum entanglement?", session_id="sess-1")
        result = await graph.ainvoke(
            {
                "query": query.query,
                "session_id": query.session_id,
                "max_iterations": query.max_iterations,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        assert result["final_report"] is not None
        assert result["final_report"].query == "What is quantum entanglement?"
        assert result["final_report"].session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_ainvoke_calls_memory_search(self) -> None:
        memory = _mock_memory()
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=memory,
            llm_client=_mock_llm(),
            tools=[],
        )
        await graph.ainvoke(
            {
                "query": "test query",
                "session_id": "sess-1",
                "max_iterations": 5,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        memory.search.assert_awaited()

    @pytest.mark.asyncio
    async def test_ainvoke_calls_llm(self) -> None:
        llm = _mock_llm()
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=llm,
            tools=[],
        )
        await graph.ainvoke(
            {
                "query": "test",
                "session_id": "s",
                "max_iterations": 5,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        assert llm.complete.await_count >= 1

    @pytest.mark.asyncio
    async def test_ainvoke_stops_at_max_iterations(self) -> None:
        """When no final_answer is produced, the graph stops at max_iterations."""
        llm = _mock_llm("Thinking... no answer yet.")  # no <final_answer>
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=_mock_memory(),
            llm_client=llm,
            tools=[],
        )
        result = await graph.ainvoke(
            {
                "query": "hard question",
                "session_id": "s",
                "max_iterations": 2,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        # Should still produce a report via synthesis even without <final_answer>
        assert result["final_report"] is not None

    @pytest.mark.asyncio
    async def test_ainvoke_saves_to_memory(self) -> None:
        memory = _mock_memory()
        graph = create_graph(
            retriever=_mock_retriever(),
            reranker=_mock_reranker(),
            memory_service=memory,
            llm_client=_mock_llm(),
            tools=[],
        )
        await graph.ainvoke(
            {
                "query": "test",
                "session_id": "sess-save",
                "max_iterations": 5,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        memory.add.assert_awaited()
