"""Unit tests for tools/firecrawl.py — FirecrawlSearchTool and FirecrawlScrapeTool.

TDD Phase: RED — these tests are written before any implementation exists.
All MCP network I/O is mocked at the boundary (streamable_http_client + ClientSession).

Run: uv run pytest tests/unit/test_firecrawl_tools.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent
from structlog.testing import capture_logs

from research_agent.tools.firecrawl import FirecrawlScrapeTool, FirecrawlSearchTool
from research_agent.tools.protocols import (
    ScrapeInput,
    SearchInput,
    Tool,
    ToolExecutionError,
    ToolResult,
)

MCP_URL = "https://mcp.firecrawl.dev/test-key/v2/mcp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=False,
    )


def make_error_result(text: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=True,
    )


def make_multiblock_result(*texts: str) -> CallToolResult:
    """Multiple TextContent blocks — all should be joined with newlines."""
    return CallToolResult(
        content=[TextContent(type="text", text=t) for t in texts],
        isError=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(name="mcp_session")
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock()
    return session


@pytest.fixture
def mock_http_transport(mock_read: MagicMock, mock_write: MagicMock) -> AsyncMock:
    cm: AsyncMock = AsyncMock(name="http_transport_cm")
    cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write, MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture
def mock_read() -> MagicMock:
    return MagicMock(name="read_stream")


@pytest.fixture
def mock_write() -> MagicMock:
    return MagicMock(name="write_stream")


@pytest.fixture
def mock_session_cm(mock_session: AsyncMock) -> AsyncMock:
    cm: AsyncMock = AsyncMock(name="session_cm")
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture
def mcp_mocks(
    mock_http_transport: AsyncMock,
    mock_session_cm: AsyncMock,
    mock_session: AsyncMock,
) -> pytest.FixtureRequest:
    """Patches streamable_http_client and ClientSession; yields (mock_http, mock_cs, mock_session)."""
    with (
        patch(
            "research_agent.tools.firecrawl.streamable_http_client",
            return_value=mock_http_transport,
        ) as mock_http,
        patch(
            "research_agent.tools.firecrawl.ClientSession",
            return_value=mock_session_cm,
        ) as mock_cs,
    ):
        yield mock_http, mock_cs, mock_session


# ---------------------------------------------------------------------------
# FirecrawlSearchTool
# ---------------------------------------------------------------------------


class TestFirecrawlSearchToolIdentity:
    def test_name(self) -> None:
        tool = FirecrawlSearchTool(MCP_URL)
        assert tool.name == "firecrawl_search"

    def test_description_is_nonempty(self) -> None:
        tool = FirecrawlSearchTool(MCP_URL)
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_satisfies_tool_protocol(self) -> None:
        tool = FirecrawlSearchTool(MCP_URL)
        assert isinstance(tool, Tool)

    def test_session_initially_none(self) -> None:
        tool = FirecrawlSearchTool(MCP_URL)
        assert tool._session is None


class TestFirecrawlSearchToolConnect:
    async def test_execute_connects_on_first_call(self, mcp_mocks: tuple) -> None:
        mock_http, mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("results")

        tool = FirecrawlSearchTool(MCP_URL)
        await tool.execute(SearchInput(query="deep learning"))

        mock_http.assert_called_once_with(MCP_URL)
        mock_session.initialize.assert_called_once()

    async def test_execute_reuses_session_on_second_call(self, mcp_mocks: tuple) -> None:
        mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("results")

        tool = FirecrawlSearchTool(MCP_URL)
        inp = SearchInput(query="AI safety")

        await tool.execute(inp)
        await tool.execute(inp)

        # Only connected once
        assert mock_http.call_count == 1
        assert mock_session.initialize.call_count == 1
        # But called the tool twice
        assert mock_session.call_tool.call_count == 2

    async def test_concurrent_execute_only_connects_once(self, mcp_mocks: tuple) -> None:
        mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("result")

        tool = FirecrawlSearchTool(MCP_URL)
        inp = SearchInput(query="test concurrency")

        await asyncio.gather(tool.execute(inp), tool.execute(inp))

        assert mock_http.call_count == 1
        assert mock_session.call_tool.call_count == 2

    async def test_inner_lock_double_check_prevents_double_connect(
        self,
        mock_http_transport: AsyncMock,
        mock_session_cm: AsyncMock,
        mock_session: AsyncMock,
    ) -> None:
        """Line 62: second coroutine finds session already set inside the lock.

        Achieves the race by making ``initialize()`` yield the event loop via
        ``asyncio.sleep(0)``.  Coroutine-2 reaches the outer check while
        coroutine-1 holds the lock inside ``initialize()``.  When coroutine-1
        releases the lock, coroutine-2 acquires it and hits the inner
        double-check — returning the existing session instead of reconnecting.
        """

        async def slow_initialize() -> None:
            await asyncio.sleep(0)  # yield so coroutine-2 can reach the lock

        mock_session.initialize = slow_initialize

        with (
            patch(
                "research_agent.tools.firecrawl.streamable_http_client",
                return_value=mock_http_transport,
            ) as mock_http,
            patch(
                "research_agent.tools.firecrawl.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            tool = FirecrawlSearchTool(MCP_URL)
            s1, s2 = await asyncio.gather(
                tool._ensure_connected(),
                tool._ensure_connected(),
            )

        assert s1 is s2  # same session object returned to both callers
        assert mock_http.call_count == 1  # only connected once

    async def test_connection_failure_cleans_up_and_reraises(
        self,
        mock_http_transport: AsyncMock,
        mock_session_cm: AsyncMock,
        mock_session: AsyncMock,
    ) -> None:
        mock_session.initialize.side_effect = RuntimeError("server unavailable")

        with (
            patch(
                "research_agent.tools.firecrawl.streamable_http_client",
                return_value=mock_http_transport,
            ),
            patch(
                "research_agent.tools.firecrawl.ClientSession",
                return_value=mock_session_cm,
            ),
        ):
            tool = FirecrawlSearchTool(MCP_URL)
            with pytest.raises(RuntimeError, match="server unavailable"):
                await tool.execute(SearchInput(query="query"))

            assert tool._session is None
            assert tool._exit_stack is None


class TestFirecrawlSearchToolExecute:
    async def test_execute_calls_firecrawl_search_tool(self, mcp_mocks: tuple) -> None:
        mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("findings")

        tool = FirecrawlSearchTool(MCP_URL)
        await tool.execute(SearchInput(query="quantum computing", limit=3))

        mock_session.call_tool.assert_called_once_with(
            "firecrawl_search",
            arguments={"query": "quantum computing", "limit": 3},
        )

    async def test_execute_returns_tool_result_with_content(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("important findings")

        tool = FirecrawlSearchTool(MCP_URL)
        result = await tool.execute(SearchInput(query="test"))

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.content == "important findings"
        assert result.error is None

    async def test_execute_joins_multiple_content_blocks(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_multiblock_result("block A", "block B")

        tool = FirecrawlSearchTool(MCP_URL)
        result = await tool.execute(SearchInput(query="test"))

        assert result.content == "block A\nblock B"

    async def test_execute_returns_error_result_when_mcp_returns_error(
        self, mcp_mocks: tuple
    ) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_error_result("rate limit exceeded")

        tool = FirecrawlSearchTool(MCP_URL)
        result = await tool.execute(SearchInput(query="test"))

        assert result.is_error is True
        assert result.error == "rate limit exceeded"
        assert result.content is None

    async def test_execute_raises_tool_execution_error_on_exception(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.side_effect = RuntimeError("connection reset")

        tool = FirecrawlSearchTool(MCP_URL)
        with pytest.raises(ToolExecutionError, match="connection reset"):
            await tool.execute(SearchInput(query="test"))

    async def test_execute_raises_type_error_for_scrape_input(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks

        tool = FirecrawlSearchTool(MCP_URL)
        with pytest.raises(TypeError, match="ScrapeInput"):
            await tool.execute(ScrapeInput(url="https://example.com"))


class TestFirecrawlSearchToolAclose:
    async def test_aclose_when_not_connected_is_noop(self) -> None:
        tool = FirecrawlSearchTool(MCP_URL)
        await tool.aclose()  # must not raise
        assert tool._session is None

    async def test_aclose_clears_session_and_stack(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("result")

        tool = FirecrawlSearchTool(MCP_URL)
        await tool.execute(SearchInput(query="connect first"))
        assert tool._session is not None

        await tool.aclose()

        assert tool._session is None
        assert tool._exit_stack is None


# ---------------------------------------------------------------------------
# FirecrawlScrapeTool
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeToolIdentity:
    def test_name(self) -> None:
        tool = FirecrawlScrapeTool(MCP_URL)
        assert tool.name == "firecrawl_scrape"

    def test_description_is_nonempty(self) -> None:
        tool = FirecrawlScrapeTool(MCP_URL)
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_satisfies_tool_protocol(self) -> None:
        tool = FirecrawlScrapeTool(MCP_URL)
        assert isinstance(tool, Tool)

    def test_session_initially_none(self) -> None:
        tool = FirecrawlScrapeTool(MCP_URL)
        assert tool._session is None


class TestFirecrawlScrapeToolExecute:
    async def test_execute_calls_firecrawl_scrape_tool(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("page content")

        tool = FirecrawlScrapeTool(MCP_URL)
        await tool.execute(ScrapeInput(url="https://docs.python.org", only_main_content=True))

        mock_session.call_tool.assert_called_once_with(
            "firecrawl_scrape",
            arguments={"url": "https://docs.python.org", "onlyMainContent": True},
        )

    async def test_execute_passes_only_main_content_false(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("full page")

        tool = FirecrawlScrapeTool(MCP_URL)
        await tool.execute(ScrapeInput(url="https://example.com", only_main_content=False))

        _, kwargs = mock_session.call_tool.call_args
        assert kwargs["arguments"]["onlyMainContent"] is False

    async def test_execute_returns_tool_result_with_content(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("scraped markdown")

        tool = FirecrawlScrapeTool(MCP_URL)
        result = await tool.execute(ScrapeInput(url="https://example.com"))

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.content == "scraped markdown"

    async def test_execute_returns_error_result_when_mcp_returns_error(
        self, mcp_mocks: tuple
    ) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_error_result("URL unreachable")

        tool = FirecrawlScrapeTool(MCP_URL)
        result = await tool.execute(ScrapeInput(url="https://example.com"))

        assert result.is_error is True
        assert result.error == "URL unreachable"

    async def test_execute_raises_tool_execution_error_on_exception(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.side_effect = OSError("DNS resolution failed")

        tool = FirecrawlScrapeTool(MCP_URL)
        with pytest.raises(ToolExecutionError, match="DNS resolution failed"):
            await tool.execute(ScrapeInput(url="https://example.com"))

    async def test_execute_raises_type_error_for_search_input(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks

        tool = FirecrawlScrapeTool(MCP_URL)
        with pytest.raises(TypeError, match="SearchInput"):
            await tool.execute(SearchInput(query="test"))

    async def test_execute_reuses_session_on_second_call(self, mcp_mocks: tuple) -> None:
        mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("content")

        tool = FirecrawlScrapeTool(MCP_URL)
        inp = ScrapeInput(url="https://example.com")

        await tool.execute(inp)
        await tool.execute(inp)

        assert mock_http.call_count == 1
        assert mock_session.call_tool.call_count == 2


class TestFirecrawlScrapeToolAclose:
    async def test_aclose_when_not_connected_is_noop(self) -> None:
        tool = FirecrawlScrapeTool(MCP_URL)
        await tool.aclose()
        assert tool._session is None

    async def test_aclose_clears_session_and_stack(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("content")

        tool = FirecrawlScrapeTool(MCP_URL)
        await tool.execute(ScrapeInput(url="https://example.com"))
        assert tool._session is not None

        await tool.aclose()

        assert tool._session is None
        assert tool._exit_stack is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFirecrawlToolsLogging:
    async def test_connect_logs_mcp_session_connect(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, _mock_session = mcp_mocks
        tool = FirecrawlSearchTool(MCP_URL)
        with capture_logs() as cap:
            await tool._ensure_connected()
        events = [e["event"] for e in cap]
        assert "mcp_session_connect" in events
        entry = next(e for e in cap if e["event"] == "mcp_session_connect")
        assert entry["log_level"] == "debug"

    async def test_search_logs_tool_search_complete_on_success(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("results")
        tool = FirecrawlSearchTool(MCP_URL)
        with capture_logs() as cap:
            await tool.execute(SearchInput(query="test query"))
        events = [e["event"] for e in cap]
        assert "tool_search_complete" in events
        entry = next(e for e in cap if e["event"] == "tool_search_complete")
        assert entry["log_level"] == "info"
        assert "latency_ms" in entry

    async def test_search_logs_tool_search_failed_on_exception(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.side_effect = RuntimeError("connection reset")
        tool = FirecrawlSearchTool(MCP_URL)
        with capture_logs() as cap, pytest.raises(ToolExecutionError):
            await tool.execute(SearchInput(query="test query"))
        events = [e["event"] for e in cap]
        assert "tool_search_failed" in events
        entry = next(e for e in cap if e["event"] == "tool_search_failed")
        assert entry["log_level"] == "error"

    async def test_scrape_logs_tool_scrape_complete_on_success(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("page content")
        tool = FirecrawlScrapeTool(MCP_URL)
        with capture_logs() as cap:
            await tool.execute(ScrapeInput(url="https://example.com/page"))
        events = [e["event"] for e in cap]
        assert "tool_scrape_complete" in events
        entry = next(e for e in cap if e["event"] == "tool_scrape_complete")
        assert entry["log_level"] == "info"
        assert "latency_ms" in entry

    async def test_scrape_logs_url_domain_not_full_url(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.return_value = make_text_result("content")
        tool = FirecrawlScrapeTool(MCP_URL)
        with capture_logs() as cap:
            await tool.execute(ScrapeInput(url="https://example.com/sensitive/path?token=abc"))
        entry = next(e for e in cap if e["event"] == "tool_scrape_complete")
        assert entry["url_domain"] == "example.com"
        assert "url" not in entry

    async def test_scrape_logs_tool_scrape_failed_on_exception(self, mcp_mocks: tuple) -> None:
        _mock_http, _mock_cs, mock_session = mcp_mocks
        mock_session.call_tool.side_effect = OSError("DNS resolution failed")
        tool = FirecrawlScrapeTool(MCP_URL)
        with capture_logs() as cap, pytest.raises(ToolExecutionError):
            await tool.execute(ScrapeInput(url="https://example.com"))
        events = [e["event"] for e in cap]
        assert "tool_scrape_failed" in events
        entry = next(e for e in cap if e["event"] == "tool_scrape_failed")
        assert entry["log_level"] == "error"
        assert entry["url_domain"] == "example.com"
