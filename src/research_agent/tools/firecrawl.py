"""Firecrawl tools: search and scrape via the Firecrawl remote MCP server.

Each tool manages its own MCP session with a lazy-connect pattern:
the connection is established on the first ``execute()`` call and reused
for all subsequent calls.  ``asyncio.Lock`` prevents duplicate connections
under concurrent use.

Usage:
    tool = FirecrawlSearchTool(mcp_url)
    result = await tool.execute(SearchInput(query="deep learning"))
    await tool.aclose()
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from research_agent.logging import get_logger
from research_agent.tools.protocols import (
    ScrapeInput,
    SearchInput,
    ToolExecutionError,
    ToolInput,
    ToolResult,
)

_log = get_logger(__name__)


def _extract_text(result: CallToolResult) -> str:
    """Join all TextContent blocks from an MCP CallToolResult into a single string."""
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


class _FirecrawlBaseTool:
    """Manages the MCP session lifecycle shared by Firecrawl tools.

    Subclasses inherit lazy-connect and cleanup behaviour; they only need to
    implement ``name``, ``description``, and ``execute``.
    """

    def __init__(self, mcp_url: str) -> None:
        self._mcp_url = mcp_url
        self._session: ClientSession | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._exit_stack: AsyncExitStack | None = None

    async def _ensure_connected(self) -> ClientSession:
        """Return the active MCP session, connecting lazily if needed.

        The double-checked locking pattern prevents redundant connections
        when multiple coroutines race to the first ``execute()`` call.

        Raises:
            Any exception raised by the MCP transport or session initialisation
            is propagated after cleaning up the partially-opened stack.
        """
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is not None:
                return self._session
            stack = AsyncExitStack()
            try:
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(self._mcp_url)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
            except Exception:
                await stack.aclose()
                raise
            self._exit_stack = stack
            self._session = session
            _log.debug("mcp_session_connect")
            return session

    async def aclose(self) -> None:
        """Close the MCP session and release all transport resources."""
        async with self._lock:
            if self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None
                self._session = None


class FirecrawlSearchTool(_FirecrawlBaseTool):
    """Wraps the ``firecrawl_search`` MCP tool exposed by the Firecrawl server.

    Connects to the Firecrawl remote MCP server on first use and reuses the
    session for all subsequent searches.
    """

    @property
    def name(self) -> str:
        return "firecrawl_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Firecrawl and return full-page markdown content. "
            "Use this when you need recent information, broad topic coverage, "
            "or want to discover relevant URLs before scraping."
        )

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """Run a web search via Firecrawl.

        Args:
            tool_input: Must be a ``SearchInput`` instance.

        Returns:
            ``ToolResult`` with ``is_error=False`` and markdown content on success,
            or ``is_error=True`` with an error message if Firecrawl reported an error.

        Raises:
            TypeError: If *tool_input* is not ``SearchInput``.
            ToolExecutionError: If the MCP call raises an exception.
        """
        if not isinstance(tool_input, SearchInput):
            raise TypeError(
                f"FirecrawlSearchTool requires SearchInput, got {type(tool_input).__name__}"
            )
        session = await self._ensure_connected()
        t0 = time.perf_counter()
        try:
            result: CallToolResult = await session.call_tool(
                "firecrawl_search",
                arguments={"query": tool_input.query, "limit": tool_input.limit},
            )
        except Exception as exc:
            _log.error("tool_search_failed", error=str(exc))
            raise ToolExecutionError(f"MCP call_tool failed: {exc}") from exc

        _log.info(
            "tool_search_complete",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        if result.isError:
            return ToolResult(is_error=True, error=_extract_text(result))
        return ToolResult(is_error=False, content=_extract_text(result))


class FirecrawlScrapeTool(_FirecrawlBaseTool):
    """Wraps the ``firecrawl_scrape`` MCP tool exposed by the Firecrawl server.

    Connects to the Firecrawl remote MCP server on first use and reuses the
    session for all subsequent scrapes.
    """

    @property
    def name(self) -> str:
        return "firecrawl_scrape"

    @property
    def description(self) -> str:
        return (
            "Scrape a single URL using Firecrawl and return the page content as markdown. "
            "Use this when you have a specific URL to fetch and need clean, structured text "
            "without navigating to it manually."
        )

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """Scrape a URL via Firecrawl.

        Args:
            tool_input: Must be a ``ScrapeInput`` instance.

        Returns:
            ``ToolResult`` with ``is_error=False`` and page markdown on success,
            or ``is_error=True`` with an error message if Firecrawl reported an error.

        Raises:
            TypeError: If *tool_input* is not ``ScrapeInput``.
            ToolExecutionError: If the MCP call raises an exception.
        """
        if not isinstance(tool_input, ScrapeInput):
            raise TypeError(
                f"FirecrawlScrapeTool requires ScrapeInput, got {type(tool_input).__name__}"
            )
        url_domain = urlparse(tool_input.url).netloc
        session = await self._ensure_connected()
        t0 = time.perf_counter()
        try:
            result: CallToolResult = await session.call_tool(
                "firecrawl_scrape",
                arguments={
                    "url": tool_input.url,
                    "onlyMainContent": tool_input.only_main_content,
                },
            )
        except Exception as exc:
            _log.error("tool_scrape_failed", url_domain=url_domain, error=str(exc))
            raise ToolExecutionError(f"MCP call_tool failed: {exc}") from exc

        _log.info(
            "tool_scrape_complete",
            url_domain=url_domain,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        if result.isError:
            return ToolResult(is_error=True, error=_extract_text(result))
        return ToolResult(is_error=False, content=_extract_text(result))
