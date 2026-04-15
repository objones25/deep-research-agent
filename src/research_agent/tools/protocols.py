"""Core protocols and value objects for the tools subsystem.

All external tool dependencies are hidden behind the Tool protocol.
Business logic (agent nodes) must depend only on what is defined here.

Value objects (ToolResult) live in ``research_agent.models.research`` because
they flow up through the layer stack (tools → agent → API).
Input DTOs (ToolInput, SearchInput, ScrapeInput) and ToolExecutionError stay
here because they form the calling contract for the Tool protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from research_agent.models.research import ToolResult

__all__ = [
    "ScrapeInput",
    "SearchInput",
    "Tool",
    "ToolExecutionError",
    "ToolInput",
    "ToolResult",
]


@dataclass(frozen=True)
class ToolInput:
    """Base class for all tool inputs.

    Subclass this to define the input contract for a new tool.  The ``Tool``
    protocol's ``execute`` method accepts any ``ToolInput``, so adding a new
    tool never requires modifying this module — only a new subclass is needed.
    """


@dataclass(frozen=True)
class SearchInput(ToolInput):
    """Input for a web-search tool invocation.

    Raises:
        ValueError: If *query* is empty/whitespace-only or *limit* is not positive.
    """

    query: str
    limit: int = 5

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")
        if self.limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {self.limit!r}")


@dataclass(frozen=True)
class ScrapeInput(ToolInput):
    """Input for a single-URL scrape tool invocation.

    Raises:
        ValueError: If *url* is not an HTTP or HTTPS URL with a valid host.
    """

    url: str
    only_main_content: bool = True

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"url must be an HTTP or HTTPS URL with a valid host, got: {self.url!r}"
            )


class ToolExecutionError(Exception):
    """Raised when a tool's MCP call fails or the remote server returns an error."""


@runtime_checkable
class Tool(Protocol):
    """Protocol for agent tools that wrap external services via MCP.

    Every concrete tool must expose a stable ``name``, a human-readable
    ``description``, and an ``execute`` coroutine that performs the I/O.
    High-level modules depend on this protocol — never on concrete implementations.
    """

    @property
    def name(self) -> str:
        """Stable identifier for this tool (matches the MCP tool name)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary used by the agent to decide when to invoke this tool."""
        ...

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """Execute the tool and return a result.

        Raises:
            TypeError: If *tool_input* is not the expected concrete type for this tool.
            ToolExecutionError: If the underlying MCP call fails.
        """
        ...
