"""Tools subsystem: wraps external services (Firecrawl MCP) behind the Tool protocol."""

from research_agent.tools.firecrawl import FirecrawlScrapeTool, FirecrawlSearchTool
from research_agent.tools.protocols import (
    ScrapeInput,
    SearchInput,
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
)

__all__ = [
    "FirecrawlScrapeTool",
    "FirecrawlSearchTool",
    "ScrapeInput",
    "SearchInput",
    "Tool",
    "ToolExecutionError",
    "ToolInput",
    "ToolResult",
]
