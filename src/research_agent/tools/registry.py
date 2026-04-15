"""Tools provider registry.

Each registered factory returns a tuple of ``Tool`` instances.  A single key
(e.g. ``"firecrawl"``) may produce multiple tools that are all activated together.
Factory functions in implementation modules self-register via
``@tools_registry.register("key")`` at module level — no manual wiring required.
"""

from __future__ import annotations

from research_agent.registry import Registry
from research_agent.tools.protocols import Tool

tools_registry: Registry[tuple[Tool, ...]] = Registry(
    label="tools",
    package="research_agent.tools",
)
