"""Memory provider registry.

Factory functions in implementation modules (e.g. ``mem0.py``) self-register
via ``@memory_registry.register("key")`` at module level.  The registry imports
those modules automatically on the first ``build()`` call — no manual wiring required.
"""

from __future__ import annotations

from research_agent.memory.protocols import MemoryService
from research_agent.registry import Registry

memory_registry: Registry[MemoryService] = Registry(
    label="memory",
    package="research_agent.memory",
)
