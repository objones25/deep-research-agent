"""Memory service package.

Exports the ``MemoryService`` protocol and the ``Mem0MemoryService``
concrete implementation so callers can import from a single location::

    from research_agent.memory import MemoryService, Mem0MemoryService
"""

from research_agent.memory.mem0 import Mem0MemoryService
from research_agent.memory.protocols import MemoryService

__all__ = ["MemoryService", "Mem0MemoryService"]
