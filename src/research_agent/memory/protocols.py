"""Memory service protocol.

Any class that satisfies ``MemoryService`` can be used as the persistent
memory backend for the research agent.  The protocol is intentionally
minimal (ISP) — callers depend only on ``add`` and ``search``.

Inject a concrete implementation via the constructor of any class that
needs memory — never import ``Mem0MemoryService`` directly from business
logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryService(Protocol):
    """Protocol for a session-scoped persistent memory store.

    All I/O methods are ``async``; implementations must not block the
    event loop.

    Terminology
    -----------
    session_id
        An opaque string that scopes memories to a single research
        session (or user, depending on the caller's convention).  It is
        passed directly to the underlying store as the entity identifier.
    """

    async def add(self, session_id: str, content: str) -> None:
        """Persist *content* under *session_id*.

        Parameters
        ----------
        session_id:
            Identifier for the session or user whose memory is being
            updated.  Must be non-empty.
        content:
            Plain-text string to store.  The implementation decides how
            to chunk or format it for the underlying store.
        """
        ...

    async def search(self, session_id: str, query: str) -> list[str]:
        """Return stored memories relevant to *query* for *session_id*.

        Parameters
        ----------
        session_id:
            Scopes the search to memories belonging to this session.
        query:
            Natural-language query used to retrieve relevant memories.

        Returns
        -------
        list[str]
            Ordered list of memory strings, most relevant first.
            Returns an empty list when no memories match.
        """
        ...
