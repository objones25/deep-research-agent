"""Mem0-backed implementation of the MemoryService protocol.

Instantiate with an injected ``AsyncMemoryClient`` so the dependency is
swappable in tests and alternative backends::

    from mem0 import AsyncMemoryClient
    from research_agent.memory import Mem0MemoryService

    client = AsyncMemoryClient(api_key=settings.mem0_api_key.get_secret_value())
    memory = Mem0MemoryService(client=client)

The ``session_id`` argument on every method maps directly to Mem0's
``user_id`` parameter, which scopes memories to a single research session.
"""

from __future__ import annotations

from mem0 import AsyncMemoryClient


class Mem0MemoryService:
    """Persistent memory service backed by the Mem0 managed API.

    Satisfies the ``MemoryService`` protocol.

    Parameters
    ----------
    client:
        An initialised ``AsyncMemoryClient`` instance.  Inject this rather
        than constructing it internally so the dependency is testable.
    """

    def __init__(self, client: AsyncMemoryClient) -> None:
        self._client = client

    async def add(self, session_id: str, content: str) -> None:
        """Store *content* as a user message under *session_id*.

        The content is wrapped in a single-element messages list using the
        ``user`` role, which is the format Mem0 expects for plain-text
        memory ingestion.
        """
        await self._client.add(
            [{"role": "user", "content": content}],
            user_id=session_id,
        )

    async def search(self, session_id: str, query: str) -> list[str]:
        """Return memory strings relevant to *query* for *session_id*.

        Mem0 returns a list of result dicts; this method extracts only the
        ``"memory"`` field (the plain-text string) from each entry.
        Entries that are missing the ``"memory"`` key are silently skipped
        — this guards against schema changes in the Mem0 API response.
        """
        results: list[dict[str, object]] = await self._client.search(
            query,
            user_id=session_id,
        )
        return [str(r["memory"]) for r in results if "memory" in r]
