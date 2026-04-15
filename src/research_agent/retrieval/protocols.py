"""Core protocols for the retrieval subsystem.

All external dependencies are hidden behind these protocols. Business logic
(agent nodes, API routes) must depend only on what is defined here — never
on concrete implementation classes.

Value objects (SearchResult) live in ``research_agent.models.research`` because
they flow up through the layer stack (retrieval → agent → API).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from research_agent.models.research import SearchResult

__all__ = ["Embedder", "Reranker", "Retriever", "SearchResult"]


@runtime_checkable
class Embedder(Protocol):
    """Converts text to a dense float vector."""

    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding for *text*.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves candidate documents for a query."""

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* documents relevant to *query*.

        Raises:
            ValueError: If *top_k* is not a positive integer.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-orders a list of ``SearchResult`` by relevance to a query."""

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return *results* sorted by relevance to *query*, highest first.

        Returns an empty list when *results* is empty.
        """
        ...
