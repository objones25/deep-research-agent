"""Core protocols and value objects for the retrieval subsystem.

All external dependencies are hidden behind these protocols. Business logic
(agent nodes, API routes) must depend only on what is defined here — never
on concrete implementation classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchResult:
    """An immutable value object representing a single retrieved document.

    ``score`` is always in [0, 1] after reranking; it may be a raw similarity
    score (cosine, dot-product) prior to reranking.
    """

    content: str
    url: str
    score: float
    metadata: dict[str, str]


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
