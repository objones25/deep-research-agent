"""Hybrid retriever: Qdrant dense + BM25 sparse + native RRF fusion.

Dense embeddings and BM25 sparse vectors are generated concurrently for each
query, then sent to Qdrant as a single ``query_points`` call with two prefetch
branches and RRF fusion — no extra round-trips.
"""

from __future__ import annotations

import asyncio

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import ScoredPoint

from research_agent.retrieval.bm25 import BM25Encoder
from research_agent.retrieval.protocols import Embedder, SearchResult


class HybridRetriever:
    """Retrieves documents via hybrid dense+BM25 search with Qdrant-native RRF.

    Args:
        client:       Authenticated :class:`AsyncQdrantClient`.
        collection:   Qdrant collection name (must have ``"dense"`` and ``"sparse"``
                      named vector spaces — see :func:`~.collection.ensure_collection`).
        embedder:     Implements :class:`~.protocols.Embedder`; generates dense vectors.
        bm25_encoder: Fitted :class:`BM25Encoder`; generates sparse vectors.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: Embedder,
        bm25_encoder: BM25Encoder,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._bm25_encoder = bm25_encoder

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* documents relevant to *query*.

        Dense embedding and BM25 sparse encoding are run concurrently via
        ``asyncio.gather``. Qdrant fuses the two ranked lists with RRF.

        Args:
            query: Natural-language research query.
            top_k: Maximum number of results to return (must be > 0).

        Raises:
            ValueError: If *top_k* is not a positive integer.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        results = await asyncio.gather(
            self._embedder.embed(query),
            asyncio.to_thread(self._bm25_encoder.encode_query, query),
        )
        dense_vec: list[float] = results[0]
        sparse_idx: list[int]
        sparse_val: list[float]
        sparse_idx, sparse_val = results[1]

        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_idx, values=sparse_val),
                    using="sparse",
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        return [self._to_search_result(p) for p in response.points]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_search_result(point: ScoredPoint) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            content=str(payload.get("content", "")),
            url=str(payload.get("url", "")),
            score=float(point.score),
            metadata={k: str(v) for k, v in payload.get("metadata", {}).items()},
        )
