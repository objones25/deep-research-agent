"""FlashRank cross-encoder reranker.

Wraps the synchronous ``flashrank.Ranker`` in ``asyncio.to_thread`` so it
integrates cleanly into the async pipeline without blocking the event loop.
The ``Ranker`` instance is injected at construction time — no model download
happens inside this class, making it straightforward to mock in tests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import cast

from flashrank import Ranker, RerankRequest

from research_agent.logging import get_logger
from research_agent.retrieval.protocols import SearchResult

_log = get_logger(__name__)


class FlashRankReranker:
    """Re-ranks :class:`~.protocols.SearchResult` items using FlashRank.

    Args:
        ranker: Pre-initialised :class:`flashrank.Ranker` instance.
        top_n:  Maximum number of results to return after reranking.
    """

    def __init__(self, ranker: Ranker, top_n: int) -> None:
        self._ranker = ranker
        self._top_n = top_n

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return the top-*n* most relevant results, scores updated.

        FlashRank is synchronous; this method delegates to a thread-pool
        executor via ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            query:   The original research query.
            results: Candidates from :class:`~.hybrid.HybridRetriever`.

        Returns:
            New :class:`SearchResult` instances with FlashRank scores, capped
            at ``top_n``, sorted by score descending. Returns ``[]`` when
            *results* is empty.
        """
        if not results:
            return []

        t0 = time.perf_counter()
        passages = [{"id": i, "text": r.content} for i, r in enumerate(results)]
        request = RerankRequest(query=query, passages=passages)
        reranked: list[dict[str, object]] = await asyncio.to_thread(self._ranker.rerank, request)

        output: list[SearchResult] = []
        for passage in reranked[: self._top_n]:
            original = results[cast(int, passage["id"])]
            score = float(passage.get("score", 0.0))  # type: ignore[arg-type]
            output.append(dataclasses.replace(original, score=score))
        _log.info(
            "rerank_complete",
            input_count=len(results),
            output_count=len(output),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return output


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

from research_agent.config import Settings  # noqa: E402
from research_agent.retrieval.registry import reranker_registry  # noqa: E402


@reranker_registry.register("flashrank")
async def _build_flashrank_reranker(settings: Settings) -> FlashRankReranker:
    import flashrank

    ranker = await asyncio.to_thread(flashrank.Ranker)
    return FlashRankReranker(ranker=ranker, top_n=settings.retrieval_rerank_top_n)
