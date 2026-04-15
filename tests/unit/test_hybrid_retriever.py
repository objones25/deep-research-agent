"""Tests for HybridRetriever dense+BM25 search with Qdrant-native RRF fusion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models
from structlog.testing import capture_logs

from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.retrieval.protocols import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scored_point(
    *,
    content: str = "test content",
    url: str = "http://example.com",
    score: float = 0.9,
    metadata: dict[str, str] | None = None,
) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "content": content,
        "url": url,
        "metadata": metadata or {},
    }
    return point


def make_retriever(
    *,
    points: list[MagicMock] | None = None,
    dense_vec: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
    collection: str = "test_collection",
) -> tuple[HybridRetriever, AsyncMock, AsyncMock, MagicMock]:
    """Return (retriever, client, embedder, bm25_encoder) with sane defaults."""
    mock_response = MagicMock()
    mock_response.points = points or []
    client = AsyncMock()
    client.query_points = AsyncMock(return_value=mock_response)

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=dense_vec or [0.1, 0.2, 0.3, 0.4])

    bm25_encoder = MagicMock()
    bm25_encoder.encode_query = MagicMock(
        return_value=(sparse_indices or [0, 1], sparse_values or [0.8, 0.6])
    )

    retriever = HybridRetriever(
        client=client,
        collection=collection,
        embedder=embedder,
        bm25_encoder=bm25_encoder,
    )
    return retriever, client, embedder, bm25_encoder


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRetrieverValidation:
    async def test_raises_value_error_for_zero_top_k(self) -> None:
        retriever, *_ = make_retriever()
        with pytest.raises(ValueError, match="positive integer"):
            await retriever.retrieve("query", top_k=0)

    async def test_raises_value_error_for_negative_top_k(self) -> None:
        retriever, *_ = make_retriever()
        with pytest.raises(ValueError, match="positive integer"):
            await retriever.retrieve("query", top_k=-5)

    async def test_positive_top_k_does_not_raise(self) -> None:
        retriever, *_ = make_retriever()
        await retriever.retrieve("query", top_k=1)  # should not raise


# ---------------------------------------------------------------------------
# Dependency call forwarding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRetrieverCalls:
    async def test_embed_called_with_query(self) -> None:
        retriever, _, embedder, _ = make_retriever()
        await retriever.retrieve("my query", top_k=5)
        embedder.embed.assert_called_once_with("my query")

    async def test_encode_query_called_with_query(self) -> None:
        retriever, _, _, bm25 = make_retriever()
        await retriever.retrieve("my query", top_k=5)
        bm25.encode_query.assert_called_once_with("my query")

    async def test_query_points_called_once(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("my query", top_k=5)
        client.query_points.assert_called_once()

    async def test_query_points_uses_correct_collection(self) -> None:
        retriever, client, *_ = make_retriever(collection="my_collection")
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "my_collection"

    async def test_query_points_limit_equals_top_k(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("query", top_k=7)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["limit"] == 7

    async def test_query_uses_rrf_fusion(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        fusion_query = kwargs["query"]
        assert isinstance(fusion_query, models.FusionQuery)
        assert fusion_query.fusion == models.Fusion.RRF

    async def test_prefetch_has_exactly_two_branches(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        assert len(kwargs["prefetch"]) == 2

    async def test_sparse_prefetch_is_first_branch(self) -> None:
        retriever, client, *_ = make_retriever(sparse_indices=[2, 5], sparse_values=[0.7, 0.4])
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        sparse_branch = kwargs["prefetch"][0]
        assert sparse_branch.using == "sparse"

    async def test_sparse_prefetch_carries_correct_indices_and_values(self) -> None:
        indices = [0, 3, 7]
        values = [0.9, 0.6, 0.3]
        retriever, client, _, bm25 = make_retriever()
        bm25.encode_query = MagicMock(return_value=(indices, values))
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        sparse_vec = kwargs["prefetch"][0].query
        assert isinstance(sparse_vec, models.SparseVector)
        assert sparse_vec.indices == indices
        assert sparse_vec.values == values

    async def test_dense_prefetch_is_second_branch(self) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        retriever, client, embedder, _ = make_retriever()
        embedder.embed = AsyncMock(return_value=vec)
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        dense_branch = kwargs["prefetch"][1]
        assert dense_branch.using == "dense"
        assert dense_branch.query == vec

    async def test_prefetch_limits_are_two_times_top_k(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("query", top_k=5)
        kwargs = client.query_points.call_args.kwargs
        for branch in kwargs["prefetch"]:
            assert branch.limit == 10  # 5 * 2

    async def test_prefetch_limits_scale_with_top_k(self) -> None:
        retriever, client, *_ = make_retriever()
        await retriever.retrieve("query", top_k=3)
        kwargs = client.query_points.call_args.kwargs
        for branch in kwargs["prefetch"]:
            assert branch.limit == 6  # 3 * 2


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRetrieverResults:
    async def test_empty_points_returns_empty_list(self) -> None:
        retriever, *_ = make_retriever(points=[])
        results = await retriever.retrieve("query", top_k=5)
        assert results == []

    async def test_returns_list_of_search_results(self) -> None:
        point = make_scored_point()
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_maps_content_from_payload(self) -> None:
        point = make_scored_point(content="important text")
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].content == "important text"

    async def test_maps_url_from_payload(self) -> None:
        point = make_scored_point(url="http://docs.example.com/page")
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].url == "http://docs.example.com/page"

    async def test_maps_score_from_scored_point(self) -> None:
        point = make_scored_point(score=0.732)
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].score == pytest.approx(0.732)

    async def test_maps_metadata_from_payload(self) -> None:
        point = make_scored_point(metadata={"source": "arxiv", "year": "2024"})
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].metadata == {"source": "arxiv", "year": "2024"}

    async def test_none_payload_uses_empty_defaults(self) -> None:
        point = MagicMock()
        point.score = 0.5
        point.payload = None  # triggers `payload or {}`
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].content == ""
        assert results[0].url == ""
        assert results[0].metadata == {}

    async def test_missing_content_defaults_to_empty_string(self) -> None:
        point = MagicMock()
        point.score = 0.5
        point.payload = {"url": "http://x.com", "metadata": {}}  # no 'content'
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].content == ""

    async def test_missing_url_defaults_to_empty_string(self) -> None:
        point = MagicMock()
        point.score = 0.5
        point.payload = {"content": "text", "metadata": {}}  # no 'url'
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].url == ""

    async def test_multiple_points_mapped_in_order(self) -> None:
        points = [make_scored_point(content=f"doc {i}", score=0.9 - i * 0.1) for i in range(3)]
        retriever, *_ = make_retriever(points=points)
        results = await retriever.retrieve("query", top_k=5)
        assert len(results) == 3
        assert results[0].content == "doc 0"
        assert results[1].content == "doc 1"
        assert results[2].content == "doc 2"

    async def test_metadata_values_converted_to_strings(self) -> None:
        """Payload metadata values may be non-string; _to_search_result coerces them."""
        point = MagicMock()
        point.score = 0.5
        point.payload = {"content": "", "url": "", "metadata": {"count": 42, "flag": True}}
        retriever, *_ = make_retriever(points=[point])
        results = await retriever.retrieve("query", top_k=5)
        assert results[0].metadata == {"count": "42", "flag": "True"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHybridRetrieverLogging:
    async def test_logs_hybrid_search_complete_with_num_results(self) -> None:
        points = [make_scored_point(content=f"doc {i}") for i in range(3)]
        retriever, *_ = make_retriever(points=points)
        with capture_logs() as cap:
            await retriever.retrieve("query", top_k=5)
        events = [e["event"] for e in cap]
        assert "hybrid_search_complete" in events
        entry = next(e for e in cap if e["event"] == "hybrid_search_complete")
        assert entry["log_level"] == "info"
        assert entry["num_results"] == 3
        assert "latency_ms" in entry

    async def test_logs_hybrid_search_complete_on_empty_results(self) -> None:
        retriever, *_ = make_retriever(points=[])
        with capture_logs() as cap:
            await retriever.retrieve("query", top_k=5)
        entry = next(e for e in cap if e["event"] == "hybrid_search_complete")
        assert entry["num_results"] == 0
