"""Tests for FlashRankReranker cross-encoder reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flashrank import RerankRequest

from research_agent.retrieval.protocols import SearchResult
from research_agent.retrieval.reranker import FlashRankReranker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_search_result(
    *,
    content: str = "test content",
    url: str = "http://example.com",
    score: float = 0.5,
    metadata: dict[str, str] | None = None,
    idx: int = 0,
) -> SearchResult:
    return SearchResult(
        content=content,
        url=url or f"http://example.com/{idx}",
        score=score,
        metadata=metadata or {"key": "value"},
    )


def make_reranker(
    *,
    top_n: int = 5,
    rerank_return: list[dict[str, object]] | None = None,
) -> tuple[FlashRankReranker, MagicMock]:
    ranker = MagicMock()
    ranker.rerank = MagicMock(
        return_value=rerank_return if rerank_return is not None else [{"id": 0, "score": 0.95}]
    )
    return FlashRankReranker(ranker=ranker, top_n=top_n), ranker


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlashRankRerankerEmptyInput:
    async def test_empty_results_returns_empty_list(self) -> None:
        reranker, _ = make_reranker()
        output = await reranker.rerank("query", [])
        assert output == []

    async def test_empty_results_does_not_call_ranker(self) -> None:
        reranker, ranker = make_reranker()
        await reranker.rerank("query", [])
        ranker.rerank.assert_not_called()

    async def test_empty_results_returns_list_type(self) -> None:
        reranker, _ = make_reranker()
        output = await reranker.rerank("query", [])
        assert isinstance(output, list)


# ---------------------------------------------------------------------------
# Passage format forwarded to ranker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlashRankRerankerPassageFormat:
    async def test_rerank_request_carries_correct_query(self) -> None:
        reranker, ranker = make_reranker()
        await reranker.rerank("research question", [make_search_result()])
        request: RerankRequest = ranker.rerank.call_args[0][0]
        assert request.query == "research question"

    async def test_passages_list_has_correct_length(self) -> None:
        reranker, ranker = make_reranker(rerank_return=[])
        results = [make_search_result(content=f"doc {i}") for i in range(3)]
        await reranker.rerank("query", results)
        request: RerankRequest = ranker.rerank.call_args[0][0]
        assert len(request.passages) == 3

    async def test_passages_use_zero_based_integer_ids(self) -> None:
        reranker, ranker = make_reranker(rerank_return=[])
        results = [make_search_result(content=f"doc {i}") for i in range(3)]
        await reranker.rerank("query", results)
        request: RerankRequest = ranker.rerank.call_args[0][0]
        ids = [p["id"] for p in request.passages]
        assert ids == [0, 1, 2]

    async def test_passages_use_result_content_as_text(self) -> None:
        reranker, ranker = make_reranker(rerank_return=[])
        results = [
            SearchResult(content="first doc", url="http://a.com", score=0.5, metadata={}),
            SearchResult(content="second doc", url="http://b.com", score=0.4, metadata={}),
        ]
        await reranker.rerank("query", results)
        request: RerankRequest = ranker.rerank.call_args[0][0]
        texts = [p["text"] for p in request.passages]
        assert texts == ["first doc", "second doc"]


# ---------------------------------------------------------------------------
# top_n truncation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlashRankRerankerTopN:
    async def test_top_n_limits_output_length(self) -> None:
        four_passages = [{"id": i, "score": 0.9 - i * 0.1} for i in range(4)]
        reranker, _ = make_reranker(top_n=2, rerank_return=four_passages)
        results = [make_search_result(idx=i) for i in range(4)]
        output = await reranker.rerank("query", results)
        assert len(output) == 2

    async def test_top_n_larger_than_ranker_output_returns_all(self) -> None:
        two_passages = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.8}]
        reranker, _ = make_reranker(top_n=10, rerank_return=two_passages)
        results = [make_search_result(idx=i) for i in range(2)]
        output = await reranker.rerank("query", results)
        assert len(output) == 2

    async def test_top_n_one_returns_single_result(self) -> None:
        three_passages = [{"id": i, "score": 0.9 - i * 0.1} for i in range(3)]
        reranker, _ = make_reranker(top_n=1, rerank_return=three_passages)
        results = [make_search_result(idx=i) for i in range(3)]
        output = await reranker.rerank("query", results)
        assert len(output) == 1


# ---------------------------------------------------------------------------
# Result correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlashRankRerankerResultCorrectness:
    async def test_score_updated_from_ranker_output(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0, "score": 0.987}])
        results = [make_search_result(score=0.1)]
        output = await reranker.rerank("query", results)
        assert output[0].score == pytest.approx(0.987)

    async def test_original_content_preserved(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0, "score": 0.9}])
        results = [
            SearchResult(content="preserved content", url="http://x.com", score=0.1, metadata={})
        ]
        output = await reranker.rerank("query", results)
        assert output[0].content == "preserved content"

    async def test_original_url_preserved(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0, "score": 0.9}])
        results = [
            SearchResult(content="text", url="http://preserved.com/doc", score=0.1, metadata={})
        ]
        output = await reranker.rerank("query", results)
        assert output[0].url == "http://preserved.com/doc"

    async def test_original_metadata_preserved(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0, "score": 0.9}])
        results = [
            SearchResult(
                content="text",
                url="http://x.com",
                score=0.1,
                metadata={"source": "arxiv", "year": "2023"},
            )
        ]
        output = await reranker.rerank("query", results)
        assert output[0].metadata == {"source": "arxiv", "year": "2023"}

    async def test_missing_score_key_defaults_to_zero(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0}])  # no "score" key
        results = [make_search_result(score=0.7)]
        output = await reranker.rerank("query", results)
        assert output[0].score == pytest.approx(0.0)

    async def test_returns_search_result_instances(self) -> None:
        reranker, _ = make_reranker(rerank_return=[{"id": 0, "score": 0.9}])
        results = [make_search_result()]
        output = await reranker.rerank("query", results)
        assert isinstance(output[0], SearchResult)

    async def test_id_used_to_map_back_to_original_result(self) -> None:
        """Ranker may return passages in any order; id is used to look up original."""
        # Ranker returns result 2 first, then result 0
        reranker, _ = make_reranker(
            top_n=2,
            rerank_return=[
                {"id": 2, "score": 0.9},
                {"id": 0, "score": 0.7},
            ],
        )
        results = [
            SearchResult(content="first", url="http://a.com", score=0.3, metadata={}),
            SearchResult(content="second", url="http://b.com", score=0.2, metadata={}),
            SearchResult(content="third", url="http://c.com", score=0.1, metadata={}),
        ]
        output = await reranker.rerank("query", results)
        assert output[0].content == "third"  # id=2
        assert output[1].content == "first"  # id=0
