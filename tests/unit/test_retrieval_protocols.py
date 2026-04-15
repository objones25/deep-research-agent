"""Tests for retrieval protocols and SearchResult value object."""
from __future__ import annotations

import dataclasses

import pytest

from research_agent.retrieval.protocols import Embedder, Reranker, Retriever, SearchResult


@pytest.mark.unit
class TestSearchResult:
    def test_creation_stores_all_fields(self) -> None:
        result = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        assert result.content == "text"
        assert result.url == "http://x.com"
        assert result.score == 0.9
        assert result.metadata == {}

    def test_content_is_immutable(self) -> None:
        result = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.content = "other"  # type: ignore[misc]

    def test_score_is_immutable(self) -> None:
        result = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.score = 0.5  # type: ignore[misc]

    def test_url_is_immutable(self) -> None:
        result = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.url = "http://y.com"  # type: ignore[misc]

    def test_metadata_is_immutable(self) -> None:
        result = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.metadata = {"k": "v"}  # type: ignore[misc]

    def test_replace_creates_new_instance_with_updated_score(self) -> None:
        original = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        updated = dataclasses.replace(original, score=0.5)

        assert updated.score == pytest.approx(0.5)
        assert original.score == pytest.approx(0.9)  # original unchanged

    def test_replace_preserves_unchanged_fields(self) -> None:
        original = SearchResult(content="original", url="http://x.com", score=0.9, metadata={"k": "v"})
        updated = dataclasses.replace(original, score=0.1)

        assert updated.content == "original"
        assert updated.url == "http://x.com"
        assert updated.metadata == {"k": "v"}

    def test_equality_with_same_field_values(self) -> None:
        r1 = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        r2 = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        assert r1 == r2

    def test_inequality_on_different_scores(self) -> None:
        r1 = SearchResult(content="text", url="http://x.com", score=0.9, metadata={})
        r2 = SearchResult(content="text", url="http://x.com", score=0.5, metadata={})
        assert r1 != r2

    def test_inequality_on_different_content(self) -> None:
        r1 = SearchResult(content="a", url="http://x.com", score=0.9, metadata={})
        r2 = SearchResult(content="b", url="http://x.com", score=0.9, metadata={})
        assert r1 != r2

    def test_metadata_with_string_values(self) -> None:
        result = SearchResult(
            content="text", url="http://x.com", score=0.9, metadata={"source": "web", "date": "2024"}
        )
        assert result.metadata == {"source": "web", "date": "2024"}


@pytest.mark.unit
class TestProtocolStructure:
    def test_embedder_satisfied_by_class_with_embed_method(self) -> None:
        class FakeEmbedder:
            async def embed(self, text: str) -> list[float]:
                return []

        assert isinstance(FakeEmbedder(), Embedder)

    def test_embedder_not_satisfied_without_embed_method(self) -> None:
        class Bad:
            pass

        assert not isinstance(Bad(), Embedder)

    def test_embedder_not_satisfied_by_wrong_method_name(self) -> None:
        class Bad:
            async def encode(self, text: str) -> list[float]:
                return []

        assert not isinstance(Bad(), Embedder)

    def test_retriever_satisfied_by_class_with_retrieve_method(self) -> None:
        class FakeRetriever:
            async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
                return []

        assert isinstance(FakeRetriever(), Retriever)

    def test_retriever_not_satisfied_without_retrieve_method(self) -> None:
        class Bad:
            pass

        assert not isinstance(Bad(), Retriever)

    def test_reranker_satisfied_by_class_with_rerank_method(self) -> None:
        class FakeReranker:
            async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
                return []

        assert isinstance(FakeReranker(), Reranker)

    def test_reranker_not_satisfied_without_rerank_method(self) -> None:
        class Bad:
            pass

        assert not isinstance(Bad(), Reranker)

    def test_embedder_not_satisfied_by_retriever(self) -> None:
        class FakeRetriever:
            async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
                return []

        assert not isinstance(FakeRetriever(), Embedder)

    def test_retriever_not_satisfied_by_reranker(self) -> None:
        class FakeReranker:
            async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
                return []

        assert not isinstance(FakeReranker(), Retriever)
