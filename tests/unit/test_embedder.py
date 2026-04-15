"""Tests for HuggingFaceEmbedder dense embedding generation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from research_agent.retrieval.embedder import HuggingFaceEmbedder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embedder(
    *,
    return_value: np.ndarray | None = None,
    expected_dim: int = 3,
) -> tuple[HuggingFaceEmbedder, AsyncMock]:
    """Return an embedder whose feature_extraction call returns *return_value*."""
    client = AsyncMock()
    client.feature_extraction = AsyncMock(
        return_value=return_value if return_value is not None else np.array([0.1, 0.2, 0.3])
    )
    embedder = HuggingFaceEmbedder(client=client, model="test-model", expected_dim=expected_dim)
    return embedder, client


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceEmbedderInputValidation:
    async def test_raises_on_empty_string(self) -> None:
        embedder, _ = make_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed("")

    async def test_raises_on_whitespace_only_string(self) -> None:
        embedder, _ = make_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed("   ")

    async def test_raises_on_newline_only_string(self) -> None:
        embedder, _ = make_embedder()
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed("\n\t")


# ---------------------------------------------------------------------------
# API call forwarding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceEmbedderAPICall:
    async def test_calls_feature_extraction_with_text(self) -> None:
        embedder, client = make_embedder()
        await embedder.embed("hello world")
        client.feature_extraction.assert_called_once_with("hello world", model="test-model")

    async def test_calls_feature_extraction_with_correct_model(self) -> None:
        client = AsyncMock()
        client.feature_extraction = AsyncMock(return_value=np.array([0.1, 0.2]))
        embedder = HuggingFaceEmbedder(
            client=client, model="BAAI/bge-large-en-v1.5", expected_dim=2
        )
        await embedder.embed("query")
        _, kwargs = client.feature_extraction.call_args
        assert kwargs["model"] == "BAAI/bge-large-en-v1.5"


# ---------------------------------------------------------------------------
# Return format
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceEmbedderReturnFormat:
    async def test_returns_list(self) -> None:
        embedder, _ = make_embedder(return_value=np.array([0.1, 0.2, 0.3]), expected_dim=3)
        result = await embedder.embed("hello")
        assert isinstance(result, list)

    async def test_returns_list_of_floats(self) -> None:
        embedder, _ = make_embedder(return_value=np.array([0.1, 0.2, 0.3]), expected_dim=3)
        result = await embedder.embed("hello")
        assert all(isinstance(v, float) for v in result)

    async def test_returns_correct_dimension(self) -> None:
        embedder, _ = make_embedder(return_value=np.array([0.1, 0.2, 0.3]), expected_dim=3)
        result = await embedder.embed("hello")
        assert len(result) == 3

    async def test_values_match_api_response(self) -> None:
        arr = np.array([0.1, 0.2, 0.3])
        embedder, _ = make_embedder(return_value=arr, expected_dim=3)
        result = await embedder.embed("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Shape handling / pooling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceEmbedderShapeHandling:
    async def test_handles_1d_array(self) -> None:
        # (dim,) — already pooled
        arr = np.array([0.1, 0.2, 0.3])
        embedder, _ = make_embedder(return_value=arr, expected_dim=3)
        result = await embedder.embed("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    async def test_handles_2d_batch_array(self) -> None:
        # (1, dim) — batch of one
        arr = np.array([[0.1, 0.2, 0.3]])
        embedder, _ = make_embedder(return_value=arr, expected_dim=3)
        result = await embedder.embed("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    async def test_handles_2d_token_level_array_takes_cls(self) -> None:
        # (seq, dim) — token-level output; CLS token is index 0
        arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        embedder, _ = make_embedder(return_value=arr, expected_dim=3)
        result = await embedder.embed("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    async def test_handles_3d_batched_token_level_array(self) -> None:
        # (1, seq, dim) — batched token-level; first batch, CLS token
        arr = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        embedder, _ = make_embedder(return_value=arr, expected_dim=3)
        result = await embedder.embed("hello")
        assert result == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceEmbedderDimensionValidation:
    async def test_raises_on_dimension_mismatch(self) -> None:
        # API returns 3-dim but we expect 5
        arr = np.array([0.1, 0.2, 0.3])
        embedder, _ = make_embedder(return_value=arr, expected_dim=5)
        with pytest.raises(ValueError, match="[Dd]imension mismatch"):
            await embedder.embed("hello")

    async def test_raises_on_shorter_than_expected(self) -> None:
        arr = np.array([0.1])
        embedder, _ = make_embedder(return_value=arr, expected_dim=1024)
        with pytest.raises(ValueError, match="[Dd]imension mismatch"):
            await embedder.embed("hello")

    async def test_correct_dimension_does_not_raise(self) -> None:
        arr = np.ones(1024, dtype=np.float32)
        embedder, _ = make_embedder(return_value=arr, expected_dim=1024)
        result = await embedder.embed("hello")
        assert len(result) == 1024


# ---------------------------------------------------------------------------
# _pool static method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPool:
    def test_1d_array_returned_unchanged(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = HuggingFaceEmbedder._pool(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, arr)

    def test_2d_array_reduced_to_1d(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = HuggingFaceEmbedder._pool(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_3d_array_reduced_to_1d(self) -> None:
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        result = HuggingFaceEmbedder._pool(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_4d_array_reduced_to_1d(self) -> None:
        arr = np.array([[[[1.0, 2.0]]]])
        result = HuggingFaceEmbedder._pool(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1.0, 2.0])
