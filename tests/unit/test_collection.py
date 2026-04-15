"""Tests for ensure_collection Qdrant collection management helper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from qdrant_client import models
from structlog.testing import capture_logs

from research_agent.retrieval.collection import ensure_collection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(*, collection_exists: bool = False) -> AsyncMock:
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=collection_exists)
    client.create_collection = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnsureCollection:
    async def test_does_not_create_when_collection_already_exists(self) -> None:
        client = make_client(collection_exists=True)
        await ensure_collection(client, "existing_coll", vector_size=512)
        client.create_collection.assert_not_called()

    async def test_checks_existence_of_the_given_collection_name(self) -> None:
        client = make_client(collection_exists=True)
        await ensure_collection(client, "my_collection", vector_size=512)
        client.collection_exists.assert_called_once_with("my_collection")

    async def test_creates_collection_when_not_exists(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "new_coll", vector_size=512)
        client.create_collection.assert_called_once()

    async def test_create_uses_correct_collection_name(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "my_collection", vector_size=512)
        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "my_collection"

    async def test_create_includes_dense_vector_key(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=512)
        kwargs = client.create_collection.call_args.kwargs
        assert "dense" in kwargs["vectors_config"]

    async def test_create_dense_uses_provided_vector_size(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=768)
        kwargs = client.create_collection.call_args.kwargs
        dense_cfg: models.VectorParams = kwargs["vectors_config"]["dense"]
        assert dense_cfg.size == 768

    async def test_create_dense_uses_cosine_distance(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=512)
        kwargs = client.create_collection.call_args.kwargs
        dense_cfg: models.VectorParams = kwargs["vectors_config"]["dense"]
        assert dense_cfg.distance == models.Distance.COSINE

    async def test_create_includes_sparse_vector_key(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=512)
        kwargs = client.create_collection.call_args.kwargs
        assert "sparse" in kwargs["sparse_vectors_config"]

    async def test_sparse_config_is_sparse_vector_params(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=512)
        kwargs = client.create_collection.call_args.kwargs
        sparse_cfg = kwargs["sparse_vectors_config"]["sparse"]
        assert isinstance(sparse_cfg, models.SparseVectorParams)

    async def test_idempotent_second_call_does_not_create_again(self) -> None:
        """Calling twice is safe: second call short-circuits on existence check."""
        client = AsyncMock()
        client.collection_exists = AsyncMock(side_effect=[False, True])
        client.create_collection = AsyncMock()

        await ensure_collection(client, "test", vector_size=512)
        await ensure_collection(client, "test", vector_size=512)

        client.create_collection.assert_called_once()

    async def test_vector_size_1024_propagated_correctly(self) -> None:
        client = make_client(collection_exists=False)
        await ensure_collection(client, "test", vector_size=1024)
        kwargs = client.create_collection.call_args.kwargs
        dense_cfg: models.VectorParams = kwargs["vectors_config"]["dense"]
        assert dense_cfg.size == 1024


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnsureCollectionLogging:
    async def test_logs_collection_status_when_already_exists(self) -> None:
        client = make_client(collection_exists=True)
        with capture_logs() as cap:
            await ensure_collection(client, "my_coll", vector_size=512)
        events = [e["event"] for e in cap]
        assert "collection_status" in events
        entry = next(e for e in cap if e["event"] == "collection_status")
        assert entry["log_level"] == "info"
        assert entry["collection_name"] == "my_coll"
        assert entry["exists"] is True
        assert entry["created"] is False

    async def test_logs_collection_status_when_created(self) -> None:
        client = make_client(collection_exists=False)
        with capture_logs() as cap:
            await ensure_collection(client, "new_coll", vector_size=768)
        events = [e["event"] for e in cap]
        assert "collection_status" in events
        entry = next(e for e in cap if e["event"] == "collection_status")
        assert entry["log_level"] == "info"
        assert entry["collection_name"] == "new_coll"
        assert entry["exists"] is False
        assert entry["created"] is True
        assert entry["vector_size"] == 768
