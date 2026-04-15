"""Integration tests for HybridRetriever against a real Qdrant instance.

Skipped automatically when ``QDRANT_URL`` is not set in the environment.

To run locally::

    docker compose up qdrant -d
    uv run pytest tests/integration/ -v -m integration
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient, models

from research_agent.retrieval.bm25 import BM25Encoder
from research_agent.retrieval.collection import ensure_collection
from research_agent.retrieval.hybrid import HybridRetriever
from research_agent.retrieval.protocols import SearchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QDRANT_URL = os.getenv("QDRANT_URL", "")
_VECTOR_DIM = 8  # intentionally small to keep tests fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _require_qdrant() -> None:
    """Skip all tests in this module when QDRANT_URL is not configured."""
    if not _QDRANT_URL:
        pytest.skip("QDRANT_URL not set — skipping Qdrant integration tests")


# ---------------------------------------------------------------------------
# Fake embedder (no HF API)
# ---------------------------------------------------------------------------


class _DeterministicEmbedder:
    """Returns a unit-normalised vector derived from the text's hash.

    Deterministic and consistent across calls: the same text always produces
    the same embedding. No external API call required.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(seed=hash(text) % (2**32))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ingest(
    client: AsyncQdrantClient,
    collection: str,
    corpus: list[str],
    embedder: _DeterministicEmbedder,
    bm25: BM25Encoder,
) -> None:
    """Upsert *corpus* documents into *collection*."""
    points: list[models.PointStruct] = []
    for i, text in enumerate(corpus):
        dense_vec = await embedder.embed(text)
        sparse_idx, sparse_val = bm25.encode_document(text)
        points.append(
            models.PointStruct(
                id=i,
                payload={
                    "content": text,
                    "url": f"http://example.com/{i}",
                    "metadata": {"index": str(i)},
                },
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_idx,
                        values=sparse_val,
                    ),
                },
            )
        )
    await client.upsert(collection_name=collection, points=points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    async def test_retrieve_returns_search_result_instances(self) -> None:
        collection = f"integ_{uuid.uuid4().hex[:8]}"
        corpus = [
            "machine learning is a subset of artificial intelligence",
            "deep learning uses neural networks with many layers",
            "natural language processing handles text understanding",
        ]
        bm25 = BM25Encoder()
        bm25.fit(corpus)
        embedder = _DeterministicEmbedder(dim=_VECTOR_DIM)
        client = AsyncQdrantClient(url=_QDRANT_URL)

        try:
            await ensure_collection(client, collection, vector_size=_VECTOR_DIM)
            await _ingest(client, collection, corpus, embedder, bm25)

            retriever = HybridRetriever(
                client=client,
                collection=collection,
                embedder=embedder,
                bm25_encoder=bm25,
            )
            results = await retriever.retrieve("machine learning neural networks", top_k=3)

            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
        finally:
            await client.delete_collection(collection)
            await client.close()

    async def test_retrieve_results_have_non_empty_content(self) -> None:
        collection = f"integ_{uuid.uuid4().hex[:8]}"
        corpus = [
            "alpha beta gamma delta",
            "epsilon zeta eta theta",
            "iota kappa lambda mu",
        ]
        bm25 = BM25Encoder()
        bm25.fit(corpus)
        embedder = _DeterministicEmbedder(dim=_VECTOR_DIM)
        client = AsyncQdrantClient(url=_QDRANT_URL)

        try:
            await ensure_collection(client, collection, vector_size=_VECTOR_DIM)
            await _ingest(client, collection, corpus, embedder, bm25)

            retriever = HybridRetriever(
                client=client,
                collection=collection,
                embedder=embedder,
                bm25_encoder=bm25,
            )
            results = await retriever.retrieve("alpha epsilon", top_k=3)
            assert all(r.content != "" for r in results)
        finally:
            await client.delete_collection(collection)
            await client.close()

    async def test_retrieve_with_top_k_1_returns_exactly_one_result(self) -> None:
        collection = f"integ_{uuid.uuid4().hex[:8]}"
        corpus = ["doc one content", "doc two content", "doc three content"]
        bm25 = BM25Encoder()
        bm25.fit(corpus)
        embedder = _DeterministicEmbedder(dim=_VECTOR_DIM)
        client = AsyncQdrantClient(url=_QDRANT_URL)

        try:
            await ensure_collection(client, collection, vector_size=_VECTOR_DIM)
            await _ingest(client, collection, corpus, embedder, bm25)

            retriever = HybridRetriever(
                client=client,
                collection=collection,
                embedder=embedder,
                bm25_encoder=bm25,
            )
            results = await retriever.retrieve("doc content", top_k=1)
            assert len(results) == 1
        finally:
            await client.delete_collection(collection)
            await client.close()

    async def test_ensure_collection_idempotent_with_real_client(self) -> None:
        collection = f"integ_{uuid.uuid4().hex[:8]}"
        client = AsyncQdrantClient(url=_QDRANT_URL)

        try:
            # Call twice — second call must not raise
            await ensure_collection(client, collection, vector_size=_VECTOR_DIM)
            await ensure_collection(client, collection, vector_size=_VECTOR_DIM)
            exists = await client.collection_exists(collection)
            assert exists
        finally:
            await client.delete_collection(collection)
            await client.close()
