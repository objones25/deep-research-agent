"""Qdrant collection management helpers.

Provides an idempotent helper for ensuring that the hybrid-search collection
(named dense + sparse vectors) exists before any retrieval or ingestion work
begins.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from research_agent.logging import get_logger

_log = get_logger(__name__)


async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    vector_size: int,
) -> None:
    """Create the hybrid-search collection if it does not already exist.

    Idempotent: calling this multiple times with the same arguments is safe.

    The collection is created with two named vector spaces:
    - ``"dense"``  — cosine-similarity dense vectors of dimension *vector_size*.
    - ``"sparse"`` — BM25 sparse vectors (Qdrant native sparse index).

    Args:
        client:      Authenticated :class:`AsyncQdrantClient`.
        name:        Collection name (from ``Settings.qdrant_collection``).
        vector_size: Dense vector dimensionality (from ``Settings.qdrant_vector_size``).
    """
    if await client.collection_exists(name):
        _log.info(
            "collection_status",
            collection_name=name,
            exists=True,
            created=False,
            vector_size=vector_size,
        )
        return

    await client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    _log.info(
        "collection_status",
        collection_name=name,
        exists=False,
        created=True,
        vector_size=vector_size,
    )
