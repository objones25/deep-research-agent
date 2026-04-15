"""Qdrant collection management helpers.

Provides an idempotent helper for ensuring that the hybrid-search collection
(named dense + sparse vectors) exists before any retrieval or ingestion work
begins.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models


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
