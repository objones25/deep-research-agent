"""Dense text embedder backed by the HuggingFace Inference API.

Uses ``AsyncInferenceClient.feature_extraction`` to generate embeddings.
The embedding model is configured via :class:`~research_agent.config.Settings`
(``EMBEDDING_MODEL`` env var, default ``BAAI/bge-large-en-v1.5``).

Output shape normalisation handles the three forms returned by the HF API:
- ``(dim,)``      — already pooled, used directly
- ``(1, dim)``    — batch of one, first element taken
- ``(seq, dim)``  — token-level, CLS token (index 0) taken
- ``(1, seq, dim)`` — batched token-level, first batch CLS token taken
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from huggingface_hub import AsyncInferenceClient


class HuggingFaceEmbedder:
    """Generates dense embeddings via the HuggingFace Inference API.

    Args:
        client:       Authenticated :class:`AsyncInferenceClient` instance.
        model:        HuggingFace model ID (e.g. ``"BAAI/bge-large-en-v1.5"``).
        expected_dim: Expected embedding dimensionality; validated on every call.
    """

    def __init__(
        self,
        client: AsyncInferenceClient,
        model: str,
        expected_dim: int,
    ) -> None:
        self._client = client
        self._model = model
        self._expected_dim = expected_dim

    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*.

        Args:
            text: Non-empty input string.

        Returns:
            A ``list[float]`` of length ``expected_dim``.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
            ValueError: If the API returns a vector of the wrong dimension.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")

        raw: Any = await self._client.feature_extraction(text, model=self._model)
        embedding = self._pool(np.asarray(raw, dtype=np.float32))

        if len(embedding) != self._expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self._expected_dim}, got {len(embedding)}. "
                f"Check that EMBEDDING_MODEL and QDRANT_VECTOR_SIZE are consistent."
            )

        return cast(list[float], embedding.tolist())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(arr: np.ndarray) -> np.ndarray:
        """Reduce an n-dimensional array to 1-D by taking index 0 at each extra dim."""
        while arr.ndim > 1:
            arr = arr[0]
        return arr
