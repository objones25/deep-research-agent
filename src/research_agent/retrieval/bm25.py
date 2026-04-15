"""BM25-based sparse vector encoder.

Converts document and query text into sparse ``(indices, values)`` pairs
suitable for storage in and querying against Qdrant's native sparse vector
index. The vocabulary is built once via :meth:`BM25Encoder.fit` and then
remains fixed — new terms encountered after fitting are silently ignored.

This class is synchronous. Call it via ``asyncio.to_thread`` when used
inside an async context.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


class BM25Encoder:
    """Converts text to BM25 sparse vectors using a fixed vocabulary.

    Usage::

        encoder = BM25Encoder()
        encoder.fit(["The quick brown fox", "Jumped over the lazy dog"])

        idx, val = encoder.encode_document("The fox jumped")
        q_idx, q_val = encoder.encode_query("quick brown")
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Build the vocabulary and IDF table from *texts*.

        Can be called multiple times to re-fit on a new corpus; previous
        state is fully replaced.

        Args:
            texts: Non-empty list of raw document strings.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError(
                "Cannot fit BM25Encoder on an empty corpus. " "Provide at least one document."
            )
        corpus = [self._tokenize(t) for t in texts]
        bm25 = BM25Okapi(corpus)

        all_terms = sorted({token for doc in corpus for token in doc})
        self._vocab = {term: idx for idx, term in enumerate(all_terms)}
        self._idf = dict(bm25.idf)
        self._fitted = True

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Convert a document to a BM25 sparse vector.

        Returns ``(indices, values)`` where ``indices[i]`` is the vocabulary
        index for a term and ``values[i]`` is its IDF-weighted TF score.
        Terms absent from the vocabulary are silently excluded.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted()
        tokens = self._tokenize(text)
        doc_len = max(len(tokens), 1)

        raw_tf: dict[str, int] = {}
        for token in tokens:
            raw_tf[token] = raw_tf.get(token, 0) + 1

        indices: list[int] = []
        values: list[float] = []
        for term, count in raw_tf.items():
            if term not in self._vocab:
                continue
            idf = self._idf.get(term, 0.0)
            tf_norm = count / doc_len
            indices.append(self._vocab[term])
            values.append(float(idf * tf_norm))

        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Convert a query to a BM25 sparse vector.

        Each known query term is assigned its IDF weight (maximum across
        repeated occurrences). Unknown terms are silently excluded.

        Returns ``([], [])`` when no query token appears in the vocabulary.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fitted()
        tokens = self._tokenize(text)

        best: dict[int, float] = {}
        for token in tokens:
            if token not in self._vocab:
                continue
            idx = self._vocab[token]
            idf = self._idf.get(token, 1.0)
            best[idx] = max(best.get(idx, 0.0), float(idf))

        if not best:
            return [], []
        return list(best.keys()), list(best.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BM25Encoder must be fitted before encoding. "
                "Call fit(texts) with a representative corpus first."
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-word characters, drop empty tokens."""
        return [t for t in re.split(r"\W+", text.lower()) if t]
