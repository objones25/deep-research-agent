"""Tests for BM25Encoder sparse vector encoding."""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from research_agent.retrieval.bm25 import BM25Encoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fitted_encoder(corpus: list[str] | None = None) -> BM25Encoder:
    enc = BM25Encoder()
    enc.fit(corpus or ["hello world", "foo bar baz", "hello foo world"])
    return enc


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderFit:
    def test_raises_value_error_on_empty_corpus(self) -> None:
        enc = BM25Encoder()
        with pytest.raises(ValueError, match="empty corpus"):
            enc.fit([])

    def test_fit_with_single_document_succeeds(self) -> None:
        enc = BM25Encoder()
        enc.fit(["only document"])
        idx, val = enc.encode_query("only")
        assert len(idx) > 0

    def test_fit_builds_vocabulary_for_all_unique_terms(self) -> None:
        enc = BM25Encoder()
        enc.fit(["apple banana", "cherry banana"])
        # All three unique tokens should be in vocab
        for term in ["apple", "banana", "cherry"]:
            idx, val = enc.encode_query(term)
            assert len(idx) > 0, f"'{term}' should be in vocabulary"

    def test_fit_replaces_previous_state_on_second_call(self) -> None:
        enc = BM25Encoder()
        enc.fit(["first corpus only"])
        enc.fit(["completely different text"])
        # 'first', 'corpus', 'only' should all be gone
        for term in ["first", "corpus", "only"]:
            idx, val = enc.encode_query(term)
            assert idx == [], f"'{term}' should no longer be in vocabulary after refit"

    def test_fit_normalises_case_via_tokenizer(self) -> None:
        enc = BM25Encoder()
        enc.fit(["Hello World"])
        # After lowercasing, 'hello' and 'world' should be in vocab
        idx, val = enc.encode_query("hello")
        assert len(idx) > 0


# ---------------------------------------------------------------------------
# Not-fitted guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderNotFitted:
    def test_encode_document_raises_runtime_error_before_fit(self) -> None:
        enc = BM25Encoder()
        with pytest.raises(RuntimeError, match="fitted"):
            enc.encode_document("hello world")

    def test_encode_query_raises_runtime_error_before_fit(self) -> None:
        enc = BM25Encoder()
        with pytest.raises(RuntimeError, match="fitted"):
            enc.encode_query("hello world")


# ---------------------------------------------------------------------------
# encode_document
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderEncodeDocument:
    def test_returns_tuple_of_two_lists(self) -> None:
        enc = fitted_encoder()
        result = enc.encode_document("hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_indices_and_values_have_equal_length(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello world")
        assert len(idx) == len(val)

    def test_indices_are_integers(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello world")
        assert all(isinstance(i, int) for i in idx)

    def test_values_are_floats(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello world")
        assert all(isinstance(v, float) for v in val)

    def test_values_are_floats_and_finite(self) -> None:
        # BM25Okapi IDF = log((N-n+0.5)/(n+0.5)); common terms can be negative.
        # The contract is float-valued, not sign-constrained.
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello world")
        import math

        assert all(isinstance(v, float) and math.isfinite(v) for v in val)

    def test_oov_only_document_returns_empty_lists(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("unknownterm_xyz_abc")
        assert idx == []
        assert val == []

    def test_empty_string_returns_empty_lists(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("")
        assert idx == []
        assert val == []

    def test_whitespace_only_returns_empty_lists(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("   ")
        assert idx == []
        assert val == []

    def test_known_terms_produce_results(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello")
        assert len(idx) > 0
        assert len(val) > 0

    def test_no_duplicate_indices(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_document("hello hello world world foo")
        # Each vocab index should appear at most once (terms aggregated)
        assert len(idx) == len(set(idx))

    def test_mixed_known_and_oov_terms_only_includes_known(self) -> None:
        enc = fitted_encoder()
        # 'hello' is known, 'zzzzzunknown' is not in the vocabulary.
        # doc_len changes when OOV tokens are present, so TF values differ;
        # but the *indices* must be identical (only known terms included).
        idx_both, _ = enc.encode_document("hello zzzzzunknown")
        idx_known, _ = enc.encode_document("hello")
        assert set(idx_both) == set(idx_known)


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderEncodeQuery:
    def test_returns_tuple_of_two_lists(self) -> None:
        enc = fitted_encoder()
        result = enc.encode_query("hello")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_known_query_term_returns_results(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_query("hello")
        assert len(idx) > 0
        assert len(val) > 0

    def test_values_are_floats_for_known_terms(self) -> None:
        # BM25 IDF can be negative for common terms; we assert float type only.
        enc = fitted_encoder()
        idx, val = enc.encode_query("foo")
        assert all(isinstance(v, float) for v in val)

    def test_rare_term_has_positive_idf(self) -> None:
        # "baz" appears in exactly 1 of 3 corpus docs → IDF > 0.
        enc = fitted_encoder()
        idx, val = enc.encode_query("baz")
        assert len(val) == 1
        assert val[0] > 0.0

    def test_all_oov_query_returns_empty_lists(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_query("completely_unknown_xyz_abc")
        assert idx == []
        assert val == []

    def test_empty_string_returns_empty_lists(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_query("")
        assert idx == []
        assert val == []

    def test_repeated_terms_produce_single_entry_per_term(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_query("hello hello hello")
        # No duplicate indices — each known term appears at most once
        assert len(idx) == len(set(idx))

    def test_indices_and_values_have_equal_length(self) -> None:
        enc = fitted_encoder()
        idx, val = enc.encode_query("hello world foo")
        assert len(idx) == len(val)

    def test_mixed_known_and_oov_excludes_oov(self) -> None:
        enc = fitted_encoder()
        idx_mixed, _ = enc.encode_query("hello unknownzzz")
        idx_known, _ = enc.encode_query("hello")
        assert set(idx_mixed) == set(idx_known)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderTokenize:
    def test_lowercases_input(self) -> None:
        tokens = BM25Encoder._tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_splits_on_non_word_characters(self) -> None:
        tokens = BM25Encoder._tokenize("foo-bar.baz")
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens

    def test_drops_empty_tokens(self) -> None:
        tokens = BM25Encoder._tokenize("   ")
        assert tokens == []

    def test_empty_string_returns_empty_list(self) -> None:
        tokens = BM25Encoder._tokenize("")
        assert tokens == []

    def test_returns_list_of_strings(self) -> None:
        tokens = BM25Encoder._tokenize("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBM25EncoderLogging:
    def test_logs_bm25_index_fitted_on_fit(self) -> None:
        enc = BM25Encoder()
        with capture_logs() as cap:
            enc.fit(["hello world", "foo bar baz"])
        events = [e["event"] for e in cap]
        assert "bm25_index_fitted" in events
        entry = next(e for e in cap if e["event"] == "bm25_index_fitted")
        assert entry["log_level"] == "info"
        assert entry["corpus_size"] == 2
        assert "vocab_size" in entry

    def test_logs_sparse_query_empty_on_oov_query(self) -> None:
        enc = BM25Encoder()
        enc.fit(["hello world", "foo bar"])
        with capture_logs() as cap:
            enc.encode_query("completely_unknown_xyz_abc")
        events = [e["event"] for e in cap]
        assert "sparse_query_empty" in events
        entry = next(e for e in cap if e["event"] == "sparse_query_empty")
        assert entry["log_level"] == "debug"

    def test_no_sparse_query_empty_log_for_known_term(self) -> None:
        enc = BM25Encoder()
        enc.fit(["hello world"])
        with capture_logs() as cap:
            enc.encode_query("hello")
        assert "sparse_query_empty" not in [e["event"] for e in cap]
