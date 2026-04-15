"""Unit tests for the agent-facing value objects in research_agent.models.research.

Covers ResearchQuery, Citation, and ResearchReport — the three frozen dataclasses
added for the agent output layer.  SearchResult, Message, and ToolResult are
exercised through the retrieval, LLM, and tool protocol test suites respectively.
"""

from __future__ import annotations

import dataclasses

import pytest

from research_agent.models.research import Citation, ResearchQuery, ResearchReport

# ---------------------------------------------------------------------------
# ResearchQuery
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchQuery:
    def test_creates_with_required_fields(self) -> None:
        q = ResearchQuery(query="climate change", session_id="sess-1")
        assert q.query == "climate change"
        assert q.session_id == "sess-1"

    def test_default_max_iterations(self) -> None:
        q = ResearchQuery(query="q", session_id="s")
        assert q.max_iterations == 5

    def test_custom_max_iterations(self) -> None:
        q = ResearchQuery(query="q", session_id="s", max_iterations=10)
        assert q.max_iterations == 10

    def test_is_frozen(self) -> None:
        q = ResearchQuery(query="q", session_id="s")
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.query = "other"  # type: ignore[misc]

    def test_max_iterations_frozen(self) -> None:
        q = ResearchQuery(query="q", session_id="s")
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.max_iterations = 99  # type: ignore[misc]

    def test_zero_max_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            ResearchQuery(query="q", session_id="s", max_iterations=0)

    def test_negative_max_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            ResearchQuery(query="q", session_id="s", max_iterations=-1)

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            ResearchQuery(query="", session_id="s")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            ResearchQuery(query="   ", session_id="s")

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            ResearchQuery(query="q", session_id="")

    def test_equality(self) -> None:
        a = ResearchQuery(query="q", session_id="s")
        b = ResearchQuery(query="q", session_id="s")
        assert a == b

    def test_inequality_different_query(self) -> None:
        a = ResearchQuery(query="a", session_id="s")
        b = ResearchQuery(query="b", session_id="s")
        assert a != b


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCitation:
    def test_creates_with_all_fields(self) -> None:
        c = Citation(
            url="https://example.com",
            content_snippet="Some relevant text.",
            relevance_score=0.95,
        )
        assert c.url == "https://example.com"
        assert c.content_snippet == "Some relevant text."
        assert c.relevance_score == 0.95

    def test_is_frozen(self) -> None:
        c = Citation(url="https://example.com", content_snippet="text", relevance_score=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.url = "https://other.com"  # type: ignore[misc]

    def test_zero_relevance_score_accepted(self) -> None:
        c = Citation(url="https://example.com", content_snippet="text", relevance_score=0.0)
        assert c.relevance_score == 0.0

    def test_max_relevance_score_accepted(self) -> None:
        c = Citation(url="https://example.com", content_snippet="text", relevance_score=1.0)
        assert c.relevance_score == 1.0

    def test_equality(self) -> None:
        a = Citation(url="https://x.com", content_snippet="text", relevance_score=0.5)
        b = Citation(url="https://x.com", content_snippet="text", relevance_score=0.5)
        assert a == b

    def test_inequality_different_url(self) -> None:
        a = Citation(url="https://a.com", content_snippet="text", relevance_score=0.5)
        b = Citation(url="https://b.com", content_snippet="text", relevance_score=0.5)
        assert a != b


# ---------------------------------------------------------------------------
# ResearchReport
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchReport:
    def _citation(self, n: int = 1) -> Citation:
        return Citation(
            url=f"https://example.com/{n}",
            content_snippet=f"snippet {n}",
            relevance_score=0.9 - n * 0.1,
        )

    def test_creates_with_all_fields(self) -> None:
        citations = [self._citation(1), self._citation(2)]
        report = ResearchReport(
            query="test query",
            summary="The answer is 42.",
            citations=citations,
            session_id="sess-1",
        )
        assert report.query == "test query"
        assert report.summary == "The answer is 42."
        assert report.citations == citations
        assert report.session_id == "sess-1"

    def test_is_frozen(self) -> None:
        report = ResearchReport(query="q", summary="s", citations=[], session_id="sess")
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.summary = "other"  # type: ignore[misc]

    def test_empty_citations_accepted(self) -> None:
        report = ResearchReport(query="q", summary="s", citations=[], session_id="sess")
        assert report.citations == []

    def test_multiple_citations(self) -> None:
        citations = [self._citation(i) for i in range(5)]
        report = ResearchReport(query="q", summary="s", citations=citations, session_id="sess")
        assert len(report.citations) == 5

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            ResearchReport(query="", summary="s", citations=[], session_id="sess")

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            ResearchReport(query="q", summary="s", citations=[], session_id="")

    def test_equality(self) -> None:
        a = ResearchReport(query="q", summary="s", citations=[], session_id="sess")
        b = ResearchReport(query="q", summary="s", citations=[], session_id="sess")
        assert a == b

    def test_inequality_different_summary(self) -> None:
        a = ResearchReport(query="q", summary="a", citations=[], session_id="sess")
        b = ResearchReport(query="q", summary="b", citations=[], session_id="sess")
        assert a != b
