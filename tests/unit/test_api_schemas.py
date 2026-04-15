"""Unit tests for research_agent.api.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.api.schemas import (
    CitationResponse,
    HealthCheckResponse,
    ResearchQueryRequest,
    ResearchReportResponse,
)

# ---------------------------------------------------------------------------
# ResearchQueryRequest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchQueryRequest:
    def test_minimal_valid_request(self) -> None:
        req = ResearchQueryRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.session_id is None
        assert req.max_iterations == 5

    def test_full_request(self) -> None:
        req = ResearchQueryRequest(query="What is RAG?", session_id="sess-123", max_iterations=10)
        assert req.session_id == "sess-123"
        assert req.max_iterations == 10

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQueryRequest(query="")

    def test_whitespace_only_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQueryRequest(query="   ")

    def test_max_iterations_too_low_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQueryRequest(query="hello", max_iterations=0)

    def test_max_iterations_too_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchQueryRequest(query="hello", max_iterations=21)

    def test_max_iterations_boundary_min(self) -> None:
        req = ResearchQueryRequest(query="hello", max_iterations=1)
        assert req.max_iterations == 1

    def test_max_iterations_boundary_max(self) -> None:
        req = ResearchQueryRequest(query="hello", max_iterations=20)
        assert req.max_iterations == 20


# ---------------------------------------------------------------------------
# HealthCheckResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheckResponse:
    def test_ok_status(self) -> None:
        resp = HealthCheckResponse(
            status="ok",
            checks={"qdrant": "ok", "mem0": "ok", "llm": "ok"},
            timestamp="2026-04-15T10:00:00Z",
        )
        assert resp.status == "ok"
        assert resp.checks["qdrant"] == "ok"

    def test_degraded_status(self) -> None:
        resp = HealthCheckResponse(
            status="degraded",
            checks={"qdrant": "ok", "mem0": "error"},
            timestamp="2026-04-15T10:00:00Z",
        )
        assert resp.status == "degraded"

    def test_error_status(self) -> None:
        resp = HealthCheckResponse(
            status="error",
            checks={"qdrant": "error", "mem0": "error", "llm": "error"},
            timestamp="2026-04-15T10:00:00Z",
        )
        assert resp.status == "error"

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            HealthCheckResponse(
                status="unknown",
                checks={},
                timestamp="2026-04-15T10:00:00Z",
            )

    def test_serialises_to_dict(self) -> None:
        resp = HealthCheckResponse(
            status="ok",
            checks={"qdrant": "ok"},
            timestamp="2026-04-15T10:00:00Z",
        )
        data = resp.model_dump()
        assert data["status"] == "ok"
        assert data["checks"] == {"qdrant": "ok"}
        assert data["timestamp"] == "2026-04-15T10:00:00Z"


# ---------------------------------------------------------------------------
# CitationResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCitationResponse:
    def test_valid_citation(self) -> None:
        cit = CitationResponse(
            url="https://example.com/paper",
            content_snippet="RAG improves accuracy",
            relevance_score=0.92,
        )
        assert cit.url == "https://example.com/paper"
        assert cit.relevance_score == 0.92


# ---------------------------------------------------------------------------
# ResearchReportResponse
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchReportResponse:
    def test_valid_report(self) -> None:
        resp = ResearchReportResponse(
            query="What is RAG?",
            summary="RAG combines retrieval and generation.",
            citations=[
                CitationResponse(
                    url="https://example.com",
                    content_snippet="RAG is ...",
                    relevance_score=0.9,
                )
            ],
            session_id="sess-abc",
        )
        assert resp.query == "What is RAG?"
        assert len(resp.citations) == 1

    def test_empty_citations_allowed(self) -> None:
        resp = ResearchReportResponse(
            query="What is RAG?",
            summary="No results.",
            citations=[],
            session_id="sess-abc",
        )
        assert resp.citations == []
