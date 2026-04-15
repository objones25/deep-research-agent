"""Integration tests for the FastAPI API layer.

Skipped automatically when ``QDRANT_URL`` is not set in the environment —
the health route probes the real Qdrant instance.

To run locally::

    docker compose up qdrant -d
    uv run pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from research_agent.api.auth import TokenPayload
from research_agent.api.dependencies import check_rate_limit, get_auth
from research_agent.models.research import Citation, ResearchReport

# ---------------------------------------------------------------------------
# Module-level guard — skip when Qdrant is unavailable
# ---------------------------------------------------------------------------

_QDRANT_URL = os.getenv("QDRANT_URL", "")


@pytest.fixture(autouse=True)
def _require_qdrant() -> None:
    """Skip all tests in this module when QDRANT_URL is not configured."""
    if not _QDRANT_URL:
        pytest.skip("QDRANT_URL not set — skipping API integration tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide minimum required env vars so Settings initialises cleanly."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
    monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
    monkeypatch.setenv("SECRET_KEY", "a" * 64)
    monkeypatch.setenv("QDRANT_URL", _QDRANT_URL)


def _make_runner(summary: str = "integration test summary") -> AsyncMock:
    runner = AsyncMock()
    runner.run.return_value = ResearchReport(
        query="integration query",
        summary=summary,
        citations=[
            Citation(
                url="https://example.com",
                content_snippet="snippet",
                relevance_score=0.9,
            )
        ],
        session_id="integ-session",
    )
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHealthIntegration:
    def test_health_returns_200_with_real_qdrant(self, _env: None) -> None:
        """GET /health returns 200 and qdrant=ok when Qdrant is reachable."""
        from research_agent.api.main import create_app

        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["qdrant"] == "ok"

    def test_health_timestamp_is_present(self, _env: None) -> None:
        """GET /health always includes a non-empty ISO 8601 timestamp."""
        from research_agent.api.main import create_app

        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        ts = response.json().get("timestamp", "")
        assert isinstance(ts, str) and len(ts) > 0


@pytest.mark.integration
class TestResearchIntegration:
    def test_research_returns_report_with_mocked_runner(self, _env: None) -> None:
        """POST /research returns 200 with auth bypassed and a mock AgentRunner."""
        from research_agent.api.main import create_app

        runner = _make_runner()
        app = create_app(agent_runner=runner)

        async def _always_auth() -> TokenPayload:
            return TokenPayload(sub="integ-user")

        async def _always_allowed() -> None:
            return None

        app.dependency_overrides[get_auth] = _always_auth
        app.dependency_overrides[check_rate_limit] = _always_allowed

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "integration query"})

        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == "integration test summary"
        assert len(body["citations"]) == 1

    def test_research_requires_auth_header(self, _env: None) -> None:
        """POST /research returns 401 when no Authorization header is provided."""
        from research_agent.api.main import create_app

        app = create_app(agent_runner=_make_runner())
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/research", json={"query": "test"})

        assert response.status_code == 401
