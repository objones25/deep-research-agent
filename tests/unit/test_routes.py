"""Unit tests for GET /health and POST /research route handlers.

All external I/O is mocked:
  - httpx.AsyncClient is patched for health probes.
  - get_auth / check_rate_limit are overridden via dependency_overrides.
  - AgentRunner is replaced with an AsyncMock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from research_agent.api.auth import TokenPayload
from research_agent.api.dependencies import check_rate_limit, get_auth
from research_agent.api.main import create_app
from research_agent.models.research import Citation, ResearchQuery, ResearchReport

# ---------------------------------------------------------------------------
# Module-level fixture: provide minimum env vars for get_settings() to succeed.
# The conftest autouse fixture clears these first; this fixture re-adds them.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set minimum env vars so create_app() / get_settings() succeeds in unit tests."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
    monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
    monkeypatch.setenv("SECRET_KEY", "a" * 64)  # 64 chars = 64 bytes > 32 minimum


@pytest.fixture()
def mock_runner() -> AsyncMock:
    """Minimal AgentRunner mock — prevents _build_agent_runner from running in lifespan."""
    return _make_runner()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_app_with_auth_bypassed(agent_runner: object = None) -> object:
    """Return a TestClient-ready app with auth + rate-limit dependencies overridden."""
    app = create_app(agent_runner=agent_runner)  # type: ignore[arg-type]

    async def _always_authenticated() -> TokenPayload:
        return TokenPayload(sub="test-user")

    async def _always_allowed() -> None:
        return None

    app.dependency_overrides[get_auth] = _always_authenticated
    app.dependency_overrides[check_rate_limit] = _always_allowed
    return app


def _make_runner(report: ResearchReport | None = None) -> AsyncMock:
    """Return an AsyncMock AgentRunner that returns *report* (or a default)."""
    runner = AsyncMock()
    if report is None:
        report = ResearchReport(
            query="test query",
            summary="test summary",
            citations=[
                Citation(
                    url="https://example.com",
                    content_snippet="snippet",
                    relevance_score=0.9,
                )
            ],
            session_id="test-session",
        )
    runner.run.return_value = report
    return runner


def _mock_httpx_success() -> tuple[object, object]:
    """Return (patch_target, patcher kwargs) for a successful httpx probe.

    Usage::

        with _httpx_success_ctx() as (mock_cls, mock_client):
            ...
    """
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_ctx, mock_client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthRoute:
    def test_health_returns_200_when_qdrant_reachable(self, mock_runner: AsyncMock) -> None:
        """Health returns 200 and status='ok' when Qdrant responds 200."""
        app = create_app(agent_runner=mock_runner)
        mock_ctx, _client = _mock_httpx_success()

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["checks"]["qdrant"] == "ok"
        assert "timestamp" in body

    def test_health_returns_503_when_qdrant_connect_error(self, mock_runner: AsyncMock) -> None:
        """Health returns 503 when Qdrant raises a connection error."""
        import httpx

        app = create_app(agent_runner=mock_runner)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            response = client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["checks"]["qdrant"] == "error"

    def test_health_returns_503_when_qdrant_timeout(self, mock_runner: AsyncMock) -> None:
        """Health returns 503 when the Qdrant probe times out."""
        import httpx

        app = create_app(agent_runner=mock_runner)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            response = client.get("/health")

        assert response.status_code == 503

    def test_health_response_includes_iso_timestamp(self, mock_runner: AsyncMock) -> None:
        """Health response always contains a non-empty ISO 8601 timestamp."""
        app = create_app(agent_runner=mock_runner)
        mock_ctx, _client = _mock_httpx_success()

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            response = client.get("/health")

        assert response.status_code == 200
        ts = response.json()["timestamp"]
        assert isinstance(ts, str) and len(ts) > 0

    def test_health_does_not_require_authorization(self, mock_runner: AsyncMock) -> None:
        """GET /health is reachable without an Authorization header (unauthenticated)."""
        app = create_app(agent_runner=mock_runner)
        mock_ctx, _client = _mock_httpx_success()

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            response = client.get("/health")  # No Authorization header

        assert response.status_code != 401

    def test_health_checks_dict_contains_qdrant_key(self, mock_runner: AsyncMock) -> None:
        """The 'checks' dict in the health response always includes a 'qdrant' key."""
        app = create_app(agent_runner=mock_runner)
        mock_ctx, _client = _mock_httpx_success()

        with (
            patch("research_agent.api.routes.health.httpx.AsyncClient", return_value=mock_ctx),
            TestClient(app) as client,
        ):
            body = client.get("/health").json()

        assert "qdrant" in body["checks"]


# ---------------------------------------------------------------------------
# POST /research
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResearchRoute:
    def test_returns_200_with_valid_request(self) -> None:
        """POST /research returns 200 and a structured report for a valid query."""
        runner = _make_runner()
        app = _make_app_with_auth_bypassed(agent_runner=runner)

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "What is quantum computing?"})

        assert response.status_code == 200
        body = response.json()
        assert "summary" in body
        assert "citations" in body
        assert "session_id" in body

    def test_report_fields_map_from_runner_response(self) -> None:
        """Response body faithfully maps from the ResearchReport returned by the runner."""
        report = ResearchReport(
            query="What is quantum computing?",
            summary="Quantum computing uses quantum mechanics.",
            citations=[
                Citation(
                    url="https://example.com",
                    content_snippet="quantum snippet",
                    relevance_score=0.95,
                )
            ],
            session_id="sess-abc",
        )
        runner = _make_runner(report=report)
        app = _make_app_with_auth_bypassed(agent_runner=runner)

        with TestClient(app) as client:
            response = client.post(
                "/research",
                json={"query": "What is quantum computing?", "session_id": "sess-abc"},
            )

        body = response.json()
        assert body["summary"] == "Quantum computing uses quantum mechanics."
        assert len(body["citations"]) == 1
        assert body["citations"][0]["url"] == "https://example.com"
        assert body["citations"][0]["relevance_score"] == pytest.approx(0.95)
        assert body["session_id"] == "sess-abc"

    def test_returns_401_without_authorization_header(self) -> None:
        """POST /research returns 401 when no Authorization header is provided."""
        # Real auth dependency — no overrides
        app = create_app(agent_runner=_make_runner())

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/research", json={"query": "test"})

        assert response.status_code == 401

    def test_returns_422_for_blank_query(self) -> None:
        """POST /research returns 422 when query is whitespace-only."""
        app = _make_app_with_auth_bypassed(agent_runner=_make_runner())

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "   "})

        assert response.status_code == 422

    def test_returns_422_for_missing_query_field(self) -> None:
        """POST /research returns 422 when the 'query' field is absent from the body."""
        app = _make_app_with_auth_bypassed(agent_runner=_make_runner())

        with TestClient(app) as client:
            response = client.post("/research", json={"max_iterations": 3})

        assert response.status_code == 422

    def test_returns_422_when_max_iterations_exceeds_limit(self) -> None:
        """POST /research returns 422 when max_iterations is above 20."""
        app = _make_app_with_auth_bypassed(agent_runner=_make_runner())

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "test", "max_iterations": 99})

        assert response.status_code == 422

    def test_session_id_auto_generated_when_not_provided(self) -> None:
        """Session ID is auto-generated (non-empty) when the client omits it."""
        runner = _make_runner()
        app = _make_app_with_auth_bypassed(agent_runner=runner)

        with TestClient(app) as client:
            response = client.post("/research", json={"query": "test query"})

        assert response.status_code == 200
        assert response.json()["session_id"]  # non-empty string

    def test_runner_receives_correctly_constructed_query(self) -> None:
        """AgentRunner.run is called with a ResearchQuery built from the request body."""
        runner = _make_runner()
        app = _make_app_with_auth_bypassed(agent_runner=runner)

        with TestClient(app) as client:
            client.post(
                "/research",
                json={
                    "query": "test query",
                    "session_id": "my-session",
                    "max_iterations": 7,
                },
            )

        runner.run.assert_called_once()
        call_arg: ResearchQuery = runner.run.call_args[0][0]
        assert call_arg.query == "test query"
        assert call_arg.session_id == "my-session"
        assert call_arg.max_iterations == 7

    def test_returns_503_when_no_agent_runner_configured(self) -> None:
        """POST /research returns 503 when no AgentRunner is wired up (None)."""
        app = _make_app_with_auth_bypassed(agent_runner=_make_runner())

        with TestClient(app, raise_server_exceptions=False) as client:
            app.state.agent_runner = None  # simulate missing runner at request time
            response = client.post("/research", json={"query": "test"})

        assert response.status_code == 503

    def test_citations_list_is_empty_when_report_has_no_citations(self) -> None:
        """Response citations list is [] when the runner returns no citations."""
        report = ResearchReport(
            query="bare query",
            summary="no sources found",
            citations=[],
            session_id="sess-empty",
        )
        runner = _make_runner(report=report)
        app = _make_app_with_auth_bypassed(agent_runner=runner)

        with TestClient(app) as client:
            response = client.post(
                "/research", json={"query": "bare query", "session_id": "sess-empty"}
            )

        body = response.json()
        assert body["citations"] == []
