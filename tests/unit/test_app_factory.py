"""Unit tests for the FastAPI app factory (create_app)."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_agent.api.main import create_app
from research_agent.api.middleware import RequestIDMiddleware

# ---------------------------------------------------------------------------
# App factory basics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_title(self) -> None:
        app = create_app()
        assert "research" in app.title.lower()

    def test_request_id_middleware_is_registered(self) -> None:
        app = create_app()
        middleware_classes = [m.cls for m in app.user_middleware if hasattr(m, "cls")]
        assert RequestIDMiddleware in middleware_classes

    def test_health_route_exists(self) -> None:
        app = create_app()
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/health" in routes

    def test_research_route_exists(self) -> None:
        app = create_app()
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/research" in routes

    def test_multiple_calls_return_independent_instances(self) -> None:
        app1 = create_app()
        app2 = create_app()
        assert app1 is not app2


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAppLifespan:
    def test_app_starts_up_cleanly(self) -> None:
        """TestClient context manager triggers lifespan startup/shutdown."""
        app = create_app()
        with TestClient(app) as client:
            # If startup raises, TestClient.__enter__ will propagate it
            assert client is not None

    def test_app_stores_agent_runner_in_state(self) -> None:
        from unittest.mock import AsyncMock

        from research_agent.api.dependencies import AgentRunner
        from research_agent.models.research import ResearchReport

        mock_runner = AsyncMock(spec=AgentRunner)
        mock_runner.run.return_value = ResearchReport(
            query="q",
            summary="s",
            citations=[],
            session_id="sess",
        )

        app = create_app(agent_runner=mock_runner)
        with TestClient(app):
            assert app.state.agent_runner is mock_runner


# ---------------------------------------------------------------------------
# Lifespan — LangSmith activation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLangSmithActivation:
    def test_langsmith_env_vars_set_when_tracing_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LangSmith env vars are written at startup when tracing + key are configured."""
        fake_key = "lsv2_pt_" + "x" * 40  # noqa: S105 — test value, not a real secret
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", fake_key)
        # Required secrets so Settings validates without a .env file
        monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
        monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
        monkeypatch.setenv("SECRET_KEY", "a" * 64)

        app = create_app()
        with TestClient(app):
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == fake_key

    def test_langsmith_env_vars_not_set_when_tracing_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LANGCHAIN_TRACING_V2 is not written when tracing is off."""
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
        monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
        monkeypatch.setenv("SECRET_KEY", "a" * 64)

        app = create_app()
        before = os.environ.get("LANGCHAIN_TRACING_V2")
        with TestClient(app):
            # Should still be whatever it was before lifespan ran
            assert os.environ.get("LANGCHAIN_TRACING_V2") == before


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGlobalExceptionHandler:
    def test_unhandled_exception_returns_500_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Routes that raise unhandled exceptions produce a safe 500 JSON response."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
        monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
        monkeypatch.setenv("SECRET_KEY", "a" * 64)

        app = create_app()

        @app.get("/boom")
        async def _explode() -> None:
            raise RuntimeError("test explosion — should be swallowed")

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/boom")

        assert response.status_code == 500
        assert response.json()["detail"] == "An unexpected error occurred."

    def test_unhandled_exception_does_not_leak_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """500 response body does not contain internal exception details."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_" + "x" * 30)
        monkeypatch.setenv("MEM0_API_KEY", "m0_test_" + "x" * 30)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test_" + "x" * 30)
        monkeypatch.setenv("SECRET_KEY", "a" * 64)

        app = create_app()

        @app.get("/secret_error")
        async def _secret_error() -> None:
            raise ValueError("db_password=supersecret")  # noqa: S106

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/secret_error")

        body_str = str(response.json())
        assert "supersecret" not in body_str
        assert "db_password" not in body_str
