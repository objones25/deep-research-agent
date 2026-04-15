"""FastAPI application factory.

``create_app()`` is the sole entry point for constructing the FastAPI
application. It is called by:
  * The ``uvicorn`` / ``fastapi dev`` process (via ``main:app`` in
    ``pyproject.toml`` scripts or the Railway ``CMD``).
  * Test fixtures — each test that needs a running app calls ``create_app()``
    with a mock ``AgentRunner`` to keep tests isolated.

Wiring order:
  1. ``lifespan`` context manager — logging init, LangSmith activation,
     agent runner stored in ``app.state``.
  2. ``RequestIDMiddleware`` — must be added after the app is created.
  3. Route inclusion — ``/health`` (unauthenticated), ``/research`` (JWT + rate-limit).
  4. Exception handler — translate unhandled exceptions to 500 JSON responses.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from research_agent.api.dependencies import AgentRunner
from research_agent.api.middleware import RequestIDMiddleware
from research_agent.api.routes import health as health_module
from research_agent.api.routes import research as research_module
from research_agent.config import get_settings
from research_agent.logging import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: run startup logic, yield, then run shutdown logic."""
    settings = get_settings()

    # Configure structured logging
    configure_logging(log_level=settings.log_level, log_json=settings.log_json)

    # Activate LangSmith tracing when both env vars are configured
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()

    yield

    # Shutdown — nothing to teardown at this stage


def create_app(agent_runner: AgentRunner | None = None) -> FastAPI:
    """Construct and configure a FastAPI application.

    Args:
        agent_runner: An ``AgentRunner`` implementation injected at construction
            time (production passes the compiled LangGraph; tests pass a mock).
            Stored as ``app.state.agent_runner`` so routes can retrieve it via
            ``request.app.state.agent_runner``.

    Returns:
        A fully configured ``FastAPI`` instance ready to be served by uvicorn.
    """
    settings = get_settings()

    app = FastAPI(
        title="Deep Research Agent",
        description=(
            "A modular AI research agent that accepts a query and returns "
            "a structured, cited report using hybrid retrieval and LangGraph."
        ),
        version=settings.app_version,
        lifespan=_lifespan,
        # Disable docs in production
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None if settings.environment == "prod" else "/redoc",
    )

    # Store the agent runner so route handlers can retrieve it from app.state
    app.state.agent_runner = agent_runner

    # Middleware (applied outermost-first in the call stack)
    app.add_middleware(RequestIDMiddleware)

    # Routes
    app.include_router(health_module.router)
    app.include_router(research_module.router)

    # Global exception handler — prevent stack traces from leaking to clients
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred."},
        )

    return app


# Module-level app instance used by ``uvicorn`` / ``fastapi dev``
app = create_app()
