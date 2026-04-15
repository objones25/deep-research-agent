"""FastAPI dependency providers and shared protocols for the API layer.

``AgentRunner`` is the sole SOLID abstraction between routes and the
LangGraph-compiled agent. Routes depend on this protocol via ``Depends()``,
never on the concrete ``CompiledGraph`` directly.

``get_auth`` and ``check_rate_limit`` are FastAPI ``Depends()``-compatible
functions that wrap the lower-level ``require_auth`` and ``RateLimiter.check``
with settings injection.
"""

from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

from fastapi import Depends, Header, Request

from research_agent.api.auth import TokenPayload, require_auth
from research_agent.api.rate_limiter import RateLimiter
from research_agent.config import Settings, get_settings
from research_agent.models.research import ResearchQuery, ResearchReport


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol satisfied by any object that can execute a research query.

    Concrete implementation (``CompiledGraph`` wrapper) is wired up in
    ``main.py``'s ``create_app()`` factory and injected via ``Depends()``.
    """

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Execute the research agent and return a structured report.

        Args:
            query: Validated research query including session context.

        Returns:
            A ``ResearchReport`` with summary and citations.

        Raises:
            Any exception from the agent graph propagates upward — callers
            are responsible for translating to HTTP errors.
        """
        ...


# ---------------------------------------------------------------------------
# FastAPI-bound auth dependency
# ---------------------------------------------------------------------------


async def get_auth(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> TokenPayload:
    """FastAPI dependency: validate the Bearer JWT and return its payload.

    Wraps ``require_auth`` with settings injection so routes can use
    ``Depends(get_auth)`` without knowing the secret or algorithm.
    """
    return await require_auth(
        authorization=authorization,
        secret=settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


# ---------------------------------------------------------------------------
# FastAPI-bound rate-limit dependency
# ---------------------------------------------------------------------------

# Module-level singleton — shared across all requests.
# Override in tests via app.dependency_overrides[check_rate_limit].
_rate_limiter: RateLimiter = RateLimiter(max_requests=60, window_seconds=60)


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency: enforce per-IP rate limiting on guarded routes."""
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limiter.check(client_ip)
