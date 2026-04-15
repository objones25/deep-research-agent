"""Research route — ``POST /research``.

Requires JWT authentication (``Depends(get_auth)``) and is subject to
per-IP rate limiting (``Depends(check_rate_limit)``).

Request flow:
  1. Pydantic validates the request body (``ResearchQueryRequest``).
  2. ``get_auth`` verifies the Bearer JWT; 401 on failure.
  3. ``check_rate_limit`` enforces per-IP limits; 429 on excess.
  4. The ``AgentRunner`` stored in ``app.state.agent_runner`` is retrieved;
     503 if it is ``None`` (service not yet initialised).
  5. A ``ResearchQuery`` is built from the body (auto-generating ``session_id``
     when the client omits it).
  6. ``runner.run(query)`` executes the LangGraph agent.
  7. The resulting ``ResearchReport`` is mapped to ``ResearchReportResponse``
     and returned as HTTP 200.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from research_agent.api.auth import TokenPayload
from research_agent.api.dependencies import AgentRunner, check_rate_limit, get_auth
from research_agent.api.schemas import (
    CitationResponse,
    ResearchQueryRequest,
    ResearchReportResponse,
)
from research_agent.models.research import ResearchQuery

router = APIRouter()


@router.post("/research", response_model=ResearchReportResponse)
async def run_research(
    body: ResearchQueryRequest,
    request: Request,
    _auth: Annotated[TokenPayload, Depends(get_auth)],
    _rate: Annotated[None, Depends(check_rate_limit)],
) -> ResearchReportResponse | JSONResponse:
    """Accept a research query and return a structured report.

    Args:
        body: Validated request body with query, optional session_id, and
            max_iterations.
        request: Starlette request used to retrieve ``app.state.agent_runner``.
        _auth: JWT payload injected by ``get_auth``; signals authentication
            passed (value not used directly).
        _rate: Rate-limit check injected by ``check_rate_limit``; signals the
            client is within limits (always ``None``).

    Returns:
        ``ResearchReportResponse`` on success.

    Raises:
        ``JSONResponse(503)`` when no ``AgentRunner`` is configured.
        Any exception from the agent propagates to the global 500 handler.
    """
    runner: AgentRunner | None = request.app.state.agent_runner

    if runner is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Research agent is not available."},
        )

    session_id = body.session_id or str(uuid.uuid4())
    query = ResearchQuery(
        query=body.query,
        session_id=session_id,
        max_iterations=body.max_iterations,
    )

    report = await runner.run(query)

    return ResearchReportResponse(
        query=report.query,
        summary=report.summary,
        citations=[
            CitationResponse(
                url=c.url,
                content_snippet=c.content_snippet,
                relevance_score=c.relevance_score,
            )
            for c in report.citations
        ],
        session_id=report.session_id,
    )
