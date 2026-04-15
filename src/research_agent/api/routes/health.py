"""Health check route — ``GET /health``.

This endpoint is intentionally unauthenticated so that Railway's startup
probe and any external monitoring can reach it without credentials.

Probes:
  * **qdrant** — ``GET {qdrant_url}/healthz`` with a 2-second timeout.

The overall status is ``"ok"`` only when all probes succeed; otherwise
``"error"`` and the response is sent with HTTP 503.
"""

from __future__ import annotations

import datetime

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from research_agent.api.schemas import HealthCheckResponse
from research_agent.config import get_settings

router = APIRouter()


async def _probe_qdrant(qdrant_url: str) -> str:
    """Return ``"ok"`` if Qdrant responds to its health endpoint; ``"error"`` otherwise."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{qdrant_url}/healthz")
            resp.raise_for_status()
            return "ok"
    except Exception:  # noqa: BLE001
        return "error"


@router.get("/health")
async def health_check() -> JSONResponse:
    """Return a health status payload with per-dependency check results.

    Probes each external dependency asynchronously. Returns HTTP 200 when
    all checks pass and HTTP 503 when any check fails.

    The ``checks`` dict always contains at least a ``"qdrant"`` key.
    """
    settings = get_settings()

    checks: dict[str, str] = {
        "qdrant": await _probe_qdrant(settings.qdrant_url),
    }

    overall: str = "ok" if all(v == "ok" for v in checks.values()) else "error"
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()

    payload = HealthCheckResponse(status=overall, checks=checks, timestamp=timestamp)
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(status_code=status_code, content=payload.model_dump())
