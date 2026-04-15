"""ASGI middleware for the deep-research-agent FastAPI application.

``RequestIDMiddleware`` — global middleware that:
  * Extracts the ``X-Request-ID`` header from incoming requests, or generates
    a UUID v4 when none is supplied.
  * Binds the request ID to a ``contextvars.ContextVar`` so that every
    structlog log call within the request lifecycle automatically includes it.
  * Echoes the request ID back in the ``X-Request-ID`` response header.
  * Clears the contextvar in a ``finally`` block to prevent leakage between
    requests in the same thread/coroutine.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Exported so that log processors and route handlers can read the bound value.
REQUEST_ID_CTX_VAR: ContextVar[str] = ContextVar("request_id", default="")

_HEADER = "x-request-id"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response cycle.

    The ID is stored in ``REQUEST_ID_CTX_VAR`` for the duration of the
    request and cleared afterwards. structlog's ``merge_contextvars``
    processor picks it up automatically when the logging pipeline is
    configured via ``research_agent.logging.configure_logging()``.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(_HEADER) or str(uuid.uuid4())
        token = REQUEST_ID_CTX_VAR.set(request_id)
        try:
            response = await call_next(request)
        finally:
            REQUEST_ID_CTX_VAR.reset(token)
        response.headers[_HEADER] = request_id
        return response
