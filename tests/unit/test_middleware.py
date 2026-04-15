"""Unit tests for research_agent.api.middleware.RequestIDMiddleware."""

from __future__ import annotations

import re
import uuid

import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from httpx import ASGITransport, AsyncClient

from research_agent.api.middleware import REQUEST_ID_CTX_VAR, RequestIDMiddleware

# UUID v4 pattern
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _make_test_app() -> FastAPI:
    """Minimal app with RequestIDMiddleware for isolation testing."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    async def ping() -> PlainTextResponse:
        # Echo the request_id from contextvars so we can assert it was bound
        rid = REQUEST_ID_CTX_VAR.get("")
        return PlainTextResponse(rid)

    return app


@pytest.fixture
def test_app() -> FastAPI:
    return _make_test_app()


# ---------------------------------------------------------------------------
# RequestIDMiddleware behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequestIDMiddleware:
    async def test_response_contains_request_id_header(self, test_app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ping")
        assert "x-request-id" in resp.headers

    async def test_generated_request_id_is_valid_uuid(self, test_app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ping")
        rid = resp.headers["x-request-id"]
        assert _UUID_RE.match(rid), f"Not a valid UUID v4: {rid!r}"

    async def test_forwarded_request_id_is_preserved(self, test_app: FastAPI) -> None:
        incoming = str(uuid.uuid4())
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ping", headers={"x-request-id": incoming})
        assert resp.headers["x-request-id"] == incoming

    async def test_different_requests_get_different_ids(self, test_app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            r1 = await client.get("/ping")
            r2 = await client.get("/ping")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    async def test_request_id_bound_to_contextvars(self, test_app: FastAPI) -> None:
        """Route handler can read request_id from contextvars."""
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ping")
        # The route echoes REQUEST_ID_CTX_VAR — must match the response header
        rid_from_header = resp.headers["x-request-id"]
        rid_from_ctx = resp.text
        assert rid_from_ctx == rid_from_header

    async def test_provided_request_id_bound_to_contextvars(self, test_app: FastAPI) -> None:
        incoming = str(uuid.uuid4())
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as client:
            resp = await client.get("/ping", headers={"x-request-id": incoming})
        assert resp.text == incoming
