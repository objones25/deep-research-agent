"""Unit tests for JWT auth and rate-limiting FastAPI dependencies."""

from __future__ import annotations

import asyncio
import time

import jwt
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from research_agent.api.auth import TokenPayload, require_auth
from research_agent.api.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Helpers — shared test JWT secret and token factory
# ---------------------------------------------------------------------------

_SECRET = "a" * 64  # 64-byte secret matching min-length requirement
_ALGORITHM = "HS256"


def _make_token(
    sub: str = "user-123",
    *,
    expired: bool = False,
    secret: str = _SECRET,
    algorithm: str = _ALGORITHM,
) -> str:
    now = time.time()
    exp = now - 60 if expired else now + 3600
    payload = {"sub": sub, "iat": int(now), "exp": int(exp)}
    return jwt.encode(payload, secret, algorithm=algorithm)


def _auth_app() -> FastAPI:
    """Minimal app that uses require_auth on its single route."""
    from unittest.mock import MagicMock

    from research_agent.config import Settings

    app = FastAPI()

    # Inject a Settings mock so require_auth doesn't need real env vars
    mock_settings = MagicMock(spec=Settings)
    mock_settings.secret_key.get_secret_value.return_value = _SECRET
    mock_settings.jwt_algorithm = _ALGORITHM

    @app.get("/protected")
    async def protected(
        payload: TokenPayload = pytest.importorskip(  # noqa: B008
            "research_agent.api.auth", reason="auth module missing"
        )
        and None,
    ) -> PlainTextResponse:  # type: ignore[assignment]
        return PlainTextResponse(payload.sub)

    return app


# ---------------------------------------------------------------------------
# require_auth — direct unit tests (call the dependency directly)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequireAuth:
    async def test_valid_token_returns_payload(self) -> None:
        token = _make_token(sub="alice")
        payload = await require_auth(
            authorization=f"Bearer {token}",
            secret=_SECRET,
            algorithm=_ALGORITHM,
        )
        assert payload.sub == "alice"

    async def test_missing_authorization_raises_401(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await require_auth(
                authorization=None,
                secret=_SECRET,
                algorithm=_ALGORITHM,
            )
        assert exc_info.value.status_code == 401

    async def test_malformed_scheme_raises_401(self) -> None:
        from fastapi import HTTPException

        token = _make_token()
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(
                authorization=f"Basic {token}",
                secret=_SECRET,
                algorithm=_ALGORITHM,
            )
        assert exc_info.value.status_code == 401

    async def test_expired_token_raises_401(self) -> None:
        from fastapi import HTTPException

        token = _make_token(expired=True)
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(
                authorization=f"Bearer {token}",
                secret=_SECRET,
                algorithm=_ALGORITHM,
            )
        assert exc_info.value.status_code == 401

    async def test_wrong_secret_raises_401(self) -> None:
        from fastapi import HTTPException

        token = _make_token(secret="b" * 64)
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(
                authorization=f"Bearer {token}",
                secret=_SECRET,
                algorithm=_ALGORITHM,
            )
        assert exc_info.value.status_code == 401

    async def test_garbage_token_raises_401(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await require_auth(
                authorization="Bearer not.a.jwt",
                secret=_SECRET,
                algorithm=_ALGORITHM,
            )
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRateLimiter:
    async def test_requests_within_limit_pass(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        # 3 requests from the same IP should all pass
        for _ in range(3):
            await limiter.check("192.168.1.1")

    async def test_request_exceeding_limit_raises_429(self) -> None:
        from fastapi import HTTPException

        limiter = RateLimiter(max_requests=2, window_seconds=60)
        await limiter.check("192.168.1.1")
        await limiter.check("192.168.1.1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("192.168.1.1")
        assert exc_info.value.status_code == 429

    async def test_different_ips_have_independent_limits(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        await limiter.check("10.0.0.1")  # first IP hits limit after this
        # Second IP should still be allowed
        await limiter.check("10.0.0.2")

    async def test_window_expiry_resets_counter(self) -> None:
        """Requests outside the sliding window are not counted."""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        await limiter.check("127.0.0.1")
        # Sleep past the window
        await asyncio.sleep(1.1)
        # Should succeed because the old request has expired from the window
        await limiter.check("127.0.0.1")

    async def test_rate_limit_error_message_is_informative(self) -> None:
        from fastapi import HTTPException

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        await limiter.check("192.168.1.1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("192.168.1.1")
        assert exc_info.value.detail  # must not be empty

    async def test_as_dependency_returns_callable_class(self) -> None:
        """as_dependency() produces a callable that behaves like check_rate_limit."""
        from unittest.mock import MagicMock

        from starlette.datastructures import Address

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        dep_class = limiter.as_dependency()
        dep_instance = dep_class()

        mock_request = MagicMock()
        mock_request.client = Address(host="10.0.0.99", port=12345)

        # Should not raise for the first request
        await dep_instance(mock_request)
