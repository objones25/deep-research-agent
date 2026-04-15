"""In-memory sliding-window rate limiter for the FastAPI layer.

``RateLimiter`` tracks request timestamps per client IP in a ``collections.deque``,
evicting entries older than ``window_seconds`` on each check. All mutations
are guarded by a per-IP ``asyncio.Lock`` to be safe under concurrent async
requests.

Usage (in route handler via Depends)::

    limiter = RateLimiter(max_requests=60, window_seconds=60)

    @router.post("/research")
    async def research(
        _: None = Depends(limiter.as_dependency()),
        ...
    ) -> ...:
        ...
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status


@dataclass
class RateLimiter:
    """Sliding-window rate limiter keyed by client IP address.

    Args:
        max_requests:    Maximum requests allowed within ``window_seconds``.
        window_seconds:  Length of the sliding window in seconds.
    """

    max_requests: int
    window_seconds: float
    _timestamps: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(deque), init=False, repr=False
    )
    _locks: dict[str, asyncio.Lock] = field(
        default_factory=lambda: defaultdict(asyncio.Lock), init=False, repr=False
    )

    async def check(self, client_ip: str) -> None:
        """Check whether ``client_ip`` is within the rate limit.

        Evicts timestamps outside the current window, then raises
        ``HTTPException(429)`` if the count would exceed ``max_requests``.

        Args:
            client_ip: Client IP address string used as the rate-limit key.

        Raises:
            HTTPException(429): If the rate limit is exceeded.
        """
        async with self._locks[client_ip]:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            bucket = self._timestamps[client_ip]

            # Evict expired entries from the left
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit exceeded: maximum {self.max_requests} requests "
                        f"per {self.window_seconds:.0f}s window."
                    ),
                )

            bucket.append(now)

    def as_dependency(self) -> type[_RateLimitDep]:  # type: ignore[name-defined]  # noqa: F821
        """Return a FastAPI dependency class bound to this limiter instance.

        Usage::

            @router.post("/research")
            async def research(_: None = Depends(limiter.as_dependency())):
                ...
        """
        limiter = self

        class _RateLimitDep:
            async def __call__(self, request: Request) -> None:
                client_ip = request.client.host if request.client else "unknown"
                await limiter.check(client_ip)

        return _RateLimitDep
