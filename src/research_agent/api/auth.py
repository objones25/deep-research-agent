"""JWT authentication dependency for the FastAPI layer.

``require_auth`` is a standalone async function used both directly in tests
and as the basis for a FastAPI ``Depends()`` wrapper in ``main.py``.

Design:
  * Accepts ``authorization``, ``secret``, and ``algorithm`` explicitly so
    that unit tests can call it without constructing a full FastAPI request.
  * The FastAPI-bound version (``get_auth``) is wired up in ``main.py``
    via ``Depends()`` and reads values from ``Settings``.
  * Raises ``HTTPException(401)`` for every invalid-token case — no detail
    differentiation that would help an attacker distinguish expired vs wrong
    signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import HTTPException, status


@dataclass(frozen=True)
class TokenPayload:
    """Decoded, validated JWT claims."""

    sub: str


async def require_auth(
    authorization: str | None,
    secret: str,
    algorithm: str,
) -> TokenPayload:
    """Validate a Bearer JWT and return its payload.

    Args:
        authorization: Value of the ``Authorization`` header (e.g.
            ``"Bearer eyJ..."``), or ``None`` if the header was absent.
        secret:     JWT signing secret (plaintext, obtained from ``Settings``).
        algorithm:  JWT algorithm (e.g. ``"HS256"``).

    Returns:
        ``TokenPayload`` with the ``sub`` claim.

    Raises:
        HTTPException(401): For missing, malformed, expired, or invalid tokens.
    """
    _unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization:
        raise _unauthorized

    parts = authorization.split(" ", maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise _unauthorized

    token = parts[1]
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError:
        raise _unauthorized from None

    sub = claims.get("sub")
    if not sub:
        raise _unauthorized

    return TokenPayload(sub=str(sub))
