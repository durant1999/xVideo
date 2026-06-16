"""Bearer-token auth dependency.

Fails closed: if no token is configured on the server, every authed route is
rejected (we never run open). /healthz stays unauthenticated for tunnel testing.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .settings import get_settings


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()

    if not settings.has_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server has no XVIDEO_API_TOKEN configured; refusing to run unauthenticated.",
        )

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header (expected 'Bearer <token>').",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to avoid leaking the token via timing.
    if not hmac.compare_digest(token, settings.api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
