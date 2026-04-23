"""API key authentication middleware.

Requires a valid ``X-API-Key`` header on every ``/api/*`` request, with a
small list of exemptions for paths that cannot carry a custom header:
``/api/health`` (the task contract), the Meta data-deletion callback (which
Meta signs with ``signed_request``), and the owner-confirm / photo-confirm
links that the shop owner opens from their phone after a WhatsApp
notification.

The valid key(s) are sourced from the ``ADMIN_API_KEY`` env var. When
multi-tenant support lands in commit 2 this module will also resolve a key
to a tenant id via the ``api_keys`` table.
"""
from __future__ import annotations

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Paths under /api/* that bypass the API key check.
#
# Rules:
#   - exact match first (EXEMPT_EXACT)
#   - then prefix match (EXEMPT_PREFIXES), used for /api/owner-confirm/<token>
#     etc. where the token is part of the URL and varies per request.
EXEMPT_EXACT: frozenset[str] = frozenset({
    "/api/health",
    "/api/meta/data-deletion",
})

EXEMPT_PREFIXES: tuple[str, ...] = (
    "/api/owner-confirm/",
    "/api/owner-deny/",
    "/api/photo-confirm/",
    "/api/photo-deny/",
)


def _is_exempt(path: str) -> bool:
    if path in EXEMPT_EXACT:
        return True
    return any(path.startswith(p) for p in EXEMPT_PREFIXES)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject /api/* requests that do not present a valid X-API-Key.

    Non-/api/* routes (the HTML pages, Meta webhooks at /webhook and
    /wa-webhook, static assets) pass through untouched.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/") or _is_exempt(path):
            return await call_next(request)

        expected = os.environ.get("ADMIN_API_KEY", "").strip()
        if not expected:
            # Fail closed — never auto-allow because the operator forgot
            # to set the env var in production.
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
            )

        provided = request.headers.get("x-api-key", "").strip()
        if provided != expected:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
            )

        return await call_next(request)
