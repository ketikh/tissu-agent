"""API key authentication middleware.

Requires a valid ``X-API-Key`` header on every ``/api/*`` request, with a
small list of exemptions for paths that cannot carry a custom header:
``/api/health`` (the task contract), the Meta data-deletion callback (which
Meta signs with ``signed_request``), and the owner-confirm / photo-confirm
links that the shop owner opens from their phone after a WhatsApp
notification.

The middleware resolves an X-API-Key to a tenant_id via the ``api_keys``
table and stashes it on ``request.state.tenant_id`` so route handlers can
scope their queries. Exempt paths and non-/api routes fall back to
``DEFAULT_TENANT_ID`` — the single-shop Tissu deployment lives there.
"""
from __future__ import annotations

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.db import DEFAULT_TENANT_ID, resolve_tenant_id

# Paths under /api/* that bypass the API key check.
#
# Rules:
#   - exact match first (EXEMPT_EXACT)
#   - then prefix match (EXEMPT_PREFIXES), used for /api/owner-confirm/<token>
#     etc. where the token is part of the URL and varies per request.
EXEMPT_EXACT: frozenset[str] = frozenset({
    "/api/health",
    "/api/storefront/health",
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

    Also stashes the resolved tenant_id on ``request.state.tenant_id`` for
    downstream handlers. Non-/api/* routes (the HTML pages, Meta webhooks
    at /webhook and /wa-webhook, static assets) default to the single-shop
    ``DEFAULT_TENANT_ID`` — useful for the Facebook Messenger webhook,
    which is always the Tissu page in this deployment.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Non-/api routes and exempt /api paths are always the default tenant.
        if not path.startswith("/api/") or _is_exempt(path):
            request.state.tenant_id = DEFAULT_TENANT_ID
            return await call_next(request)

        provided = request.headers.get("x-api-key", "").strip()
        if not provided:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # Primary path: resolve via the api_keys table — that's the
        # multi-tenant source of truth.
        try:
            tenant_id = await resolve_tenant_id(provided)
        except Exception:
            # DB hiccup — fall back to env comparison rather than locking
            # the owner out of their own admin panel. Still single-tenant.
            tenant_id = None

        if tenant_id is None:
            # Bootstrap path: if the presented key matches the env var and
            # the api_keys row hasn't been seeded yet, accept it as the
            # default tenant. init_db seeds this row on boot, so this
            # only helps the very first request after a fresh migration.
            expected_env = os.environ.get("ADMIN_API_KEY", "").strip()
            if expected_env and provided == expected_env:
                tenant_id = DEFAULT_TENANT_ID
            else:
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                )

        request.state.tenant_id = tenant_id
        return await call_next(request)
