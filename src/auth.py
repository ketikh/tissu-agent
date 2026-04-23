"""API key authentication middleware.

Requires a valid ``X-API-Key`` header on every ``/api/*`` request, with a
small list of exemptions for paths that cannot carry a custom header:
``/api/health`` (the task contract), the Meta data-deletion callback (which
Meta signs with ``signed_request``), and the owner-confirm / photo-confirm
links that the shop owner opens from their phone after a WhatsApp
notification.

The middleware resolves an X-API-Key to a tenant_id via the ``api_keys``
table, checks the key's scope against the path, and stashes both on the
request for downstream handlers. Exempt paths and non-/api routes fall
back to ``DEFAULT_TENANT_ID``.

Key scopes:
  - 'admin'      — unrestricted, every /api/* path
  - 'storefront' — read-only, only /api/storefront/* paths
"""
from __future__ import annotations

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.db import DEFAULT_TENANT_ID, resolve_api_key

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
        # multi-tenant source of truth. Returns (tenant_id, scope) or None.
        tenant_id: str | None = None
        scope: str = "admin"
        try:
            hit = await resolve_api_key(provided)
        except Exception:
            hit = None

        if hit is not None:
            tenant_id, scope = hit
        else:
            # Bootstrap path: if the presented key matches one of the
            # env vars and the api_keys row hasn't been seeded yet,
            # accept it with the appropriate scope. init_db seeds these
            # rows on boot, so this only helps the very first request
            # after a fresh migration.
            admin_env = os.environ.get("ADMIN_API_KEY", "").strip()
            sf_env = os.environ.get("STOREFRONT_API_KEY", "").strip()
            if admin_env and provided == admin_env:
                tenant_id = DEFAULT_TENANT_ID
                scope = "admin"
            elif sf_env and provided == sf_env:
                tenant_id = DEFAULT_TENANT_ID
                scope = "storefront"
            else:
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                )

        # Scope enforcement: storefront-scoped keys can only touch
        # /api/storefront/*. Admin-scoped keys can touch anything.
        if scope == "storefront" and not path.startswith("/api/storefront/"):
            return JSONResponse(
                {"error": "forbidden", "reason": "key is read-only storefront scope"},
                status_code=403,
            )

        request.state.tenant_id = tenant_id
        request.state.api_key_scope = scope
        return await call_next(request)
