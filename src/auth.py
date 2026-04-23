"""API key + admin session authentication middleware.

Every ``/api/*`` request is authenticated by EITHER a valid
``X-API-Key`` header OR a valid ``admin_session`` cookie — whichever
arrives first. The cookie path is what admin.html uses once the
shop owner has logged in; the header path stays for external API
clients (the storefront site, curl scripts, etc.).

A small list of exemptions covers paths that cannot carry either:
``/api/health`` (the task contract), ``/api/storefront/health``
(public liveness probe), the Meta data-deletion callback (signed by
Meta itself), and the owner-confirm / photo-confirm links that the
shop owner opens from their phone after a WhatsApp notification.

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
from src.sessions import (
    SESSION_COOKIE_NAME, SHORT_SESSION_SECONDS, LONG_SESSION_SECONDS,
    load_session_token,
)

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

        tenant_id: str | None = None
        scope: str = "admin"
        auth_source: str = ""

        # Path 1 — X-API-Key header. External API clients use this.
        provided = request.headers.get("x-api-key", "").strip()
        if provided:
            try:
                hit = await resolve_api_key(provided)
            except Exception:
                hit = None
            if hit is not None:
                tenant_id, scope = hit
                auth_source = "api_key"
            else:
                # Bootstrap path: env-var match pre-seed.
                admin_env = os.environ.get("ADMIN_API_KEY", "").strip()
                sf_env = os.environ.get("STOREFRONT_API_KEY", "").strip()
                if admin_env and provided == admin_env:
                    tenant_id = DEFAULT_TENANT_ID
                    scope = "admin"
                    auth_source = "api_key_env"
                elif sf_env and provided == sf_env:
                    tenant_id = DEFAULT_TENANT_ID
                    scope = "storefront"
                    auth_source = "api_key_env"

        # Path 2 — admin session cookie. The admin HTML pages use this
        # (browsers send it automatically on same-origin fetches). We
        # accept it as admin-scoped for the current tenant.
        if tenant_id is None:
            cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
            if cookie:
                session = load_session_token(cookie, max_age_seconds=LONG_SESSION_SECONDS)
                if session and "user_id" in session:
                    tenant_id = session.get("tenant_id") or DEFAULT_TENANT_ID
                    scope = "admin"
                    auth_source = "session"
                    request.state.admin_user_id = session["user_id"]

        if tenant_id is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # Scope enforcement: storefront-scoped keys can only touch
        # /api/storefront/*. Admin-scoped keys can touch anything.
        if scope == "storefront" and not path.startswith("/api/storefront/"):
            return JSONResponse(
                {"error": "forbidden", "reason": "key is read-only storefront scope"},
                status_code=403,
            )

        request.state.tenant_id = tenant_id
        request.state.api_key_scope = scope
        request.state.auth_source = auth_source
        return await call_next(request)


# Paths under /admin/* that do NOT require a logged-in session (the
# login flow itself, plus password-reset, plus logout). Everything
# else under /admin/* redirects to /admin/login when unauthenticated.
ADMIN_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/admin/login",
    "/admin/logout",
    "/admin/forgot-password",
    "/admin/reset-password",
})


class AdminSessionMiddleware(BaseHTTPMiddleware):
    """Gate HTML pages under /admin/* behind the session cookie.

    If the visitor hits /admin, /admin/chat-test, etc. without a
    valid session, redirect them to /admin/login with a flag so the
    form can show the right message ("your session expired"). If they
    have a valid session, attach user_id + tenant_id to request.state
    so downstream handlers can show the user's name etc.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/admin"):
            return await call_next(request)
        # Exact-match public admin paths — login form, logout, the
        # forgot-password routes which don't exist yet but will in
        # commit 8. Matching as a startswith so /admin/reset-password
        # with any query string works.
        if (path in ADMIN_PUBLIC_PATHS
                or path.startswith("/admin/reset-password")
                or path.startswith("/admin/forgot-password")):
            return await call_next(request)

        cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
        if cookie:
            session = load_session_token(cookie, max_age_seconds=LONG_SESSION_SECONDS)
            if session and "user_id" in session:
                request.state.admin_user_id = session["user_id"]
                request.state.tenant_id = session.get("tenant_id") or DEFAULT_TENANT_ID
                return await call_next(request)

        # No / expired / tampered cookie → send them to the login form.
        return JSONResponse(
            status_code=302,
            content=None,
            headers={"location": "/admin/login?error=expired"},
        ) if False else _redirect_to_login()


def _redirect_to_login():
    """Small helper to avoid pulling RedirectResponse into auth.py
    (keeps the middleware independent of FastAPI's response helpers
    at module-import time)."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/admin/login?error=expired", status_code=302)
