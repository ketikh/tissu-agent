"""Signed session cookie helpers for the admin panel.

We issue a single cookie, ``admin_session``, carrying the minimum
payload the middleware needs to re-identify the user:
``{user_id, tenant_id}``. The cookie is HS256-signed with a
timestamp so it auto-expires without a server-side store — if we
ever want revocation-on-logout we'll add a Redis or DB backed
allowlist, but for a single-shop admin that's overkill.

The signing secret comes from ``SESSION_SECRET``. For safety in
development, we fall back to a stable per-project value so a
missing env var doesn't crash the boot — but that's marked and
refused in production via ``require_session_secret``.
"""
from __future__ import annotations

import os
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


SESSION_COOKIE_NAME = "admin_session"
CSRF_COOKIE_NAME = "admin_csrf"

# Cookie lifetimes — the login form exposes a "remember me" toggle
# that picks between the two.
SHORT_SESSION_SECONDS = 24 * 60 * 60        # 24h
LONG_SESSION_SECONDS = 30 * 24 * 60 * 60    # 30d

# Two different salts per use — prevents a bug where the session
# token is replayed as a CSRF token (or vice-versa).
_SESSION_SALT = "admin-session-v1"
_CSRF_SALT = "admin-csrf-v1"


def _secret() -> str:
    """Return the active signing secret.

    In production (``ENV=production`` or Railway auto-set
    ``RAILWAY_ENVIRONMENT``), refuse to run without ``SESSION_SECRET``.
    Local dev falls back to a constant stand-in so booting a fresh
    checkout doesn't require env setup.
    """
    secret = os.environ.get("SESSION_SECRET", "").strip()
    if secret:
        return secret
    is_prod = (
        os.environ.get("ENV") == "production"
        or bool(os.environ.get("RAILWAY_ENVIRONMENT"))
    )
    if is_prod:
        raise RuntimeError(
            "SESSION_SECRET must be set in production — refusing to boot "
            "without a real session signing key."
        )
    # Stable dev fallback — documented, never used in prod.
    return "tissu-dev-session-secret-do-not-use-in-production"


def _serializer(salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret(), salt=salt)


# ── Session tokens ────────────────────────────────────────────

def issue_session_token(user_id: int, tenant_id: str) -> str:
    """Return a signed cookie value carrying the user id + tenant id."""
    payload = {"user_id": int(user_id), "tenant_id": str(tenant_id)}
    return _serializer(_SESSION_SALT).dumps(payload)


def load_session_token(token: str, max_age_seconds: int = LONG_SESSION_SECONDS) -> dict | None:
    """Return the payload if ``token`` is valid and within max_age, else None."""
    if not token:
        return None
    try:
        data = _serializer(_SESSION_SALT).loads(token, max_age=max_age_seconds)
    except SignatureExpired:
        return None
    except BadSignature:
        return None
    except Exception:
        return None
    if not isinstance(data, dict) or "user_id" not in data:
        return None
    return data


# ── CSRF tokens ───────────────────────────────────────────────
#
# A CSRF token is just a signed random-ish string scoped to the
# session. The server issues it alongside the session cookie; the
# admin UI echoes it back in an ``X-CSRF-Token`` header on every
# unsafe request. Commit 9 enforces the check.

def issue_csrf_token(session_token: str) -> str:
    """Derive a CSRF token from the current session token.

    Because it's signed with a different salt, it can't be swapped
    with the session cookie. Because the payload is the session
    token, it automatically rotates when the user logs in / out.
    """
    return _serializer(_CSRF_SALT).dumps(session_token)


def verify_csrf_token(csrf_token: str, session_token: str, max_age_seconds: int = LONG_SESSION_SECONDS) -> bool:
    """Return True iff ``csrf_token`` is a valid signed pairing of
    ``session_token``. False on any error — never raise."""
    if not csrf_token or not session_token:
        return False
    try:
        bound = _serializer(_CSRF_SALT).loads(
            csrf_token, max_age=max_age_seconds,
        )
    except Exception:
        return False
    return bound == session_token


def cookie_kwargs(max_age_seconds: int, secure: bool = True) -> dict[str, Any]:
    """Common kwargs for setting our cookies on a FastAPI response."""
    return {
        "max_age": max_age_seconds,
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }
