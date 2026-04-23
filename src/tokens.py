"""Random token helpers for password resets, confirm links, etc.

Tokens are generated with ``secrets`` (CSPRNG) so they're unguessable.
We return both the raw token (handed to the user, never stored) and
its SHA-256 hash (stored in the DB so a dump can't resurrect active
tokens). Verification re-hashes the presented raw token and compares.

Not argon2 — these tokens are high-entropy random strings, not
low-entropy human input, so a fast hash is appropriate. The attack
surface is "DB dump leaks active tokens", which SHA-256 defeats.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets


RESET_TOKEN_BYTES = 32  # 256 bits — plenty of entropy


def generate_reset_token() -> tuple[str, str]:
    """Return ``(raw_token, token_hash)``.

    raw_token is URL-safe (no padding issues in query strings).
    token_hash is the hex SHA-256 to store in admin_password_resets.
    """
    raw = secrets.token_urlsafe(RESET_TOKEN_BYTES)
    return raw, _sha256_hex(raw)


def hash_reset_token(raw: str) -> str:
    """Hash ``raw`` the same way ``generate_reset_token`` does — used
    on the verify side."""
    return _sha256_hex(raw)


def constant_time_equals(a: str, b: str) -> bool:
    """Timing-safe string compare for hex digests. ``secrets`` already
    does this, but wrap it for readability at call sites."""
    return hmac.compare_digest(a, b)


def _sha256_hex(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
