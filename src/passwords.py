"""Password hashing utilities — argon2id only.

We use argon2id (the OWASP recommended default at time of writing) via
argon2-cffi. Never bcrypt, never plain. All parameters live here so we
can rotate them without touching call sites.

Contract:
- ``hash_password(raw)`` returns an argon2id-encoded string safe to
  store in the DB ``password_hash`` column.
- ``verify_password(raw, stored_hash)`` returns True/False in
  constant-ish time — never leak timing info to callers.
- ``needs_rehash(stored_hash)`` tells the login flow whether to
  silently re-hash the password on successful login (e.g. when we bump
  the cost parameters globally).
"""
from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

# argon2id tuning — OWASP "interactive" profile. Memory is the main
# knob; 64 MiB per hash fits comfortably on Railway's smallest tiers.
_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(raw: str) -> str:
    """Return an argon2id-encoded hash of ``raw``.

    The returned string includes the salt, params, and hash so it's
    self-describing — no separate salt column needed.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError("password must be a non-empty string")
    return _HASHER.hash(raw)


def verify_password(raw: str, stored_hash: str) -> bool:
    """Return True iff ``raw`` matches ``stored_hash``.

    Wraps argon2's verify in a try/except so mismatched / invalid /
    corrupted hashes all fold into a plain False — the caller has no
    way to tell apart "wrong password" vs "stored hash was garbled".
    That's deliberate: we want one generic error surface at the login
    boundary.
    """
    if not raw or not stored_hash:
        return False
    try:
        return _HASHER.verify(stored_hash, raw)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        # Defensive: never leak an unexpected crash as a login success.
        return False


def needs_rehash(stored_hash: str) -> bool:
    """Has the global cost changed since this hash was stored?

    Call after a successful ``verify_password`` to decide whether to
    re-hash with the current parameters and write that back to the DB.
    """
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except Exception:
        return False
