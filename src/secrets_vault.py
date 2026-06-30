"""Symmetric encryption for per-tenant secrets (Facebook page tokens,
WhatsApp tokens, etc.).

We use Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography
package. The key lives in the ``FB_TOKEN_ENCRYPTION_KEY`` env var —
generate with ``python -c "from cryptography.fernet import Fernet;
print(Fernet.generate_key().decode())"`` and set on Railway before
the first tenant is onboarded with a Facebook token.

Why this layer exists: a Supabase DB dump (or a read-only credential
leaking) should NOT immediately hand out working Facebook / WhatsApp
page tokens. Storing them encrypted at rest means the attacker also
needs FB_TOKEN_ENCRYPTION_KEY (which only Railway has) to decrypt.

Rotating the key: replace the env var, then on first boot the
platform re-encrypts every stored token. That's not implemented here
yet — we'll add it when we actually need to rotate.
"""
from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    key = os.environ.get("FB_TOKEN_ENCRYPTION_KEY", "").strip()
    if not key:
        # Refuse to silently encrypt-with-placeholder-key — that would
        # mean a production DB full of unrecoverable garbage if the
        # dev fallback ever leaked. A missing key is a config error.
        raise RuntimeError(
            "FB_TOKEN_ENCRYPTION_KEY env var is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"\n"
            "and set it on Railway before storing tenant Facebook tokens."
        )
    return Fernet(key.encode("utf-8"))


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a string and return a base64-safe ciphertext you can
    stick straight into a TEXT column. Returns '' for empty input so
    callers can round-trip 'no token yet' without special cases."""
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str) -> str:
    """Inverse of ``encrypt_secret``. Returns '' for empty input or
    any decryption failure — callers should treat the empty string as
    "no token" and fall back cleanly. We deliberately swallow
    InvalidToken rather than raising so a corrupted row doesn't take
    the webhook dispatcher down on startup."""
    if not ciphertext:
        return ""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return ""
    except Exception:
        return ""


def redacted(value: str, keep: int = 4) -> str:
    """Return a UI-safe preview of a secret (first ``keep`` chars +
    '…'). Used in the super-admin form so the operator can sanity-
    check that the token they pasted looks right without us echoing
    the whole thing back."""
    if not value:
        return ""
    if len(value) <= keep:
        return "•" * len(value)
    return value[:keep] + "…"
