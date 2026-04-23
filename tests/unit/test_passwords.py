"""Unit tests for the password hashing helpers.

These cover the contract at `src/passwords.py` — hash/verify round-trip,
constant-time-ish rejection of wrong passwords, handling of malformed
stored hashes, and needs_rehash behaviour. We never inspect the hash
format directly (that's argon2-cffi's concern) — just the observable
truth-value of our thin wrapper.
"""
from __future__ import annotations

import pytest

from src.passwords import hash_password, needs_rehash, verify_password


def test_hash_then_verify_roundtrips():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h) is True


def test_verify_rejects_wrong_password():
    h = hash_password("s3cret-P@ssword")
    assert verify_password("not the password", h) is False


def test_verify_rejects_empty_inputs():
    assert verify_password("", "") is False
    assert verify_password("x", "") is False
    assert verify_password("", "$argon2id$irrelevant$") is False


def test_verify_rejects_malformed_hash_instead_of_crashing():
    # A caller could pass a legacy bcrypt hash or a typo'd value —
    # we must return False, not raise.
    assert verify_password("anything", "not-an-argon2-hash") is False
    assert verify_password("anything", "$2b$12$BcryptLooking.But.NotOurFormat") is False


def test_hash_same_password_twice_yields_different_hashes():
    """Because of the random salt, two hashes of the same password
    should almost never collide — and both must verify."""
    a = hash_password("hunter2")
    b = hash_password("hunter2")
    assert a != b
    assert verify_password("hunter2", a) is True
    assert verify_password("hunter2", b) is True


def test_hash_password_rejects_empty_or_nonstring():
    with pytest.raises(ValueError):
        hash_password("")
    with pytest.raises(ValueError):
        hash_password(None)  # type: ignore[arg-type]


def test_needs_rehash_is_false_for_a_fresh_hash():
    """A hash just produced with current parameters shouldn't need a
    rehash. (If we ever bump the cost, this stays honest — the helper
    just reports False while params match.)"""
    h = hash_password("whatever")
    assert needs_rehash(h) is False


def test_needs_rehash_is_false_for_garbled_input():
    """Defensive: never return True for garbled input — we don't want
    a nonsense hash to trigger a rehash attempt in the login flow."""
    assert needs_rehash("not-a-hash") is False
