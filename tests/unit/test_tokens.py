"""Unit tests for password-reset token helpers."""
from __future__ import annotations

from src.tokens import (
    generate_reset_token,
    hash_reset_token,
    constant_time_equals,
)


def test_generate_reset_token_returns_raw_and_hash():
    raw, h = generate_reset_token()
    assert raw
    assert h
    assert raw != h
    # hex sha256 is 64 chars
    assert len(h) == 64
    # url-safe token doesn't contain padding characters that would
    # need URL-encoding
    assert "=" not in raw


def test_generated_hash_matches_hash_reset_token():
    raw, h = generate_reset_token()
    assert hash_reset_token(raw) == h


def test_generate_reset_token_is_unique_per_call():
    # 1000 generations, no collisions — exercises the RNG.
    seen = set()
    for _ in range(1000):
        raw, _ = generate_reset_token()
        assert raw not in seen
        seen.add(raw)


def test_hash_reset_token_is_deterministic():
    a = hash_reset_token("same-input")
    b = hash_reset_token("same-input")
    assert a == b


def test_hash_reset_token_changes_with_input():
    a = hash_reset_token("one")
    b = hash_reset_token("two")
    assert a != b


def test_constant_time_equals_matches_hmac_compare_digest():
    assert constant_time_equals("abc", "abc") is True
    assert constant_time_equals("abc", "abd") is False
    assert constant_time_equals("", "") is True
    assert constant_time_equals("abc", "") is False
