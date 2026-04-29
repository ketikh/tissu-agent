"""Unit tests for the signed session cookie helpers.

These cover the round-trip: issue a session token, load it back, get
the original payload. Plus tamper/expiry/CSRF-binding behavior.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def fixed_secret(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "unit-test-secret-value")
    yield


def _reimport():
    """Session module caches `_secret` via function calls on each use,
    so it picks up the patched env var without reload — but we also
    need the constants to be visible after the env is set, so always
    grab a fresh import for clarity."""
    import importlib
    import src.sessions as s
    importlib.reload(s)
    return s


def test_session_token_roundtrips():
    s = _reimport()
    tok = s.issue_session_token(user_id=42, tenant_id="default")
    loaded = s.load_session_token(tok)
    assert loaded == {"user_id": 42, "tenant_id": "default", "epoch": 0}


def test_session_token_carries_epoch():
    s = _reimport()
    tok = s.issue_session_token(user_id=7, tenant_id="abc", epoch=3)
    loaded = s.load_session_token(tok)
    assert loaded["epoch"] == 3


def test_session_token_rejects_garbled_input():
    s = _reimport()
    assert s.load_session_token("") is None
    assert s.load_session_token("not.a.real.token") is None
    assert s.load_session_token("abc") is None


def test_session_token_expires():
    s = _reimport()
    tok = s.issue_session_token(1, "default")
    # Sleep past the tiny max_age so itsdangerous trips the expiry check.
    # Use 2 seconds to cleanly exceed itsdangerous' integer-second
    # comparison.
    time.sleep(2.1)
    assert s.load_session_token(tok, max_age_seconds=1) is None


def test_session_token_with_different_secret_fails(monkeypatch):
    s = _reimport()
    tok = s.issue_session_token(1, "default")
    monkeypatch.setenv("SESSION_SECRET", "a-completely-different-secret")
    s2 = _reimport()
    # Token signed by old secret cannot be loaded with new one.
    assert s2.load_session_token(tok) is None


def test_csrf_token_binds_to_session():
    s = _reimport()
    sess = s.issue_session_token(7, "default")
    csrf = s.issue_csrf_token(sess)
    assert s.verify_csrf_token(csrf, sess) is True


def test_csrf_token_mismatched_session_is_rejected():
    s = _reimport()
    sess_a = s.issue_session_token(1, "default")
    sess_b = s.issue_session_token(2, "default")
    csrf_for_a = s.issue_csrf_token(sess_a)
    # Replaying csrf_for_a with sess_b's session must fail.
    assert s.verify_csrf_token(csrf_for_a, sess_b) is False


def test_csrf_token_rejects_empty_inputs():
    s = _reimport()
    assert s.verify_csrf_token("", "sess") is False
    assert s.verify_csrf_token("csrf", "") is False


def test_session_refuses_to_boot_without_secret_in_prod(monkeypatch):
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.setenv("ENV", "production")
    s = _reimport()
    with pytest.raises(RuntimeError):
        s.issue_session_token(1, "default")


def test_cookie_kwargs_defaults_to_secure_and_httponly():
    s = _reimport()
    kw = s.cookie_kwargs(3600)
    assert kw["httponly"] is True
    assert kw["secure"] is True
    assert kw["samesite"] == "lax"
    assert kw["max_age"] == 3600
