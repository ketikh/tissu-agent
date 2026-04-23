"""Unit tests for the in-process rate limiter."""
from __future__ import annotations

import time

import pytest

from src.rate_limit import RateLimiter


def _limiter():
    # Tight window so tests finish quickly; 1s window, 1s lockout.
    return RateLimiter(max_attempts=3, window_seconds=1, lockout_seconds=1)


def test_allows_until_threshold():
    rl = _limiter()
    assert rl.allow("1.2.3.4") is True
    rl.record_failure("1.2.3.4")
    rl.record_failure("1.2.3.4")
    # 2 failures — still allowed for the 3rd try.
    assert rl.allow("1.2.3.4") is True


def test_locks_out_after_threshold():
    rl = _limiter()
    for _ in range(3):
        rl.record_failure("5.5.5.5")
    assert rl.allow("5.5.5.5") is False


def test_unlocks_after_lockout_seconds():
    rl = RateLimiter(max_attempts=2, window_seconds=1, lockout_seconds=1)
    rl.record_failure("9.9.9.9")
    rl.record_failure("9.9.9.9")
    assert rl.allow("9.9.9.9") is False
    time.sleep(1.05)
    assert rl.allow("9.9.9.9") is True


def test_success_clears_failures():
    rl = _limiter()
    rl.record_failure("2.2.2.2")
    rl.record_failure("2.2.2.2")
    rl.record_success("2.2.2.2")
    assert rl.allow("2.2.2.2") is True
    # No carry-over: we can now fail 3 more times before lockout.
    rl.record_failure("2.2.2.2")
    rl.record_failure("2.2.2.2")
    assert rl.allow("2.2.2.2") is True


def test_separate_keys_are_isolated():
    rl = _limiter()
    for _ in range(3):
        rl.record_failure("attacker")
    assert rl.allow("attacker") is False
    # A different IP is untouched.
    assert rl.allow("innocent") is True


def test_empty_key_always_allowed():
    """Defensive: if we ever fail to resolve the client IP, fail open
    — better than locking out everyone because of one empty string."""
    rl = _limiter()
    for _ in range(10):
        rl.record_failure("")
    assert rl.allow("") is True


def test_seconds_until_unlock():
    rl = RateLimiter(max_attempts=2, window_seconds=60, lockout_seconds=60)
    rl.record_failure("ip")
    rl.record_failure("ip")
    remaining = rl.seconds_until_unlock("ip")
    assert 55 <= remaining <= 60
