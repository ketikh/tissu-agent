"""In-process rate limiter for login + password-reset endpoints.

Keyed by client IP. Each bucket stores the timestamps of the recent
failed attempts; attempts older than the window get pruned lazily on
every call. If the count of failures within the window exceeds the
threshold, the IP is considered locked out for ``lockout_seconds``.

This deliberately does NOT persist across restarts — Railway rebuilds
reset the state, which is the right trade-off for a single-instance
admin. When we scale horizontally or want stricter lockout, swap the
backing dict for Redis. The public API won't change.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    failures: list[float] = field(default_factory=list)
    locked_until: float = 0.0


class RateLimiter:
    """Simple sliding-window + lockout limiter.

    Usage:
        limiter = RateLimiter(max_attempts=5, window_seconds=900,
                              lockout_seconds=900)
        if not limiter.allow(ip):
            return 429
        ... try operation ...
        if failure:
            limiter.record_failure(ip)
        else:
            limiter.record_success(ip)
    """

    def __init__(self, max_attempts: int, window_seconds: int, lockout_seconds: int):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _prune(self, bucket: _Bucket, now: float) -> None:
        cutoff = now - self.window_seconds
        bucket.failures = [t for t in bucket.failures if t >= cutoff]

    def allow(self, key: str) -> bool:
        """Return True if ``key`` is currently allowed to attempt."""
        if not key:
            return True
        now = time.time()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return True
            if bucket.locked_until and now < bucket.locked_until:
                return False
            self._prune(bucket, now)
            return len(bucket.failures) < self.max_attempts

    def record_failure(self, key: str) -> None:
        """Record a failed attempt. Engages lockout when threshold hits."""
        if not key:
            return
        now = time.time()
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket())
            self._prune(bucket, now)
            bucket.failures.append(now)
            if len(bucket.failures) >= self.max_attempts:
                bucket.locked_until = now + self.lockout_seconds

    def record_success(self, key: str) -> None:
        """Clear the bucket — a successful login forgives prior failures."""
        if not key:
            return
        with self._lock:
            self._buckets.pop(key, None)

    def seconds_until_unlock(self, key: str) -> int:
        """How long until ``key`` can try again. 0 if not locked."""
        if not key:
            return 0
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket or not bucket.locked_until:
                return 0
            remaining = bucket.locked_until - time.time()
            return max(0, int(remaining))


# Default shared limiters used by the admin login + password reset
# endpoints. Tuning lives here so we can adjust in one place.
login_limiter = RateLimiter(
    max_attempts=5, window_seconds=15 * 60, lockout_seconds=15 * 60,
)

password_reset_limiter = RateLimiter(
    max_attempts=3, window_seconds=60 * 60, lockout_seconds=60 * 60,
)
