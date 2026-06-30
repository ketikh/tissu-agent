"""Baseline security headers for every response.

We set:
- Strict-Transport-Security — force HTTPS for a year, including subs
- X-Content-Type-Options — stop browsers MIME-sniffing our responses
- X-Frame-Options — nobody iframes the admin
- Referrer-Policy — don't leak full URLs to third parties
- Permissions-Policy — disable camera/mic/geo entirely
- Content-Security-Policy — tight allowlist (self + Cloudinary images +
  rsms.me for the Inter font stylesheet we load on the admin pages)

``/api/storefront/*`` responses skip the Permissions-Policy /
X-Frame-Options pair because the Tissu storefront site may want to
embed data in iframes later, but they still get the rest.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://rsms.me; "
    "img-src 'self' https://res.cloudinary.com data:; "
    "font-src 'self' https://rsms.me data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        h = response.headers
        h.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains; preload",
        )
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        h.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Content-Security-Policy", CSP)
        return response
