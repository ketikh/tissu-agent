"""Tissu Agent Server.

AI-powered sales agent for Tissu Shop. Facebook Messenger bot with
Gemini Vision, WhatsApp owner notifications, and admin panel.
"""
import os
import asyncio
import json
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import asyncpg
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from src.config import API_HOST, API_PORT
from src.db import (
    init_db, get_db, close_pool, DEFAULT_TENANT_ID,
    find_admin_user_by_email, find_admin_user_by_id,
    find_admin_users_by_email_global,
    mark_admin_user_logged_in, update_admin_user_password,
    create_password_reset, consume_password_reset,
    invalidate_all_password_resets_for,
    list_tenants, get_tenant, create_tenant, update_tenant_status,
    create_admin_user_pending, create_activation_token,
    update_tenant_fb_credentials, delete_tenant_cascade,
    update_tenant_features, list_pricing_plans, update_pricing_plan,
    log_super_action, list_super_actions, regenerate_tenant_api_key,
    bump_admin_session_epoch, get_admin_session_epoch,
    get_bot_config, upsert_bot_config,
    get_site_sections, upsert_site_section,
    list_promo_codes, create_promo_code, update_promo_code, delete_promo_code,
    list_necklace_options, create_necklace_option, update_necklace_option, delete_necklace_option,
    get_necklace_base_price, set_necklace_base_price,
    list_gallery_photos, create_gallery_photo, update_gallery_photo, delete_gallery_photo,
    list_product_gallery, create_product_gallery_photo,
    update_product_gallery_photo, delete_product_gallery_photo,
    create_site_order, list_site_orders, get_site_order,
    update_site_order_status, ORDER_STATUSES,
    list_reviews, create_review, update_review,
    reorder_reviews, delete_review,
    list_size_variants, create_size_variant, delete_size_variant,
    size_variant_map_for_ids,
    list_inspiration_photos, create_inspiration_photo,
    update_inspiration_photo, delete_inspiration_photo,
)
from src.sessions import IMPERSONATION_SECONDS
from src.secrets_vault import encrypt_secret, redacted
from src.passwords import needs_rehash, verify_password, hash_password
from src.sessions import (
    SESSION_COOKIE_NAME, CSRF_COOKIE_NAME,
    SHORT_SESSION_SECONDS, LONG_SESSION_SECONDS,
    issue_session_token, load_session_token,
    issue_csrf_token, cookie_kwargs,
)
from src.rate_limit import login_limiter, password_reset_limiter
from src.tokens import generate_reset_token, hash_reset_token


def get_tenant_id(request: Request) -> str:
    """Pull the tenant_id that the APIKeyMiddleware stashed on the request.
    Falls back to DEFAULT_TENANT_ID for non-/api paths and exempt endpoints,
    so internal callers and Meta webhooks get a sensible value."""
    return getattr(request.state, "tenant_id", DEFAULT_TENANT_ID)
from src.models import ChatRequest, ChatResponse, LeadCreate, ContentCreate
from src.engine import run_agent
from src.agents.support_sales import get_support_sales_agent
from src.agents.marketing import get_marketing_agent
from src.channels import get_adapter, ADAPTERS
from src.webhooks.facebook import router as fb_router
from src.webhooks.telegram import router as tg_router
from src.webhooks.whatsapp import router as wa_router
from src.api.storefront import router as storefront_router, simple_router as products_router
from src.auth import APIKeyMiddleware, AdminSessionMiddleware
from src.security_headers import SecurityHeadersMiddleware


async def _register_telegram_webhook() -> None:
    """If a Telegram bot token is configured, register our /tg-webhook URL
    so Telegram pushes owner messages to us. Idempotent — re-registering
    with the same URL is a cheap no-op."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return
    public_url = os.environ.get("PUBLIC_URL", "https://tissu-agent-production.up.railway.app").rstrip("/")
    webhook_url = f"{public_url}/tg-webhook"
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/setWebhook",
                json={"url": webhook_url, "drop_pending_updates": False},
            )
            data = resp.json()
            if data.get("ok"):
                print(f"[TG] Webhook set: {webhook_url}", flush=True)
            else:
                print(f"[TG] Webhook set failed: {data}", flush=True)
    except Exception as e:
        print(f"[TG] Webhook setup error: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_knowledge_base()
    await _register_telegram_webhook()
    yield
    await close_pool()


app = FastAPI(
    title="Tissu Agents",
    description="AI-powered sales agent for Tissu Shop",
    version="0.2.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key / session guard on /api/*. Must be added AFTER CORS so
# preflight (OPTIONS) requests still get the CORS headers before we
# check the key.
app.add_middleware(APIKeyMiddleware)

# Redirect unauthenticated /admin/* HTML requests to /admin/login.
# Runs in addition to the API key middleware; the two don't overlap
# because this one only acts on /admin/* paths.
app.add_middleware(AdminSessionMiddleware)

# Baseline HTTP security headers on every response (HSTS, CSP,
# X-Frame-Options, etc.). Added last so it wraps every other
# middleware's output.
app.add_middleware(SecurityHeadersMiddleware)

# Include webhook routers
app.include_router(fb_router)
app.include_router(wa_router)
app.include_router(tg_router)
# Public storefront read API — also under /api/* so it goes through the
# same X-API-Key gate; the router itself re-uses the tenant_id from the
# middleware to scope every query.
app.include_router(storefront_router)
app.include_router(products_router)


# ── Pages ────────────────────────────────────────────────────

@app.get("/")
async def root_redirect():
    """The root URL used to serve the chat widget, but the Tissu bot lives
    on Facebook Messenger / Instagram DMs — this server is the admin
    backend, not a customer-facing site. Send visitors straight to the
    admin panel instead of a confusing chat shell."""
    return RedirectResponse(url="/admin", status_code=302)


# CSP for the admin page — mirrors the default plus an allowance for the
# Pinterest agent iframe loaded inside the "Pinterest" tab.
_ADMIN_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://rsms.me; "
    "img-src 'self' https://res.cloudinary.com data:; "
    "font-src 'self' https://rsms.me data:; "
    "connect-src 'self'; "
    "frame-src https://pinterest-agent.railway.app; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    html_path = Path(__file__).parent / "admin.html"
    # Always serve fresh — the admin HTML is small and iterating on UX
    # feels broken when browsers cache the previous revision for hours.
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Content-Security-Policy": _ADMIN_CSP,
        },
    )


# Embedded Pinterest agent dashboard — the Pinterest agent is a separate
# Railway deployment; we just iframe its admin so the operator stays inside
# one tab. CSP is loosened only for this route so the iframe loads.
PINTEREST_DASHBOARD_URL = "https://pinterest-agent.railway.app/admin/dashboard"


@app.get("/admin/pinterest", response_class=HTMLResponse)
async def admin_pinterest():
    html = f"""<!DOCTYPE html>
<html lang="ka">
<head>
    <meta charset="utf-8">
    <title>Pinterest Agent — Tissu Admin</title>
    <style>
        html, body {{ margin: 0; padding: 0; height: 100%; background: #fafafa; }}
        iframe {{ width: 100%; height: 100vh; border: 0; display: block; }}
    </style>
</head>
<body>
    <iframe src="{PINTEREST_DASHBOARD_URL}" allow="clipboard-write"></iframe>
</body>
</html>"""
    csp = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "frame-src https://pinterest-agent.railway.app; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    )
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": csp,
        },
    )


# ── Admin authentication ────────────────────────
# Real email + password form backed by argon2id, with a signed
# HTTP-only session cookie. Rate-limited per IP. CSRF protection
# for the non-login admin forms comes in the security-headers commit.

def _is_secure_request(request: Request) -> bool:
    """Cookie Secure flag — off on plain HTTP localhost, on for prod."""
    if request.url.scheme == "https":
        return True
    # Railway / Nginx terminate TLS then forward as HTTP — honor the
    # X-Forwarded-Proto header when it says "https".
    forwarded = request.headers.get("x-forwarded-proto", "").lower()
    return forwarded == "https"


def _client_ip(request: Request) -> str:
    """Best-effort client IP — uses the first X-Forwarded-For entry
    when behind Railway's proxy, otherwise the raw socket address."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_form(request: Request):
    # If the visitor already has a valid session, skip the login form.
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing and load_session_token(existing) is not None:
        return RedirectResponse(url="/admin", status_code=302)
    html_path = Path(__file__).parent / "admin-login.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/admin/login")
async def admin_login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember: str = Form(""),
):
    ip = _client_ip(request)
    # Rate limit first — same generic error on all failures to avoid
    # user-enumeration via timing or message differences.
    if not login_limiter.allow(ip):
        return RedirectResponse(url="/admin/login?error=locked", status_code=303)

    # Search across every tenant — the login form is single-email, no
    # tenant selector. Iterate all matches and verify the password
    # against each so a customer in a non-default tenant can sign in.
    candidates = await find_admin_users_by_email_global(email.strip().lower())
    user = None
    ok = False
    for candidate in candidates:
        if verify_password(password, candidate["password_hash"]):
            user = candidate
            ok = True
            break

    if not ok:
        login_limiter.record_failure(ip)
        # Same 303 back to the form with a generic error — no mention of
        # whether the email existed. (Constant-time: we always call
        # verify_password on at least one hash, so timing is similar
        # whether the user exists or not. For strict parity we'd hash a
        # dummy on the miss path — acceptable gap for v1.)
        return RedirectResponse(url="/admin/login?error=bad", status_code=303)

    # Success — clear prior failures, stamp last_login_at, rotate the
    # password hash if the cost parameters moved since it was stored.
    login_limiter.record_success(ip)
    await mark_admin_user_logged_in(user["id"])
    if needs_rehash(user["password_hash"]):
        try:
            await update_admin_user_password(user["id"], hash_password(password))
        except Exception:
            pass  # silent — rehash is a nice-to-have, not login-blocking

    max_age = LONG_SESSION_SECONDS if remember else SHORT_SESSION_SECONDS
    session_token = issue_session_token(user["id"], user["tenant_id"])
    csrf_token = issue_csrf_token(session_token)

    response = RedirectResponse(url="/admin", status_code=303)
    secure = _is_secure_request(request)
    response.set_cookie(
        SESSION_COOKIE_NAME, session_token, **cookie_kwargs(max_age, secure),
    )
    # CSRF cookie is readable by JS (httponly=False) so the admin
    # frontend can echo it back in X-CSRF-Token. Same salt would make
    # it a session-equivalent bearer token — we use a different salt
    # in sessions.py.
    response.set_cookie(
        CSRF_COOKIE_NAME, csrf_token,
        max_age=max_age, httponly=False, secure=secure,
        samesite="lax", path="/",
    )
    return response


@app.post("/admin/logout")
async def admin_logout(request: Request):
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    return response


@app.get("/admin/logout")
async def admin_logout_get(request: Request):
    # Convenience for the "logout" link in the header — browsers send
    # GET on <a href>, so we mirror the POST behavior.
    return await admin_logout(request)


# ── Password reset ─────────────────────────────
# We generate a random one-time token, store its hash in
# admin_password_resets, and email the raw token back to the user.
# Without SMTP configured (see TISSU-HANDOFF.md §7) we fall back to
# printing the link to the server log so the owner can rescue
# themselves in an emergency.

@app.get("/admin/forgot-password", response_class=HTMLResponse)
async def admin_forgot_password_form():
    html_path = Path(__file__).parent / "admin-forgot-password.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/forgot-password")
async def admin_forgot_password_submit(
    request: Request,
    email: str = Form(...),
):
    ip = _client_ip(request)
    if not password_reset_limiter.allow(ip):
        return RedirectResponse(url="/admin/forgot-password?error=rate", status_code=303)
    password_reset_limiter.record_failure(ip)  # count every request, not just successes

    email_norm = email.strip().lower()
    # Search globally — same reasoning as the login form: there's no
    # tenant selector, so a customer in any tenant has to be able to
    # reset by email alone. If multiple users share an email across
    # tenants we send a reset link for each (rare edge case).
    candidates = await find_admin_users_by_email_global(email_norm)
    public = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")
    for user in candidates:
        raw, token_hash = generate_reset_token()
        await create_password_reset(user["id"], token_hash, ttl_seconds=30 * 60)
        link = f"{public}/admin/reset-password?token={raw}"
        # Email hook. If no SMTP is wired up, print to the log so the
        # operator can grab the link from the deploy console.
        _send_or_log_reset_email(email_norm, link)

    # Same response regardless of whether the email exists — no user
    # enumeration via different status / timing on this endpoint.
    return RedirectResponse(url="/admin/forgot-password?sent=1", status_code=303)


def _send_or_log_reset_email(to_email: str, link: str) -> None:
    """Deliver the reset link. Wraps SMTP in a try/except so an
    outbound-mail outage can't lock the owner out permanently —
    the link also lands in the server log as a last resort."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    # Log first (never logs the full token value in a real deployment —
    # but for v1 we need the link reachable, and the log is only
    # readable by the operator).
    print(f"[password-reset] link for {to_email}: {link}", flush=True)
    if not smtp_host:
        # No SMTP configured — the log line above is the fallback.
        return
    # SMTP branch intentionally left minimal — wire up in a separate
    # ticket once we pick a provider (SendGrid / Postmark / etc.). For
    # now we fail-quiet so the flow still completes.
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = os.getenv("SMTP_FROM", "no-reply@tissu.local")
        msg["To"] = to_email
        msg["Subject"] = "Tissu ადმინი — პაროლის აღდგენა"
        msg.set_content(
            f"შენ ითხოვე პაროლის განახლება Tissu ადმინზე.\n\n"
            f"ეს ბმული მოქმედებს 30 წუთი:\n{link}\n\n"
            f"თუ შენ არ მოგითხოვია, უბრალოდ უგულებელყავი ეს წერილი."
        )
        host = smtp_host
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASSWORD", "")
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
    except Exception as e:
        print(f"[password-reset] SMTP send failed: {e}", flush=True)


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send a plain-text email via SMTP. Returns False (and logs) when SMTP
    isn't configured or the send fails — callers must treat email as best-effort
    so an outbound-mail problem never breaks the surrounding request."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    to_email = (to_email or "").strip()
    if not smtp_host or not to_email:
        print(f"[email] skipped (smtp set: {bool(smtp_host)}, to: {to_email or '—'}): {subject}", flush=True)
        return False
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = os.getenv("SMTP_FROM", "no-reply@tissu.local")
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASSWORD", "")
        with smtplib.SMTP(smtp_host, port, timeout=10) as s:
            s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
        print(f"[email] sent: {subject} -> {to_email}", flush=True)
        return True
    except Exception as e:
        print(f"[email] send failed ({to_email}): {e}", flush=True)
        return False


def _send_order_emails(order: dict) -> None:
    """Best-effort: email the customer an order confirmation/invoice and notify
    the shop owner. Never raises — a mail problem must not fail order creation."""
    try:
        order_id = order.get("id") or ""
        name = (order.get("customer_name") or "").strip()
        cust_email = (order.get("customer_email") or "").strip()
        total = order.get("total") or 0
        lines = []
        for it in (order.get("items") or []):
            pname = (it.get("product_name_ka") or it.get("product_name_en") or "").strip() or "—"
            qty = it.get("quantity") or 1
            price = it.get("price") or 0
            lines.append(f"  • {pname} × {qty} — {price}₾")
        items_text = "\n".join(lines) if lines else "  —"
        addr = ", ".join(p for p in [order.get("address_street"), order.get("address_city")] if p)
        body = (
            f"გმადლობთ შეკვეთისთვის, {name}!\n\n"
            f"შეკვეთის ნომერი: {order_id}\n\n"
            f"პროდუქცია:\n{items_text}\n\n"
            f"ჯამი: {total}₾\n"
            + (f"მისამართი: {addr}\n" if addr else "")
            + "\nმალე დაგიკავშირდებით შეკვეთის დასადასტურებლად.\n\nTissu · tissu.ge"
        )
        if cust_email:
            _send_email(cust_email, f"Tissu — შეკვეთა მიღებულია (#{order_id})", body)
        owner_email = os.getenv("ORDER_NOTIFY_EMAIL", "").strip()
        if owner_email:
            _send_email(owner_email, f"ახალი შეკვეთა #{order_id} — {name} ({total}₾)", body)
    except Exception as e:
        print(f"[order-email] failed: {e}", flush=True)


@app.get("/admin/reset-password", response_class=HTMLResponse)
async def admin_reset_password_form(request: Request, token: str = ""):
    # We don't verify the token at GET time — the user might refresh
    # the page, and we'd burn a one-shot token. Verification happens
    # on submit.
    html_path = Path(__file__).parent / "admin-reset-password.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/whoami")
async def api_whoami(request: Request):
    """Return who is currently signed in so the admin UI can show
    the user's email + shop name in the header. Session-cookie
    callers get a full payload; X-API-Key callers get 'none' for
    admin_user because the key doesn't identify a human."""
    user_id = getattr(request.state, "admin_user_id", None)
    if not user_id:
        return {
            "admin_user": None,
            "tenant_id": getattr(request.state, "tenant_id", DEFAULT_TENANT_ID),
            "auth": getattr(request.state, "auth_source", "api_key"),
        }
    user = await find_admin_user_by_id(int(user_id))
    if not user:
        return {"admin_user": None}
    tenant = await get_tenant(user["tenant_id"])
    t = tenant or {}
    impersonator_id = getattr(request.state, "impersonator_id", None)
    impersonator_email = None
    if impersonator_id:
        try:
            imp = await find_admin_user_by_id(int(impersonator_id))
            if imp:
                impersonator_email = imp["email"]
        except Exception:
            pass
    return {
        "admin_user": {
            "id": user["id"],
            "email": user["email"],
            "role": user["role"],
        },
        "tenant": {
            "tenant_id": user["tenant_id"],
            "shop_name": t.get("shop_name") or user["tenant_id"],
            "status": t.get("status") or "active",
            # New Phase 1 fields — frontend Phase 3 will use these to
            # decide which admin tabs to render.
            "feature_set": t.get("feature_set") or "bot",
            "pricing_tier": t.get("pricing_tier") or "start",
            "bot_enabled": bool(t.get("bot_enabled")),
            "site_enabled": bool(t.get("site_enabled")),
            # Backward-compat: keep the old "plan" key returning the
            # pricing_tier value so any caller still reading it works.
            "plan": t.get("pricing_tier") or "start",
        },
        # Phase 2: Surface impersonation so admin.html can render a
        # "you are acting as <X>" banner with a stop button.
        "impersonation": (
            {
                "active": True,
                "impersonator_email": impersonator_email,
            }
            if impersonator_id else {"active": False}
        ),
    }


async def _require_super_admin(request: Request) -> dict:
    """Guard used by every /api/super/* endpoint. Loads the logged-in
    admin user, returns it if role='super', otherwise raises 403.
    Relies on AdminSessionMiddleware / APIKeyMiddleware to already
    have verified the session and stashed admin_user_id on state."""
    user_id = getattr(request.state, "admin_user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="not signed in")
    user = await find_admin_user_by_id(int(user_id))
    if not user or user.get("role") != "super":
        raise HTTPException(status_code=403, detail="super-admin only")
    return user


@app.get("/admin/suspended", response_class=HTMLResponse)
async def admin_suspended_page():
    html_path = Path(__file__).parent / "admin-suspended.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/admin/super", response_class=HTMLResponse)
async def admin_super_page(request: Request):
    """Serve the super-admin HTML. The API behind it enforces the
    super-admin role — this route just serves the shell, so even a
    plain admin loading the page sees an empty table."""
    html_path = Path(__file__).parent / "admin-super.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/admin/super/pricing", response_class=HTMLResponse)
async def admin_super_pricing_page(request: Request):
    """Serve the pricing matrix HTML — the 9-cell grid where the
    operator sets per-plan prices. The /api/super/pricing endpoint
    behind it enforces the super-admin role."""
    html_path = Path(__file__).parent / "admin-super-pricing.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/super/tenants")
async def api_super_list_tenants(request: Request):
    await _require_super_admin(request)
    return {"tenants": await list_tenants()}


@app.get("/api/super/pricing")
async def api_super_list_pricing(request: Request):
    """Return the 9-cell price matrix for the super-admin pricing page."""
    await _require_super_admin(request)
    return {"pricing_plans": await list_pricing_plans()}


@app.put("/api/super/pricing/{feature_set}/{pricing_tier}")
async def api_super_update_pricing(
    feature_set: str, pricing_tier: str, request: Request,
):
    """Owner sets / clears the price for one cell of the matrix.
    Pass null for any field to leave it empty."""
    await _require_super_admin(request)
    if feature_set not in ("bot", "site", "combo"):
        raise HTTPException(400, "feature_set must be bot/site/combo")
    if pricing_tier not in ("start", "grow", "pro"):
        raise HTTPException(400, "pricing_tier must be start/grow/pro")
    data = await request.json()

    def _num(v):
        if v is None or v == "":
            return None
        try:
            n = float(v)
        except (TypeError, ValueError):
            raise HTTPException(400, "price values must be numeric or null")
        if n < 0:
            raise HTTPException(400, "price must be >= 0")
        return n

    price_gel = _num(data.get("price_gel"))
    setup_fee_gel = _num(data.get("setup_fee_gel"))
    description = data.get("description")
    if description is not None:
        description = str(description).strip() or None

    await update_pricing_plan(
        feature_set, pricing_tier, price_gel, setup_fee_gel, description,
    )
    return {"ok": True}


@app.put("/api/super/tenants/{tenant_id}/features")
async def api_super_update_features(tenant_id: str, request: Request):
    """Flip bot_enabled / site_enabled per-tenant (independent of
    feature_set). Useful when a tenant's bot is broken and the
    operator wants to disable just that without changing the plan."""
    await _require_super_admin(request)
    if tenant_id == DEFAULT_TENANT_ID:
        raise HTTPException(400, "default tenant feature flags are fixed")
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")
    data = await request.json()
    bot = data.get("bot_enabled")
    site = data.get("site_enabled")
    await update_tenant_features(
        tenant_id,
        bot_enabled=bool(bot) if bot is not None else None,
        site_enabled=bool(site) if site is not None else None,
    )
    return {"ok": True}


@app.post("/api/super/tenants")
async def api_super_create_tenant(request: Request):
    """Create a new tenant + their first admin_user + a 7-day
    activation link. Returns the activation URL — this is the *only*
    time the super-admin gets to see it, so the UI copies it into a
    one-off card for the operator to send to the customer.

    Two plan axes:
      feature_set  ∈ {bot, site, combo}  — what the tenant gets
      pricing_tier ∈ {start, grow, pro}  — the price band

    For backward compat, callers passing the old ``plan`` field with a
    pricing_tier value are still accepted (defaults feature_set='bot').
    """
    await _require_super_admin(request)
    data = await request.json()
    shop_name = (data.get("shop_name") or "").strip()
    owner_email = (data.get("owner_email") or "").strip().lower()
    feature_set = (data.get("feature_set") or "bot").strip().lower()
    pricing_tier = (
        data.get("pricing_tier") or data.get("plan") or "start"
    ).strip().lower()
    if not shop_name or not owner_email:
        raise HTTPException(400, "shop_name + owner_email required")
    if feature_set not in ("bot", "site", "combo"):
        raise HTTPException(400, "feature_set must be bot/site/combo")
    if pricing_tier not in ("start", "grow", "pro"):
        raise HTTPException(400, "pricing_tier must be start/grow/pro")
    if "@" not in owner_email or len(owner_email) > 120:
        raise HTTPException(400, "invalid email")

    # Deterministic tenant_id from the shop name + a 6-char suffix so
    # two shops with the same name don't collide. The suffix uses
    # secrets.token_hex so it's hard to guess and doesn't leak order
    # of creation to an external observer.
    import re
    import secrets
    slug_base = re.sub(r"[^a-z0-9]+", "_", shop_name.lower()).strip("_")[:24] or "shop"
    tenant_id = f"{slug_base}_{secrets.token_hex(3)}"

    try:
        await create_tenant(
            tenant_id, shop_name, owner_email,
            pricing_tier=pricing_tier, feature_set=feature_set,
            status="trial",
        )
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(409, "tenant_id clash, please retry")

    # Create the pending admin_user (no usable password yet) and a
    # 7-day activation token.
    user_id = await create_admin_user_pending(tenant_id, owner_email, role="owner")
    raw, token_hash = generate_reset_token()
    await create_activation_token(user_id, token_hash)

    public = os.getenv("PUBLIC_URL", "http://localhost:8000").rstrip("/")
    activation_url = f"{public}/admin/activate?token={raw}"

    super_user = await _require_super_admin(request)  # idempotent, returns user
    await log_super_action(
        super_user["id"], "create_tenant", target_tenant_id=tenant_id,
        payload={
            "shop_name": shop_name, "owner_email": owner_email,
            "feature_set": feature_set, "pricing_tier": pricing_tier,
        },
        ip=_client_ip(request),
    )

    return {
        "tenant_id": tenant_id,
        "activation_url": activation_url,
        "note": "Link is one-time use, 7 day expiry.",
    }


@app.post("/api/super/tenants/{tenant_id}/suspend")
async def api_super_suspend(tenant_id: str, request: Request):
    super_user = await _require_super_admin(request)
    # Don't let the super-admin accidentally lock themselves out of
    # the default tenant (that's where their own account lives).
    if tenant_id == DEFAULT_TENANT_ID:
        raise HTTPException(400, "cannot suspend the default tenant")
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")
    await update_tenant_status(tenant_id, "suspended")
    await log_super_action(
        super_user["id"], "suspend", target_tenant_id=tenant_id,
        payload={"shop_name": t.get("shop_name")},
        ip=_client_ip(request),
    )
    return {"ok": True}


@app.post("/api/super/tenants/{tenant_id}/fb-credentials")
async def api_super_set_fb_credentials(tenant_id: str, request: Request):
    """Save a tenant's Facebook Messenger webhook credentials. The
    page token is stored encrypted via Fernet; the verify token and
    page id stay plaintext (page_id is public anyway, verify token
    is a shared secret with Meta's webhook POSTs — we just need to
    match it on inbound)."""
    await _require_super_admin(request)
    data = await request.json()
    fb_page_id = (data.get("fb_page_id") or "").strip()
    fb_page_token = (data.get("fb_page_token") or "").strip()
    fb_verify_token = (data.get("fb_verify_token") or "").strip()

    if not fb_page_id or not fb_page_token or not fb_verify_token:
        raise HTTPException(
            400,
            "fb_page_id, fb_page_token, fb_verify_token all required",
        )

    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")

    try:
        encrypted = encrypt_secret(fb_page_token)
    except RuntimeError as e:
        # FB_TOKEN_ENCRYPTION_KEY not set on the server — surface the
        # error to the operator instead of storing garbage.
        raise HTTPException(500, str(e))

    await update_tenant_fb_credentials(
        tenant_id, fb_page_id, encrypted, fb_verify_token,
    )
    return {
        "ok": True,
        "fb_page_id": fb_page_id,
        "fb_page_token_preview": redacted(fb_page_token, keep=6),
    }


@app.delete("/api/super/tenants/{tenant_id}")
async def api_super_delete_tenant(tenant_id: str, request: Request):
    """Permanently wipe a tenant + all their data. Super-admin only,
    irreversible. The default tenant is hard-blocked because it
    carries the operator's own super account."""
    super_user = await _require_super_admin(request)
    if tenant_id == DEFAULT_TENANT_ID:
        raise HTTPException(400, "cannot delete the default tenant")
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")
    try:
        receipt = await delete_tenant_cascade(tenant_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    await log_super_action(
        super_user["id"], "delete_tenant", target_tenant_id=tenant_id,
        payload={"shop_name": t.get("shop_name"), "deleted": receipt},
        ip=_client_ip(request),
    )
    return {"ok": True, "tenant_id": tenant_id, "deleted": receipt}


@app.post("/api/super/tenants/{tenant_id}/mark-paid")
async def api_super_mark_paid(tenant_id: str, request: Request):
    """Extend the paid-through date by 30 days and flip the status
    back to 'active'. This is the button the super-admin presses
    when a bank transfer lands."""
    super_user = await _require_super_admin(request)
    from datetime import timedelta
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")
    # Extend from whichever is later: today, or the current paid-through.
    now = datetime.now(timezone.utc)
    current = t.get("payment_due_date")
    base = now
    if current:
        try:
            base = max(now, datetime.fromisoformat(current))
        except Exception:
            base = now
    new_due = (base + timedelta(days=30)).isoformat()
    await update_tenant_status(tenant_id, "active", payment_due_date=new_due)
    await log_super_action(
        super_user["id"], "mark_paid", target_tenant_id=tenant_id,
        payload={"shop_name": t.get("shop_name"), "new_due": new_due},
        ip=_client_ip(request),
    )
    return {"ok": True, "payment_due_date": new_due}


# ── Phase 2: regenerate API key + impersonate ──────────────

@app.post("/api/super/tenants/{tenant_id}/regenerate-key")
async def api_super_regenerate_key(tenant_id: str, request: Request):
    """Issue a fresh admin API key for the tenant and invalidate the
    old one. The new key is returned ONCE — the operator must copy
    it out of the response and hand it to the customer; we never
    store the raw key, only the api_keys row."""
    super_user = await _require_super_admin(request)
    if tenant_id == DEFAULT_TENANT_ID:
        raise HTTPException(400, "cannot rotate the default tenant key here")
    t = await get_tenant(tenant_id)
    if not t:
        raise HTTPException(404, "tenant not found")
    new_key = await regenerate_tenant_api_key(tenant_id)
    await log_super_action(
        super_user["id"], "regenerate_key", target_tenant_id=tenant_id,
        payload={
            "shop_name": t.get("shop_name"),
            "key_preview": new_key[:14] + "…",
        },
        ip=_client_ip(request),
    )
    return {
        "ok": True, "tenant_id": tenant_id, "new_key": new_key,
        "note": "Copy this now — the raw key is shown only once.",
    }


# NOTE: the previous "login as" / impersonation flow lived here. It
# was removed because giving the platform operator silent access to a
# tenant's admin is a privacy red flag — even with audit logs, the
# customer never knew when their account was being viewed. The
# customer-facing replacement (forgot password + per-tenant settings
# page where they manage their own password and active sessions)
# lives in /admin/forgot-password and /admin/settings.
#
# The old /admin/stop-impersonation route is also gone for the same
# reason. Existing audit rows from the impersonation era stay in
# super_admin_actions for the historical record.


@app.get("/api/super/audit")
async def api_super_audit(request: Request, limit: int = 100):
    """Paginated tail of the super_admin_actions log. Defaults to
    100 most recent rows; the UI shows a single page."""
    await _require_super_admin(request)
    limit = max(1, min(int(limit or 100), 500))
    return {"actions": await list_super_actions(limit=limit)}


@app.get("/admin/super/audit", response_class=HTMLResponse)
async def admin_super_audit_page(request: Request):
    html_path = Path(__file__).parent / "admin-super-audit.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── Activation (set your password on first login) ──────────
# Uses the same admin_password_resets table as the regular reset
# flow — an activation token is just a reset token with a 7-day TTL
# handed out by the super-admin instead of requested by the user.

@app.get("/admin/activate", response_class=HTMLResponse)
async def admin_activate_form(request: Request, token: str = ""):
    """Render the 'set your password' page for first-time activation.
    Reuses the reset-password HTML with a flag in the URL so the form
    can show the right Georgian welcome copy."""
    html_path = Path(__file__).parent / "admin-reset-password.html"
    html = html_path.read_text(encoding="utf-8")
    # Light tweak: replace the default heading for activation flow so
    # it reads "ანგარიშის გააქტიურება" instead of "ახალი პაროლი".
    html = html.replace(
        "<h1>ახალი პაროლი</h1>",
        "<h1>ანგარიშის გააქტიურება</h1>",
    ).replace(
        "<p class=\"subtitle\">შეარჩიე ახალი პაროლი ანგარიშისთვის — მინიმუმ 8 სიმბოლო.</p>",
        "<p class=\"subtitle\">კეთილი იყოს მობრძანება! შექმენი პაროლი და გადადი ადმინზე — მინიმუმ 8 სიმბოლო.</p>",
    ).replace(
        'action="/admin/reset-password"',
        'action="/admin/activate"',
    )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/admin/activate")
async def admin_activate_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    """Set the initial password via the same consume-token flow as the
    reset endpoint. After success, auto-log-in so the customer lands
    straight on /admin without a second password prompt."""
    if len(password) < 8:
        return RedirectResponse(
            url=f"/admin/activate?token={token}&error=short", status_code=303,
        )
    if password != password2:
        return RedirectResponse(
            url=f"/admin/activate?token={token}&error=mismatch", status_code=303,
        )
    token_hash = hash_reset_token(token)
    user_id = await consume_password_reset(token_hash)
    if not user_id:
        return RedirectResponse(
            url="/admin/activate?error=invalid", status_code=303,
        )
    new_hash = hash_password(password)
    await update_admin_user_password(user_id, new_hash)
    await invalidate_all_password_resets_for(user_id)
    user = await find_admin_user_by_id(user_id)
    if not user:
        return RedirectResponse(url="/admin/login?activated=1", status_code=303)
    await mark_admin_user_logged_in(user_id)

    # If the visitor already has a valid admin session for some OTHER
    # user (e.g. the super-admin testing an activation link in their
    # own browser), don't clobber their cookie with the freshly
    # activated account — that would silently log them out. Activate
    # the password and bounce back to /admin/super with a success
    # flag. A real first-time customer arrives without a session and
    # hits the auto-login path below.
    existing_cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
    if existing_cookie:
        existing = load_session_token(
            existing_cookie, max_age_seconds=LONG_SESSION_SECONDS,
        )
        if existing and existing.get("user_id") and existing["user_id"] != user_id:
            return RedirectResponse(
                url="/admin/super?activated=" + user["tenant_id"],
                status_code=303,
            )

    # Auto-login: issue a fresh session cookie so the newly activated
    # user lands on /admin without a separate trip through /admin/login.
    session_token = issue_session_token(user["id"], user["tenant_id"])
    csrf_token = issue_csrf_token(session_token)
    response = RedirectResponse(url="/admin", status_code=303)
    secure = _is_secure_request(request)
    response.set_cookie(
        SESSION_COOKIE_NAME, session_token, **cookie_kwargs(SHORT_SESSION_SECONDS, secure),
    )
    response.set_cookie(
        CSRF_COOKIE_NAME, csrf_token,
        max_age=SHORT_SESSION_SECONDS, httponly=False, secure=secure,
        samesite="lax", path="/",
    )
    return response


@app.post("/admin/reset-password")
async def admin_reset_password_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    if len(password) < 8:
        return RedirectResponse(
            url=f"/admin/reset-password?token={token}&error=short",
            status_code=303,
        )
    if password != password2:
        return RedirectResponse(
            url=f"/admin/reset-password?token={token}&error=mismatch",
            status_code=303,
        )

    token_hash = hash_reset_token(token)
    user_id = await consume_password_reset(token_hash)
    if not user_id:
        return RedirectResponse(
            url="/admin/reset-password?error=invalid",
            status_code=303,
        )

    # Rotate the password hash and kill every other outstanding
    # reset token for this user.
    new_hash = hash_password(password)
    await update_admin_user_password(user_id, new_hash)
    await invalidate_all_password_resets_for(user_id)
    return RedirectResponse(url="/admin/login?reset=1", status_code=303)


@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings_page(request: Request):
    """Account settings page: change password + logout all devices."""
    html_path = Path(__file__).parent / "admin-settings.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/admin/change-password")
async def admin_change_password(request: Request):
    """Change the logged-in admin's password.

    Body: {"old_password": str, "new_password": str, "new_password2": str}

    On success:
    - Updates the password hash
    - Bumps session_epoch → invalidates every other active session
    - Re-issues cookies for the current session with the new epoch
    so the caller stays logged in on this device.
    """
    user_id = getattr(request.state, "admin_user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="not signed in")

    body = await request.json()
    old_pw = (body.get("old_password") or "").strip()
    new_pw = (body.get("new_password") or "").strip()
    new_pw2 = (body.get("new_password2") or "").strip()

    if not old_pw or not new_pw:
        raise HTTPException(status_code=400, detail="missing fields")
    if len(new_pw) < 8:
        raise HTTPException(status_code=400, detail="password_too_short")
    if new_pw != new_pw2:
        raise HTTPException(status_code=400, detail="passwords_mismatch")

    user = await find_admin_user_by_id(int(user_id))
    if not user or not verify_password(old_pw, user["password_hash"]):
        raise HTTPException(status_code=400, detail="wrong_old_password")

    new_hash = hash_password(new_pw)
    await update_admin_user_password(int(user_id), new_hash)
    new_epoch = await bump_admin_session_epoch(int(user_id))

    # Re-issue session + CSRF cookies with the new epoch so this device
    # stays logged in; every other browser is kicked out automatically.
    tenant_id = getattr(request.state, "tenant_id", DEFAULT_TENANT_ID)
    new_token = issue_session_token(int(user_id), tenant_id, epoch=new_epoch)
    new_csrf = issue_csrf_token(new_token)
    secure = _is_secure_request(request)

    from fastapi.responses import JSONResponse as _JSONResponse
    response = _JSONResponse({"ok": True})
    response.set_cookie(
        SESSION_COOKIE_NAME, new_token,
        **cookie_kwargs(LONG_SESSION_SECONDS, secure),
    )
    response.set_cookie(
        CSRF_COOKIE_NAME, new_csrf,
        max_age=LONG_SESSION_SECONDS, httponly=False,
        secure=secure, samesite="lax", path="/",
    )
    return response


@app.post("/api/admin/logout-all-devices")
async def admin_logout_all_devices(request: Request):
    """Bump session_epoch → invalidate every active session for this
    user, including the current browser. Caller should redirect to
    /admin/login immediately after."""
    user_id = getattr(request.state, "admin_user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="not signed in")

    await bump_admin_session_epoch(int(user_id))

    from fastapi.responses import JSONResponse as _JSONResponse
    response = _JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    return response


@app.get("/api/admin/bot-config")
async def admin_get_bot_config(request: Request):
    """Return the current tenant's bot configuration."""
    tenant_id = get_tenant_id(request)
    cfg = await get_bot_config(tenant_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="bot config not found")
    return cfg


@app.put("/api/admin/bot-config")
async def admin_put_bot_config(request: Request):
    """Save (partial update) the current tenant's bot configuration."""
    tenant_id = get_tenant_id(request)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    allowed_fields = {
        "company_name", "bot_greeting", "product_catalog", "size_guide",
        "delivery_info", "payment_accounts", "currency_symbol",
        "off_topic_reply", "confidential_reply", "unavailable_reply",
        "custom_order_reply", "corporate_info", "tone",
        "emoji_enabled", "custom_instructions",
    }
    filtered = {k: v for k, v in data.items() if k in allowed_fields}
    if not filtered:
        raise HTTPException(status_code=400, detail="no valid fields provided")
    await upsert_bot_config(tenant_id, filtered)
    return {"ok": True}


_VALID_CMS_PAGES = frozenset({"home", "about", "faq", "shop", "contact"})
_VALID_CMS_SECTIONS: dict[str, frozenset[str]] = {
    "home": frozenset({
        "hero", "marquee", "products_grid", "necklace_grid",
        "about", "reviews", "newsletter",
    }),
    "faq":     frozenset({"items"}),
    "shop":    frozenset({"hero"}),
    "about":   frozenset({"hero", "two_sides", "process", "cta"}),
    "contact": frozenset({"hero", "info", "wholesale_card", "details"}),
}


@app.get("/api/admin/site/{page}")
async def admin_get_site_page(page: str, request: Request):
    """Return all saved sections for one CMS page."""
    if page not in _VALID_CMS_PAGES:
        raise HTTPException(status_code=404, detail="unknown page")
    tenant_id = get_tenant_id(request)
    sections = await get_site_sections(tenant_id, page)
    return {"page": page, "sections": sections}


@app.put("/api/admin/site/{page}/{section}")
async def admin_put_site_section(page: str, section: str, request: Request):
    """Upsert one section's payload for a CMS page."""
    if page not in _VALID_CMS_PAGES:
        raise HTTPException(status_code=404, detail="unknown page")
    allowed = _VALID_CMS_SECTIONS.get(page, frozenset())
    if allowed and section not in allowed:
        raise HTTPException(status_code=400, detail="unknown section for this page")
    tenant_id = get_tenant_id(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    payload = body.get("payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    position = int(body.get("position", 0))
    await upsert_site_section(tenant_id, page, section, payload, position)
    return {"ok": True}


@app.get("/admin/chat-test", response_class=HTMLResponse)
async def admin_chat_test():
    """Text-only bot prompt tester. Handy for iterating on the system
    prompt without opening Messenger, but limited — it can't send
    photos and therefore doesn't exercise the Vision pipeline. For the
    full flow (photo → match → owner confirm) still use Facebook/IG."""
    html_path = Path(__file__).parent / "chat.html"
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── Legal pages (public) ────────────────────────
# Required by Meta App Review and by customer-facing trust. These are
# plain static HTML served from /legal/. Browsers can cache them.

def _legal_page(filename: str) -> HTMLResponse:
    path = Path(__file__).parent / "legal" / filename
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    return _legal_page("privacy.html")


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    return _legal_page("terms.html")


@app.get("/data-deletion", response_class=HTMLResponse)
async def data_deletion_page():
    return _legal_page("data-deletion.html")


# ── Meta Data-Deletion Callback ─────────────────
# Meta calls this when a user removes the app through Facebook's
# "Apps and Websites" panel. We must respond with a URL + confirmation
# code so Meta can show the user a status page. The heavy lifting
# (actually deleting the rows) is queued via the tickets table and the
# owner processes it manually from the admin insights.

@app.post("/api/meta/data-deletion")
async def meta_data_deletion(request: Request):
    import base64
    import hashlib
    import hmac
    import urllib.parse

    form = await request.form()
    signed_request = form.get("signed_request", "")
    user_id = ""
    if signed_request and "." in signed_request:
        try:
            _sig, payload_b64 = signed_request.split(".", 1)
            # base64url-decode with padding fix
            pad = "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad).decode("utf-8"))
            user_id = str(payload.get("user_id", ""))
        except Exception as e:
            print(f"[DATA-DELETE] Could not parse signed_request: {e}", flush=True)

    # Log the request so the owner can act on it. Uses the tickets table
    # which already surfaces in the admin Insights section.
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    try:
        await pool.execute(
            "INSERT INTO tickets (subject, description, status, priority, customer_email, conversation_id, tenant_id, created_at, updated_at) "
            "VALUES ($1, $2, 'open', 'urgent', $3, $4, $5, $6, $6)",
            "[DATA-DELETION] Meta user requested deletion",
            json.dumps({"user_id": user_id, "source": "meta_callback"}, ensure_ascii=False),
            "", f"meta_user_{user_id}" if user_id else "meta_user_unknown",
            DEFAULT_TENANT_ID, now,
        )
    except Exception as e:
        print(f"[DATA-DELETE] Failed to log ticket: {e}", flush=True)

    # Per Meta spec, respond with a URL where the user can check status
    # plus an opaque confirmation code.
    code = user_id or "anon"
    public = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app").rstrip("/")
    return {
        "url": f"{public}/data-deletion?code={urllib.parse.quote(code)}",
        "confirmation_code": code,
    }


# ── Agent Chat Endpoints ─────────────────────────────────────

@app.post("/api/support", response_model=ChatResponse)
async def chat_support(req: ChatRequest):
    enriched_message = req.message

    if req.customer_context:
        ctx = req.customer_context
        context_parts = []
        if ctx.name:
            context_parts.append(f"Customer name: {ctx.name}")
        if ctx.email:
            context_parts.append(f"Email: {ctx.email}")
        if ctx.product_interest:
            context_parts.append(f"Product interest: {ctx.product_interest}")
        if req.channel:
            context_parts.append(f"Channel: {req.channel}")
        if context_parts:
            enriched_message = f"[Context: {'; '.join(context_parts)}]\n\n{req.message}"

    agent = await get_support_sales_agent(tenant_id)
    result = await run_agent(agent, enriched_message, req.conversation_id)
    try:
        return ChatResponse(**result)
    except Exception:
        return ChatResponse(
            reply=result.get("reply", ""),
            conversation_id=result.get("conversation_id", ""),
            agent_type=result.get("agent_type", "support_sales"),
            tool_calls_made=result.get("tool_calls_made", []),
        )


@app.post("/api/webhook/{channel}")
async def channel_webhook(channel: str, request: Request):
    """Universal webhook endpoint for any channel (N8N routes here)."""
    if channel not in ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown channel: {channel}. Available: {list(ADAPTERS.keys())}")

    adapter = get_adapter(channel)
    payload = await request.json()
    chat_request = adapter.parse_incoming(payload)
    enriched_message = chat_request.message
    if chat_request.customer_context:
        ctx = chat_request.customer_context
        context_parts = [f"Channel: {channel}"]
        if ctx.name:
            context_parts.append(f"Customer name: {ctx.name}")
        if ctx.product_interest:
            context_parts.append(f"Product interest: {ctx.product_interest}")
        enriched_message = f"[Context: {'; '.join(context_parts)}]\n\n{chat_request.message}"

    agent = await get_support_sales_agent(tenant_id)
    result = await run_agent(agent, enriched_message, chat_request.conversation_id)
    return {"agent_response": result, "channel": channel}


@app.post("/api/marketing", response_model=ChatResponse)
async def chat_marketing(req: ChatRequest):
    agent = get_marketing_agent()
    result = await run_agent(agent, req.message, req.conversation_id)
    return ChatResponse(**result)


# ── Data Endpoints ───────────────────────────────────────────

@app.get("/api/leads")
async def list_leads(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM leads WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"leads": [dict(r) for r in rows]}


@app.post("/api/leads")
async def create_lead(lead: LeadCreate, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO leads (name, email, company, phone, source, notes, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
        lead.name, lead.email, lead.company, lead.phone, lead.source, lead.notes, tenant_id, now, now,
    )
    return {"id": row["id"], "message": "Lead created"}


@app.get("/api/tickets")
async def list_tickets(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM tickets WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"tickets": [dict(r) for r in rows]}


@app.get("/api/content")
async def list_content(content_type: str = "", status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM content WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if content_type:
        query += f" AND content_type = ${idx}"
        params.append(content_type)
        idx += 1
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"content": [dict(r) for r in rows]}


@app.post("/api/content")
async def create_content(item: ContentCreate, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO content (title, body, content_type, tags, scheduled_at, status, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, 'draft', $6, $7, $8) RETURNING id",
        item.title, item.body, item.content_type, json.dumps(item.tags), item.scheduled_at, tenant_id, now, now,
    )
    return {"id": row["id"], "message": "Content created"}


@app.get("/api/conversations")
async def list_conversations(agent_type: str = "", limit: int = 100, tenant_id: str = Depends(get_tenant_id)):
    """List recent conversations with message count and a snippet of the
    last customer message. Powers the admin ჩატები tab."""
    pool = await get_db()
    query = """
        SELECT c.id, c.agent_type, c.updated_at, c.created_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count,
               (SELECT m.content FROM messages m
                 WHERE m.conversation_id = c.id AND m.role = 'user'
                 ORDER BY m.created_at DESC LIMIT 1) AS last_user_msg,
               (SELECT m.content FROM messages m
                 WHERE m.conversation_id = c.id AND m.role IN ('assistant','model')
                 ORDER BY m.created_at DESC LIMIT 1) AS last_bot_msg
        FROM conversations c
        WHERE c.tenant_id = $1 AND c.id NOT LIKE 'debug_%'
    """
    params: list = [tenant_id]
    idx = 2
    if agent_type:
        query += f" AND c.agent_type = ${idx}"
        params.append(agent_type)
        idx += 1
    query += f" ORDER BY c.updated_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    from src.insights import clean_user_message
    out = []
    for r in rows:
        last_user = clean_user_message(r["last_user_msg"] or "")
        out.append({
            "id": r["id"],
            "agent_type": r["agent_type"],
            "updated_at": r["updated_at"],
            "created_at": r["created_at"],
            "msg_count": int(r["msg_count"] or 0),
            "last_user_msg": last_user[:140],
            "last_bot_msg": (r["last_bot_msg"] or "")[:140],
        })
    return {"conversations": out}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    # Verify the conversation belongs to this tenant before returning its
    # messages — prevents a tenant with a valid key from reading another
    # tenant's chat history by guessing IDs.
    owner = await pool.fetchrow(
        "SELECT tenant_id FROM conversations WHERE id = $1", conversation_id,
    )
    if not owner or owner["tenant_id"] != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = await pool.fetch(
        "SELECT role, content, created_at FROM messages "
        "WHERE conversation_id = $1 AND tenant_id = $2 ORDER BY created_at",
        conversation_id, tenant_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": [dict(r) for r in rows]}


@app.get("/api/knowledge")
async def list_knowledge(category: str = "", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM knowledge_base WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    rows = await pool.fetch(query, *params)
    return {"articles": [dict(r) for r in rows]}


@app.post("/api/knowledge")
async def add_knowledge(question: str, answer: str, category: str = "general", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO knowledge_base (question, answer, category, tenant_id, created_at) "
        "VALUES ($1, $2, $3, $4, $5) RETURNING id",
        question, answer, category, tenant_id, now,
    )
    return {"id": row["id"], "message": "Knowledge article added"}


# ── Inventory Endpoints ──────────────────────────────────────

def _cloudinary_configured() -> bool:
    """Cloudinary wins when CLOUDINARY_URL (or CLOUD/API key triplet) is set.
    Railway's filesystem is ephemeral — without Cloudinary, every deploy
    would wipe admin-uploaded photos. Local dev falls back to /static."""
    return bool(os.getenv("CLOUDINARY_URL")) or (
        os.getenv("CLOUDINARY_CLOUD_NAME") and os.getenv("CLOUDINARY_API_KEY") and os.getenv("CLOUDINARY_API_SECRET")
    )


def _upload_to_cloudinary(upload: UploadFile, public_id: str, folder: str = "tissu/uploads"):
    """Upload to Cloudinary and return the secure HTTPS URL. Returns None on failure.

    ``folder`` lets callers segregate uploads (e.g. ``tissu/gallery`` for
    lookbook photos that should stay out of the product folder)."""
    try:
        import cloudinary
        import cloudinary.uploader
        if os.getenv("CLOUDINARY_URL"):
            cloudinary.config(secure=True)
        else:
            cloudinary.config(
                cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
                api_key=os.getenv("CLOUDINARY_API_KEY"),
                api_secret=os.getenv("CLOUDINARY_API_SECRET"),
                secure=True,
            )
        ext = Path(upload.filename or "").suffix.lower()
        file_obj = upload.file
        # HEIC from iOS needs converting to JPEG before Cloudinary accepts it.
        if ext in ('.heic', '.heif'):
            from io import BytesIO
            img = Image.open(upload.file)
            buf = BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=90)
            buf.seek(0)
            file_obj = buf
        result = cloudinary.uploader.upload(
            file_obj,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        return result.get("secure_url") or result.get("url")
    except Exception as e:
        print(f"[UPLOAD] Cloudinary failed, falling back to local: {e}", flush=True)
        return None


def save_uploaded_image(upload: UploadFile, prefix: str, folder: str = "tissu/uploads") -> str:
    """Persist an uploaded image and return a URL the frontend can render.

    Prefers Cloudinary when credentials are configured (prod / Railway) so
    images survive deploys; falls back to the local ``static/products``
    directory for dev environments without Cloudinary set up. The optional
    ``folder`` arg routes Cloudinary uploads into a sibling folder (e.g.
    ``tissu/gallery`` for lookbook photos).
    """
    filename_base = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if _cloudinary_configured():
        url = _upload_to_cloudinary(upload, public_id=filename_base, folder=folder)
        if url:
            return url
        # Rewind the file pointer so the local fallback still works when
        # Cloudinary rejected the upload mid-stream.
        try:
            upload.file.seek(0)
        except Exception:
            pass

    save_dir = Path(__file__).parent / "static" / "products"
    save_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(upload.filename or "").suffix.lower()
    if ext in ('.heic', '.heif'):
        img = Image.open(upload.file)
        filename = f"{filename_base}.jpg"
        img.convert("RGB").save(save_dir / filename, "JPEG", quality=85)
    else:
        filename = f"{filename_base}{ext or '.jpg'}"
        with open(save_dir / filename, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    return f"/static/products/{filename}"


# ── Categories registry ────────────────────────────────────

@app.get("/api/categories")
async def list_categories(tenant_id: str = Depends(get_tenant_id)):
    """Return every catalog category with its field schema and live count.
    Driven by the `categories` table — new rows show up in the admin
    sidebar the moment they're inserted."""
    pool = await get_db()
    rows = await pool.fetch("""
        SELECT c.slug, c.name, c.name_en, c.emoji, c.fields, c.sort_order,
               COALESCE(cnt.n, 0) AS count
        FROM categories c
        LEFT JOIN (
            SELECT category, COUNT(*) AS n FROM inventory
            WHERE tenant_id = $1
            GROUP BY category
        ) cnt ON cnt.category = c.slug
        WHERE c.tenant_id = $1
        ORDER BY c.sort_order ASC, c.name ASC
    """, tenant_id)
    out = []
    for r in rows:
        fields = r["fields"]
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except Exception:
                fields = []
        out.append({
            "slug": r["slug"],
            "name": r["name"],
            "name_ka": r["name"],
            "name_en": r["name_en"] or "",
            "emoji": r["emoji"],
            "fields": fields,
            "sort_order": r["sort_order"],
            "count": int(r["count"] or 0),
        })
    return {"categories": out}


@app.post("/api/categories")
async def add_category(request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Register a new product category. Body: {slug, name, emoji, fields}.
    `fields` is a list of {key, label} objects describing the per-product
    custom attributes the admin UI should render."""
    import re as _re
    data = await request.json()
    slug = (data.get("slug") or "").strip().lower()
    # Accept either name (legacy) or name_ka (new bilingual API).
    name = (data.get("name_ka") or data.get("name") or "").strip()
    name_en = (data.get("name_en") or "").strip()
    emoji = (data.get("emoji") or "📦").strip()[:8]
    fields = data.get("fields") or []
    if not slug or not name:
        raise HTTPException(status_code=400, detail="slug და name აუცილებელია")
    if not _re.match(r"^[a-z][a-z0-9_-]{1,31}$", slug):
        raise HTTPException(status_code=400, detail="slug არ ვარგა — მხოლოდ a-z, 0-9, _ ან -")
    cleaned_fields = []
    for f in fields if isinstance(fields, list) else []:
        if isinstance(f, dict) and f.get("key") and f.get("label"):
            cleaned_fields.append({"key": str(f["key"]).strip(), "label": str(f["label"]).strip()})
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    try:
        await pool.execute(
            """INSERT INTO categories (slug, name, name_en, emoji, fields, sort_order, tenant_id, created_at)
               VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)""",
            slug, name, name_en, emoji, json.dumps(cleaned_fields), 100, tenant_id, now,
        )
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=409, detail="ეს slug უკვე არსებობს")
    return {"ok": True, "slug": slug}


@app.delete("/api/categories/{slug}")
async def delete_category(slug: str, tenant_id: str = Depends(get_tenant_id)):
    """Delete a category — refuses when any inventory row still uses it,
    so the owner doesn't orphan products by accident."""
    if slug in ("bag", "necklace"):
        raise HTTPException(status_code=400, detail="ძირითადი კატეგორიები არ წაიშლება")
    pool = await get_db()
    in_use = await pool.fetchval(
        "SELECT COUNT(*) FROM inventory WHERE tenant_id = $1 AND category = $2",
        tenant_id, slug,
    )
    if int(in_use or 0) > 0:
        raise HTTPException(status_code=409, detail=f"ამ კატეგორიაში {in_use} პროდუქტია, ჯერ გადაიტანე სხვაგან")
    await pool.execute(
        "DELETE FROM categories WHERE slug = $1 AND tenant_id = $2",
        slug, tenant_id,
    )
    return {"ok": True}


@app.put("/api/categories/{slug}")
async def update_category(slug: str, request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Update name / emoji / fields, and optionally rename the slug.

    Slug rename is a two-table operation: the categories row itself plus
    every inventory row that points at the old slug. Done in one
    transaction so the catalog never goes through a state where products
    point at a non-existent category.
    """
    import re as _re
    data = await request.json()
    pool = await get_db()

    new_slug = data.get("slug")
    if new_slug is not None:
        new_slug = str(new_slug).strip().lower()
        if not _re.match(r"^[a-z0-9_-]{1,40}$", new_slug):
            raise HTTPException(
                status_code=400,
                detail="slug მხოლოდ a-z, 0-9, _ და - სიმბოლოები (1-40 ცალი)",
            )

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Confirm the row exists for this tenant before touching anything.
            current = await conn.fetchrow(
                "SELECT slug FROM categories WHERE slug = $1 AND tenant_id = $2",
                slug, tenant_id,
            )
            if not current:
                raise HTTPException(status_code=404, detail="კატეგორია ვერ მოიძებნა")

            # Slug rename — must come first so the subsequent UPDATE on
            # name/emoji/fields targets the new key.
            effective_slug = slug
            if new_slug and new_slug != slug:
                clash = await conn.fetchval(
                    "SELECT 1 FROM categories WHERE slug = $1 AND tenant_id = $2",
                    new_slug, tenant_id,
                )
                if clash:
                    raise HTTPException(
                        status_code=409,
                        detail=f"slug '{new_slug}' უკვე გამოყენებულია",
                    )
                await conn.execute(
                    "UPDATE categories SET slug = $1 WHERE slug = $2 AND tenant_id = $3",
                    new_slug, slug, tenant_id,
                )
                await conn.execute(
                    "UPDATE inventory SET category = $1 "
                    "WHERE category = $2 AND tenant_id = $3",
                    new_slug, slug, tenant_id,
                )
                effective_slug = new_slug

            updates = []
            params: list = []
            idx = 1
            # name_ka is the bilingual alias for the legacy `name` column.
            if "name_ka" in data and data["name_ka"] is not None and "name" not in data:
                data["name"] = data["name_ka"]
            for key in ("name", "name_en", "emoji"):
                if key in data and data[key] is not None:
                    updates.append(f"{key} = ${idx}")
                    params.append(str(data[key]).strip())
                    idx += 1
            if "fields" in data and isinstance(data["fields"], list):
                cleaned = [
                    {"key": str(f["key"]).strip(), "label": str(f["label"]).strip()}
                    for f in data["fields"]
                    if isinstance(f, dict) and f.get("key") and f.get("label")
                ]
                updates.append(f"fields = ${idx}::jsonb")
                params.append(json.dumps(cleaned))
                idx += 1
            if updates:
                params.append(effective_slug)
                params.append(tenant_id)
                await conn.execute(
                    f"UPDATE categories SET {', '.join(updates)} "
                    f"WHERE slug = ${idx} AND tenant_id = ${idx + 1}",
                    *params,
                )

    return {"ok": True, "slug": effective_slug}


# ── Promo codes ──────────────────────────────────────────────────────────────
# Simple flat list — code + description + percent. Bot integration can read
# from this same table later (e.g. "give 10% off if customer types code X").

def _serialize_promo(row: dict) -> dict:
    return {
        "id": row["id"],
        "code": row["code"],
        "description": row.get("description") or "",
        "discount_percent": row["discount_percent"],
        "active": bool(row["active"]),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


@app.get("/api/promo-codes")
async def api_list_promo_codes(tenant_id: str = Depends(get_tenant_id)):
    rows = await list_promo_codes(tenant_id)
    return {"promo_codes": [_serialize_promo(r) for r in rows]}


@app.post("/api/promo-codes")
async def api_create_promo_code(request: Request, tenant_id: str = Depends(get_tenant_id)):
    data = await request.json()
    code = (data.get("code") or "").strip()
    description = (data.get("description") or "").strip()
    try:
        percent = int(data.get("discount_percent") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="discount_percent უნდა იყოს რიცხვი")
    active = bool(data.get("active", True))
    if not code:
        raise HTTPException(status_code=400, detail="კოდი აუცილებელია")
    if not (1 <= percent <= 100):
        raise HTTPException(status_code=400, detail="პროცენტი უნდა იყოს 1-დან 100-მდე")
    try:
        row = await create_promo_code(tenant_id, code, description, percent, active)
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=409, detail="ეს კოდი უკვე არსებობს")
    return _serialize_promo(row)


@app.put("/api/promo-codes/{promo_id}")
async def api_update_promo_code(
    promo_id: int, request: Request, tenant_id: str = Depends(get_tenant_id),
):
    data = await request.json()
    kwargs: dict = {}
    if "code" in data:
        if not (data["code"] or "").strip():
            raise HTTPException(status_code=400, detail="კოდი აუცილებელია")
        kwargs["code"] = data["code"]
    if "description" in data:
        kwargs["description"] = data["description"] or ""
    if "discount_percent" in data:
        try:
            percent = int(data["discount_percent"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="discount_percent უნდა იყოს რიცხვი")
        if not (1 <= percent <= 100):
            raise HTTPException(status_code=400, detail="პროცენტი უნდა იყოს 1-დან 100-მდე")
        kwargs["discount_percent"] = percent
    if "active" in data:
        kwargs["active"] = bool(data["active"])
    try:
        row = await update_promo_code(tenant_id, promo_id, **kwargs)
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=409, detail="ეს კოდი უკვე არსებობს")
    if not row:
        raise HTTPException(status_code=404, detail="პრომოკოდი ვერ მოიძებნა")
    return _serialize_promo(row)


@app.delete("/api/promo-codes/{promo_id}")
async def api_delete_promo_code(promo_id: int, tenant_id: str = Depends(get_tenant_id)):
    ok = await delete_promo_code(tenant_id, promo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="პრომოკოდი ვერ მოიძებნა")
    return {"ok": True}


# ── Necklace options (fabrics + charms) ──────────────────────────────────────
# Each option is an image + display name. Customer picks one fabric and
# one charm when ordering a necklace. Same table for both; `kind` discriminates.

def _serialize_necklace_option(row: dict) -> dict:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row.get("name") or "",
        "image_url": row["image_url"],
        "active": bool(row["active"]),
        "sort_order": row.get("sort_order") or 0,
        "extra_price": float(row.get("extra_price") or 0),
        "variants": row.get("variants") or [],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


@app.get("/api/necklace-options")
async def api_list_necklace_options(
    kind: str = "",
    tenant_id: str = Depends(get_tenant_id),
):
    """Admin list — returns both active and inactive so the operator can
    re-enable disabled rows from the UI."""
    kind_arg = kind if kind in ("fabric", "charm") else None
    rows = await list_necklace_options(tenant_id, kind=kind_arg, include_inactive=True)
    return {"options": [_serialize_necklace_option(r) for r in rows]}


@app.post("/api/necklace-options")
async def api_create_necklace_option(
    kind: str = Form(...),
    name: str = Form(""),
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    if kind not in ("fabric", "charm"):
        raise HTTPException(status_code=400, detail="kind უნდა იყოს fabric ან charm")
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    image_url = save_uploaded_image(image, prefix=f"necklace_{kind}")
    row = await create_necklace_option(tenant_id, kind, name, image_url)
    return _serialize_necklace_option(row)


@app.put("/api/necklace-options/{option_id}")
async def api_update_necklace_option(
    option_id: int, request: Request, tenant_id: str = Depends(get_tenant_id),
):
    data = await request.json()
    kwargs: dict = {}
    if "name" in data:
        kwargs["name"] = data["name"] or ""
    if "active" in data:
        kwargs["active"] = bool(data["active"])
    if "sort_order" in data:
        try:
            kwargs["sort_order"] = int(data["sort_order"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="sort_order უნდა იყოს რიცხვი")
    if "extra_price" in data:
        try:
            ep = float(data["extra_price"] or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="extra_price უნდა იყოს რიცხვი")
        if ep < 0:
            raise HTTPException(status_code=400, detail="extra_price დადებითი უნდა იყოს")
        kwargs["extra_price"] = ep
    if "variants" in data:
        v = data["variants"]
        if v is None:
            v = []
        if not isinstance(v, list):
            raise HTTPException(status_code=400, detail="variants უნდა იყოს მასივი")
        kwargs["variants"] = v
    if "image_url" in data and data["image_url"]:
        kwargs["image_url"] = str(data["image_url"]).strip()
    row = await update_necklace_option(tenant_id, option_id, **kwargs)
    if not row:
        raise HTTPException(status_code=404, detail="ვარიანტი ვერ მოიძებნა")
    return _serialize_necklace_option(row)


@app.post("/api/necklace-options/{option_id}/image")
async def api_replace_necklace_option_image(
    option_id: int,
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Swap the photo of an existing option (admin can re-upload without
    deleting/recreating)."""
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    image_url = save_uploaded_image(image, prefix=f"necklace_{option_id}")
    row = await update_necklace_option(tenant_id, option_id, image_url=image_url)
    if not row:
        raise HTTPException(status_code=404, detail="ვარიანტი ვერ მოიძებნა")
    return _serialize_necklace_option(row)


@app.delete("/api/necklace-options/{option_id}")
async def api_delete_necklace_option(option_id: int, tenant_id: str = Depends(get_tenant_id)):
    ok = await delete_necklace_option(tenant_id, option_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ვარიანტი ვერ მოიძებნა")
    return {"ok": True}


@app.get("/api/necklace-base-price")
async def api_get_necklace_base_price(tenant_id: str = Depends(get_tenant_id)):
    return {"base_price": await get_necklace_base_price(tenant_id)}


@app.put("/api/necklace-base-price")
async def api_set_necklace_base_price(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    data = await request.json()
    try:
        price = float(data.get("base_price"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="base_price უნდა იყოს რიცხვი")
    if price < 0:
        raise HTTPException(status_code=400, detail="base_price დადებითი უნდა იყოს")
    saved = await set_necklace_base_price(tenant_id, price)
    return {"base_price": saved}


# ── Lookbook gallery (admin) ─────────────────────────────────
# Lifestyle photos shown on the website, kept out of the product
# catalog so the Messenger bot never sees them.

def _serialize_gallery(row: dict) -> dict:
    return {
        "id": row["id"],
        "image_url": row["image_url"],
        "caption": row.get("caption") or "",
        "position": int(row.get("position") or 0),
        "width": row.get("width"),
        "height": row.get("height"),
        "created_at": row.get("created_at"),
    }


def _read_image_dimensions(upload: UploadFile):
    """Best-effort PIL probe of an uploaded image. Rewinds the file
    pointer so the subsequent upload still gets the full bytes. Returns
    (None, None) on any failure — width/height are optional in the DB."""
    try:
        upload.file.seek(0)
        with Image.open(upload.file) as img:
            w, h = img.size
        return int(w), int(h)
    except Exception:
        return None, None
    finally:
        try:
            upload.file.seek(0)
        except Exception:
            pass


@app.get("/api/admin/gallery")
async def api_list_gallery(tenant_id: str = Depends(get_tenant_id)):
    """Admin list — every gallery photo for the current tenant."""
    rows = await list_gallery_photos(tenant_id)
    return {"photos": [_serialize_gallery(r) for r in rows]}


@app.post("/api/admin/gallery")
async def api_create_gallery(
    image: UploadFile = File(...),
    caption: str = Form(""),
    tenant_id: str = Depends(get_tenant_id),
):
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    width, height = _read_image_dimensions(image)
    image_url = save_uploaded_image(image, prefix="gallery", folder="tissu/gallery")
    row = await create_gallery_photo(
        tenant_id, image_url, caption=caption,
        width=width, height=height,
    )
    return _serialize_gallery(row)


@app.put("/api/admin/gallery/{photo_id}")
async def api_update_gallery(
    photo_id: int, request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Patch caption and/or position. Send only the fields you want to change."""
    data = await request.json()
    kwargs: dict = {}
    if "caption" in data:
        kwargs["caption"] = data["caption"] or ""
    if "position" in data:
        try:
            kwargs["position"] = int(data["position"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position უნდა იყოს რიცხვი")
    row = await update_gallery_photo(tenant_id, photo_id, **kwargs)
    if not row:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return _serialize_gallery(row)


@app.post("/api/admin/gallery/{photo_id}/image")
async def api_replace_gallery_image(
    photo_id: int,
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Swap the photo of an existing gallery row without changing its
    position or caption."""
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    width, height = _read_image_dimensions(image)
    image_url = save_uploaded_image(image, prefix=f"gallery_{photo_id}", folder="tissu/gallery")
    row = await update_gallery_photo(
        tenant_id, photo_id, image_url=image_url, width=width, height=height,
    )
    if not row:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return _serialize_gallery(row)


@app.delete("/api/admin/gallery/{photo_id}")
async def api_delete_gallery(photo_id: int, tenant_id: str = Depends(get_tenant_id)):
    ok = await delete_gallery_photo(tenant_id, photo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return {"ok": True}


# ── Per-product lookbook (storefront product page + Pinterest) ────

def _serialize_product_gallery(row: dict) -> dict:
    return {
        "id": row["id"],
        "inventory_id": row["inventory_id"],
        "image_url": row["image_url"],
        "position": int(row.get("position") or 0),
        "created_at": row.get("created_at"),
    }


@app.get("/api/product-gallery")
async def api_list_product_gallery(
    inventory_id: int = 0,
    tenant_id: str = Depends(get_tenant_id),
):
    """List per-product lookbook photos. Pass ``?inventory_id=N`` to scope
    to a single product; omit it to get every row for this tenant."""
    inv_id = inventory_id if inventory_id > 0 else None
    rows = await list_product_gallery(tenant_id, inventory_id=inv_id)
    return {"photos": [_serialize_product_gallery(r) for r in rows]}


@app.post("/api/product-gallery")
async def api_create_product_gallery(
    inventory_id: int = Form(...),
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    if inventory_id <= 0:
        raise HTTPException(status_code=400, detail="inventory_id აუცილებელია")
    # Confirm the inventory row belongs to this tenant — prevents an
    # attacker with one tenant's key from attaching photos to another
    # tenant's product.
    pool = await get_db()
    owns = await pool.fetchval(
        "SELECT 1 FROM inventory WHERE id = $1 AND tenant_id = $2",
        inventory_id, tenant_id,
    )
    if not owns:
        raise HTTPException(status_code=404, detail="პროდუქტი ვერ მოიძებნა")
    image_url = save_uploaded_image(
        image, prefix=f"lookbook_{inventory_id}",
        folder=f"tissu/lookbook/{inventory_id}",
    )
    row = await create_product_gallery_photo(tenant_id, inventory_id, image_url)
    return _serialize_product_gallery(row)


@app.put("/api/product-gallery/{photo_id}")
async def api_update_product_gallery(
    photo_id: int, request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Patch position (used for reorder). Send only the fields you want."""
    data = await request.json()
    kwargs: dict = {}
    if "position" in data:
        try:
            kwargs["position"] = int(data["position"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position უნდა იყოს რიცხვი")
    row = await update_product_gallery_photo(tenant_id, photo_id, **kwargs)
    if not row:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return _serialize_product_gallery(row)


@app.delete("/api/product-gallery/{photo_id}")
async def api_delete_product_gallery(
    photo_id: int, tenant_id: str = Depends(get_tenant_id),
):
    ok = await delete_product_gallery_photo(tenant_id, photo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return {"ok": True}


# ── Site orders (storefront checkout) ────────────────────────
# The website posts new orders here; the operator manages them from the
# admin panel. Distinct from the bot's /api/orders endpoints.

def _new_order_id() -> str:
    """Random URL-safe id, matches the cuid-ish ids the site already uses."""
    import secrets as _s
    return "ord_" + _s.token_urlsafe(12)


def _new_review_id() -> str:
    import secrets as _s
    return "rev_" + _s.token_urlsafe(10)


@app.post("/api/admin/orders")
async def api_create_site_order(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Create a new order from the storefront. Body shape mirrors the
    site's checkout payload (see CLAUDE.md for the schema)."""
    data = await request.json()
    items = data.get("items") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="items აუცილებელია")
    name = (data.get("customer_name") or "").strip()
    phone = (data.get("customer_phone") or "").strip()
    if not name or not phone:
        raise HTTPException(
            status_code=400, detail="customer_name და customer_phone აუცილებელია",
        )
    status = (data.get("status") or "pending_confirmation").strip()
    if status not in ORDER_STATUSES:
        raise HTTPException(status_code=400, detail=f"არასწორი status: {status}")
    order = await create_site_order(
        tenant_id, _new_order_id(),
        customer_name=name,
        customer_phone=phone,
        address_street=(data.get("address_street") or "").strip(),
        address_city=(data.get("address_city") or "").strip(),
        items=items,
        status=status,
        subtotal=float(data.get("subtotal") or 0),
        shipping=float(data.get("shipping") or 0),
        discount=float(data.get("discount") or 0),
        total=float(data.get("total") or 0),
        payment_method=(data.get("payment_method") or "undecided"),
        contact_method=(data.get("contact_method") or "phone"),
        customer_email=(data.get("customer_email") or None),
        address_zone=data.get("address_zone") or {},
        notes=(data.get("notes") or None),
    )
    # Best-effort confirmation/invoice email (offloaded so a slow SMTP server
    # can't hold up the checkout response). No-op until SMTP_* env vars are set.
    await asyncio.to_thread(_send_order_emails, order)
    return order


@app.get("/api/admin/orders")
async def api_list_site_orders(
    status: str = "", limit: int = 200,
    tenant_id: str = Depends(get_tenant_id),
):
    """List orders, optionally filtered by status."""
    status_arg = status.strip() or None
    if status_arg and status_arg not in ORDER_STATUSES:
        raise HTTPException(status_code=400, detail=f"არასწორი status: {status_arg}")
    rows = await list_site_orders(tenant_id, status=status_arg, limit=limit)
    return {"orders": rows}


@app.get("/api/admin/orders/{order_id}")
async def api_get_site_order(
    order_id: str, tenant_id: str = Depends(get_tenant_id),
):
    order = await get_site_order(tenant_id, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="შეკვეთა ვერ მოიძებნა")
    return order


@app.patch("/api/admin/orders/{order_id}")
async def api_patch_site_order(
    order_id: str, request: Request,
    tenant_id: str = Depends(get_tenant_id),
):
    """Change order status. Stock is debited / refunded when crossing the
    reserved boundary (see update_site_order_status)."""
    data = await request.json()
    new_status = (data.get("status") or "").strip()
    if not new_status:
        raise HTTPException(status_code=400, detail="status აუცილებელია")
    if new_status not in ORDER_STATUSES:
        raise HTTPException(status_code=400, detail=f"არასწორი status: {new_status}")
    order = await update_site_order_status(tenant_id, order_id, new_status)
    if not order:
        raise HTTPException(status_code=404, detail="შეკვეთა ვერ მოიძებნა")
    return order


# ── Reviews ──────────────────────────────────────────────────

def _serialize_review(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row.get("name") or "",
        "comment": row.get("comment") or "",
        "photo_url": row.get("photo_url"),
        "product_id": row.get("product_id"),
        "position": int(row.get("position") or 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


@app.get("/api/admin/reviews")
async def api_list_reviews(tenant_id: str = Depends(get_tenant_id)):
    rows = await list_reviews(tenant_id)
    return {"reviews": [_serialize_review(r) for r in rows]}


@app.post("/api/admin/reviews")
async def api_create_review(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Create a review. JSON body — photo_url is optional; for direct
    upload use POST /api/admin/reviews/upload instead."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    comment = (data.get("comment") or "").strip()
    if not comment:
        raise HTTPException(status_code=400, detail="comment აუცილებელია")
    row = await create_review(
        tenant_id, _new_review_id(),
        name=name, comment=comment,
        photo_url=(data.get("photo_url") or None),
        product_id=(data.get("product_id") or None),
    )
    return _serialize_review(row)


@app.post("/api/admin/reviews/upload")
async def api_upload_review_photo(
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload a review photo to Cloudinary tissu/reviews/ and return the
    URL — the frontend then posts the URL via /api/admin/reviews."""
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    image_url = save_uploaded_image(image, prefix="review", folder="tissu/reviews")
    return {"image_url": image_url}


@app.post("/api/admin/reviews/{review_id}/photo")
async def api_upload_review_photo_inline(
    review_id: str,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload a photo and attach it to an existing review in one call.
    Mirrors the storefront's expected migration path."""
    if not file:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    image_url = save_uploaded_image(file, prefix=f"review_{review_id}", folder="tissu/reviews")
    row = await update_review(tenant_id, review_id, photo_url=image_url)
    if not row:
        raise HTTPException(status_code=404, detail="რევიუ ვერ მოიძებნა")
    return {"photo_url": image_url, "review": _serialize_review(row)}


@app.post("/api/admin/reviews/bulk-import")
async def api_bulk_import_reviews(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Import an array of reviews in one call, preserving created_at and
    position when present. Idempotent at the row level via a content
    hash check (name + first 60 chars of comment) so re-runs don't
    duplicate. Returns counts so the caller knows what happened."""
    data = await request.json()
    rows = data if isinstance(data, list) else data.get("reviews") or []
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="reviews მასივი აუცილებელია")

    pool = await get_db()
    existing = await pool.fetch(
        "SELECT name, LEFT(comment, 60) AS cprefix FROM reviews WHERE tenant_id = $1",
        tenant_id,
    )
    seen = {(r["name"].strip(), (r["cprefix"] or "").strip()) for r in existing}

    inserted = 0
    skipped = 0
    for item in rows:
        name = (item.get("name") or "").strip()
        comment = (item.get("comment") or "").strip()
        if not name or not comment:
            skipped += 1
            continue
        key = (name, comment[:60].strip())
        if key in seen:
            skipped += 1
            continue
        await create_review(
            tenant_id, _new_review_id(),
            name=name, comment=comment,
            photo_url=(item.get("photo_url") or None),
            product_id=(str(item.get("product_id")) if item.get("product_id") is not None else None),
            position=(int(item["position"]) if item.get("position") is not None else None),
            created_at=item.get("created_at"),
        )
        seen.add(key)
        inserted += 1
    return {"inserted": inserted, "skipped": skipped, "total": len(rows)}


@app.patch("/api/admin/reviews/{review_id}")
async def api_update_review(
    review_id: str, request: Request,
    tenant_id: str = Depends(get_tenant_id),
):
    data = await request.json()
    kwargs: dict = {}
    for key in ("name", "comment", "photo_url", "product_id"):
        if key in data:
            kwargs[key] = data[key]
    if "position" in data:
        try:
            kwargs["position"] = int(data["position"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position უნდა იყოს რიცხვი")
    row = await update_review(tenant_id, review_id, **kwargs)
    if not row:
        raise HTTPException(status_code=404, detail="რევიუ ვერ მოიძებნა")
    return _serialize_review(row)


@app.put("/api/admin/reviews/order")
async def api_reorder_reviews(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Apply a new ordering. Body: { ids: ["rev_aaa", "rev_bbb", ...] }"""
    data = await request.json()
    ids = data.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise HTTPException(status_code=400, detail="ids უნდა იყოს string მასივი")
    updated = await reorder_reviews(tenant_id, ids)
    return {"updated": updated}


@app.delete("/api/admin/reviews/{review_id}")
async def api_delete_review(
    review_id: str, tenant_id: str = Depends(get_tenant_id),
):
    ok = await delete_review(tenant_id, review_id)
    if not ok:
        raise HTTPException(status_code=404, detail="რევიუ ვერ მოიძებნა")
    return {"ok": True}


# ── Size variant links (small + big of the same design) ─────

@app.get("/api/admin/size-variants")
async def api_list_size_variants(tenant_id: str = Depends(get_tenant_id)):
    """All pairs for the operator UI. Each entry already includes
    enough product detail (code, image, price) to render the row."""
    return {"links": await list_size_variants(tenant_id)}


@app.post("/api/admin/size-variants")
async def api_create_size_variant(
    request: Request, tenant_id: str = Depends(get_tenant_id),
):
    """Link a small and a big inventory row. Body: {small_id, big_id}."""
    data = await request.json()
    try:
        small_id = int(data.get("small_id"))
        big_id = int(data.get("big_id"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="small_id და big_id რიცხვები უნდა იყოს")
    try:
        result = await create_size_variant(tenant_id, small_id, big_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.delete("/api/admin/size-variants/{link_id}")
async def api_delete_size_variant(
    link_id: int, tenant_id: str = Depends(get_tenant_id),
):
    ok = await delete_size_variant(tenant_id, link_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ლინკი ვერ მოიძებნა")
    return {"ok": True}


# ── Inspiration photos (Pinterest agent references) ─────────
# Operator-uploaded reference images the AI-content agent pulls as
# visual inspiration when generating new designs. NEVER rendered on
# the public storefront — admin uploads, media-scope reads.

def _serialize_inspiration(row: dict) -> dict:
    return {
        "id": row["id"],
        "category": row.get("category") or "necklace",
        "image_url": row["image_url"],
        "caption": row.get("caption") or "",
        "position": int(row.get("position") or 0),
        "created_at": row.get("created_at"),
    }


@app.get("/api/admin/inspirations")
async def api_list_inspirations_admin(
    category: str = "",
    tenant_id: str = Depends(get_tenant_id),
):
    cat = category.strip() or None
    rows = await list_inspiration_photos(tenant_id, category=cat)
    return {"photos": [_serialize_inspiration(r) for r in rows]}


@app.post("/api/admin/inspirations")
async def api_create_inspiration(
    image: UploadFile = File(...),
    category: str = Form("necklace"),
    caption: str = Form(""),
    tenant_id: str = Depends(get_tenant_id),
):
    if not image:
        raise HTTPException(status_code=400, detail="ფოტო აუცილებელია")
    cat = (category or "necklace").strip() or "necklace"
    image_url = save_uploaded_image(
        image, prefix=f"inspiration_{cat}", folder=f"tissu/inspirations/{cat}",
    )
    row = await create_inspiration_photo(
        tenant_id, image_url, category=cat, caption=caption,
    )
    return _serialize_inspiration(row)


@app.patch("/api/admin/inspirations/{photo_id}")
async def api_update_inspiration(
    photo_id: int, request: Request,
    tenant_id: str = Depends(get_tenant_id),
):
    data = await request.json()
    kwargs: dict = {}
    if "caption" in data:
        kwargs["caption"] = data["caption"] or ""
    if "position" in data:
        try:
            kwargs["position"] = int(data["position"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position უნდა იყოს რიცხვი")
    row = await update_inspiration_photo(tenant_id, photo_id, **kwargs)
    if not row:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return _serialize_inspiration(row)


@app.delete("/api/admin/inspirations/{photo_id}")
async def api_delete_inspiration(
    photo_id: int, tenant_id: str = Depends(get_tenant_id),
):
    ok = await delete_inspiration_photo(tenant_id, photo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ფოტო ვერ მოიძებნა")
    return {"ok": True}


@app.get("/api/inspirations")
async def api_list_inspirations_for_agent(
    request: Request,
    category: str = "",
    tenant_id: str = Depends(get_tenant_id),
):
    """Pinterest agent read endpoint — returns the same shape the admin
    list does but lives at a media-scope-allowed path."""
    cat = category.strip() or None
    rows = await list_inspiration_photos(tenant_id, category=cat)
    return {"photos": [_serialize_inspiration(r) for r in rows]}


@app.put("/api/inventory/{item_id}/attr")
async def set_inventory_attr(item_id: int, request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Merge a single key/value into inventory.attrs — used by admin to
    save the per-field text inputs for category-driven products."""
    data = await request.json()
    key = (data.get("key") or "").strip()
    value = data.get("value", "")
    if not key:
        raise HTTPException(status_code=400, detail="key required")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE inventory SET attrs = attrs || jsonb_build_object($1::text, $2::text), updated_at = $3 "
        "WHERE id = $4 AND tenant_id = $5",
        key, str(value), now, item_id, tenant_id,
    )
    return {"ok": True}


@app.get("/api/inventory")
async def list_inventory(category: str = "", tenant_id: str = Depends(get_tenant_id)):
    """List inventory. Omit category to get everything (admin); pass
    `?category=bag` or `?category=necklace` to scope results — bot-facing
    callers should always pass category='bag' to avoid cross-category leakage.

    Each row gets an `angles` field with the product's extra angle photos
    (rows from product_extra_photos where photo_type='angle')."""
    pool = await get_db()
    # `id ASC` as the final tiebreaker keeps the order stable: products
    # with the same model + size used to swap positions every time the
    # operator uploaded or deleted a photo, because Postgres has no
    # implicit row order. Sorting by id last freezes the relative order
    # within each (model, size) group to creation order.
    if category:
        rows = await pool.fetch(
            "SELECT * FROM inventory WHERE tenant_id = $1 AND category = $2 "
            "ORDER BY model, size, id",
            tenant_id, category,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM inventory WHERE tenant_id = $1 "
            "ORDER BY model, size, id",
            tenant_id,
        )

    inv_ids = [r["id"] for r in rows]
    angles_by_id: dict = {}
    if inv_ids:
        try:
            angle_rows = await pool.fetch(
                "SELECT id, inventory_id, image_url FROM product_extra_photos "
                "WHERE tenant_id = $1 AND photo_type = 'angle' "
                "AND inventory_id = ANY($2::int[]) ORDER BY created_at ASC",
                tenant_id, inv_ids,
            )
            for ar in angle_rows:
                angles_by_id.setdefault(ar["inventory_id"], []).append(
                    {"id": ar["id"], "image_url": ar["image_url"]}
                )
        except Exception as e:
            # Table might not exist yet on very old schemas — degrade to no angles.
            print(f"[inventory] angles query skipped: {e}", flush=True)

    out = []
    for r in rows:
        d = dict(r)
        d["angles"] = angles_by_id.get(r["id"], [])
        out.append(d)
    return {"inventory": out}


@app.post("/api/inventory/{item_id}/angle")
async def add_product_angle(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    """Upload an extra angle photo for a product — goes into
    product_extra_photos with photo_type='angle' so it lives alongside the
    existing lifestyle photos but is clearly distinguishable. Returns the
    new row's id + image_url for the frontend to insert inline."""
    image_url = save_uploaded_image(image, f"angle_{item_id}")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "SELECT code FROM inventory WHERE id = $1 AND tenant_id = $2",
        item_id, tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    code = row["code"] or ""
    new = await pool.fetchrow(
        "INSERT INTO product_extra_photos (inventory_id, code, image_url, photo_type, tenant_id, created_at) "
        "VALUES ($1, $2, $3, 'angle', $4, $5) RETURNING id",
        item_id, code, image_url, tenant_id, now,
    )
    return {"id": new["id"], "image_url": image_url}


@app.delete("/api/inventory/angle/{photo_id}")
async def delete_product_angle(photo_id: int, tenant_id: str = Depends(get_tenant_id)):
    """Remove a single angle photo by its extra-photos row id."""
    pool = await get_db()
    await pool.execute(
        "DELETE FROM product_extra_photos "
        "WHERE id = $1 AND tenant_id = $2 AND photo_type = 'angle'",
        photo_id, tenant_id,
    )
    return {"ok": True}


@app.post("/api/inventory")
async def add_inventory(
    product_name: str = Form(...),
    model: str = Form(""),
    size: str = Form(""),
    price: float = Form(...),
    stock: int = Form(...),
    color: str = Form(""),
    style: str = Form(""),
    category: str = Form("bag"),
    attrs_json: str = Form("{}"),
    image: UploadFile = File(None),
    tenant_id: str = Depends(get_tenant_id),
):
    image_url = ""
    if image and image.filename:
        prefix = f"{model}_{size}_{color}".replace(" ", "_")
        image_url = save_uploaded_image(image, prefix)

    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Auto-generate a code for non-bag categories using the first two
    # uppercase letters of the slug (e.g. necklace → NE, belt → BE). Bag
    # codes stay on the legacy FP/TP/FD/TD convention and are not
    # generated here. Codes are unique per tenant.
    new_code = ""
    if category and category != "bag":
        prefix = (category[:2] or "XX").upper()
        max_n = await pool.fetchval(
            "SELECT COALESCE(MAX(CAST(SUBSTRING(code, 3) AS INTEGER)), 0) "
            "FROM inventory WHERE tenant_id = $1 AND category = $2 "
            "AND code ~ ('^' || $3 || '[0-9]+$')",
            tenant_id, category, prefix,
        )
        new_code = f"{prefix}{int(max_n or 0) + 1}"

    try:
        attrs_value = attrs_json if attrs_json else "{}"
        json.loads(attrs_value)  # validate
    except Exception:
        attrs_value = "{}"

    row = await pool.fetchrow(
        "INSERT INTO inventory (product_name, model, size, color, style, code, category, attrs, price, stock, image_url, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13, $14) RETURNING id",
        product_name, model, size, color, style, new_code, category, attrs_value, price, stock, image_url, tenant_id, now, now,
    )
    return {"id": row["id"], "code": new_code, "category": category, "image_url": image_url, "message": "Product added"}


@app.put("/api/inventory/{item_id}")
async def update_inventory(
    item_id: int,
    stock: int = None, price: float = None, model: str = None,
    size: str = None, color: str = None, tags: str = None,
    on_sale: bool = None, sale_price: float = None,
    description: str = None, product_name: str = None,
    product_name_en: str = None,
    tenant_id: str = Depends(get_tenant_id),
):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    if stock is not None:
        await pool.execute("UPDATE inventory SET stock = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", stock, now, item_id, tenant_id)
    if price is not None:
        await pool.execute("UPDATE inventory SET price = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", price, now, item_id, tenant_id)
    if model is not None:
        await pool.execute("UPDATE inventory SET model = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", model, now, item_id, tenant_id)
    if size is not None:
        new_price = 74 if "დიდი" in size else 69
        await pool.execute("UPDATE inventory SET size = $1, price = $2, updated_at = $3 WHERE id = $4 AND tenant_id = $5", size, new_price, now, item_id, tenant_id)
    if color is not None:
        await pool.execute("UPDATE inventory SET color = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", color, now, item_id, tenant_id)
    if tags is not None:
        await pool.execute("UPDATE inventory SET tags = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", tags, now, item_id, tenant_id)
    if on_sale is not None:
        await pool.execute("UPDATE inventory SET on_sale = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", on_sale, now, item_id, tenant_id)
    if sale_price is not None:
        # Passing a literal 0 or negative clears the sale price back to null.
        sp = sale_price if sale_price > 0 else None
        await pool.execute("UPDATE inventory SET sale_price = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", sp, now, item_id, tenant_id)
    if description is not None:
        # Empty string clears the description back to NULL so the
        # storefront treats it as "no description" rather than blank.
        cleaned = description.strip() or None
        await pool.execute("UPDATE inventory SET description = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", cleaned, now, item_id, tenant_id)
    if product_name is not None:
        # Empty string clears the override → the storefront falls back
        # to model + size (see _serialize in src/api/storefront.py).
        cleaned = product_name.strip() or ""
        await pool.execute("UPDATE inventory SET product_name = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", cleaned, now, item_id, tenant_id)
    if product_name_en is not None:
        cleaned = product_name_en.strip() or ""
        await pool.execute("UPDATE inventory SET product_name_en = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", cleaned, now, item_id, tenant_id)
    return {"message": f"Item #{item_id} updated"}


@app.post("/api/inventory/{item_id}/image")
async def upload_product_image(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    image_url = save_uploaded_image(image, f"product_{item_id}_front")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", image_url, now, item_id, tenant_id)
    return {"image_url": image_url}


@app.post("/api/inventory/{item_id}/image_back")
async def upload_back_image(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    image_url = save_uploaded_image(image, f"product_{item_id}_back")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", image_url, now, item_id, tenant_id)
    return {"image_url": image_url}


@app.post("/api/inventory/swap-images")
async def swap_images(request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Swap images between two inventory slots (drag & drop support)."""
    data = await request.json()
    from_id = data["from_id"]
    from_side = data["from_side"]
    to_id = data["to_id"]
    to_side = data["to_side"]

    pool = await get_db()
    r1 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1 AND tenant_id = $2", from_id, tenant_id)
    r2 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1 AND tenant_id = $2", to_id, tenant_id)
    if not r1 or not r2:
        raise HTTPException(status_code=404)

    from_url = r1["image_url"] if from_side == "front" else r1["image_url_back"]
    to_url = r2["image_url"] if to_side == "front" else r2["image_url_back"]

    now = datetime.now(timezone.utc).isoformat()
    from_col = "image_url" if from_side == "front" else "image_url_back"
    to_col = "image_url" if to_side == "front" else "image_url_back"

    await pool.execute(f"UPDATE inventory SET {from_col} = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", to_url, now, from_id, tenant_id)
    await pool.execute(f"UPDATE inventory SET {to_col} = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", from_url, now, to_id, tenant_id)
    return {"message": "Images swapped"}


@app.post("/api/inventory/{item_id}/clear-image")
async def clear_image(item_id: int, side: str = "back", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    col = "image_url" if side == "front" else "image_url_back"
    await pool.execute(f"UPDATE inventory SET {col} = '', updated_at = $1 WHERE id = $2 AND tenant_id = $3", now, item_id, tenant_id)
    return {"message": f"{side} image cleared"}


@app.post("/api/inventory/{item_id}/remove-back")
async def remove_back_image(item_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = '', updated_at = $1 WHERE id = $2 AND tenant_id = $3", now, item_id, tenant_id)
    return {"message": "Back image removed"}


@app.delete("/api/inventory/{item_id}")
async def delete_inventory(item_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute("DELETE FROM inventory WHERE id = $1 AND tenant_id = $2", item_id, tenant_id)
    return {"message": f"Item #{item_id} deleted"}


# ── Orders ───────────────────────────────────────────────────

@app.get("/api/orders")
async def list_orders(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM orders WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"orders": [dict(r) for r in rows]}


@app.put("/api/orders/{order_id}")
async def update_order(order_id: int, request: Request, tenant_id: str = Depends(get_tenant_id)):
    data = await request.json()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    allowed_fields = ("customer_phone", "customer_address", "status", "notes", "items", "total")
    for field in allowed_fields:
        if field in data:
            await pool.execute(
                f"UPDATE orders SET {field} = $1, updated_at = $2 "
                f"WHERE id = $3 AND tenant_id = $4",
                data[field], now, order_id, tenant_id,
            )
    return {"success": True}


@app.delete("/api/orders/{order_id}")
async def delete_order(order_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute(
        "DELETE FROM orders WHERE id = $1 AND tenant_id = $2",
        order_id, tenant_id,
    )
    return {"success": True}


@app.post("/api/orders/{order_id}/decrease-stock")
async def decrease_stock_for_order(order_id: int, tenant_id: str = Depends(get_tenant_id)):
    """When order moves to ready_for_send, decrease stock for the product code."""
    pool = await get_db()
    order = await pool.fetchrow(
        "SELECT items FROM orders WHERE id = $1 AND tenant_id = $2",
        order_id, tenant_id,
    )
    if not order:
        raise HTTPException(status_code=404)
    items_raw = order["items"].strip().upper()
    import re as _re
    code_match = _re.search(r'(FP|TP|FD|TD)\d+', items_raw)
    item_code = code_match.group(0) if code_match else items_raw
    await pool.execute(
        "UPDATE inventory SET stock = GREATEST(0, stock - 1), updated_at = $1 "
        "WHERE tenant_id = $2 AND UPPER(code) = $3 AND stock > 0",
        datetime.now(timezone.utc).isoformat(), tenant_id, item_code,
    )
    return {"success": True, "code": item_code}


# ── Product Pairs (same print, different style) ─────────────

@app.get("/api/pairs")
async def list_pairs(tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT * FROM product_pairs WHERE tenant_id = $1 ORDER BY code_a",
        tenant_id,
    )
    return {"pairs": [dict(r) for r in rows]}


@app.post("/api/pairs")
async def add_pair(request: Request, tenant_id: str = Depends(get_tenant_id)):
    data = await request.json()
    code_a = data["code_a"].upper().strip()
    code_b = data["code_b"].upper().strip()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Find all codes already connected to A or B (transitive group),
    # scoped to this tenant so different shops' pairs can't merge.
    group = set([code_a, code_b])
    to_check = [code_a, code_b]
    while to_check:
        current = to_check.pop(0)
        rows = await pool.fetch(
            "SELECT code_b FROM product_pairs WHERE tenant_id = $1 AND code_a = $2",
            tenant_id, current,
        )
        for r in rows:
            if r["code_b"] not in group:
                group.add(r["code_b"])
                to_check.append(r["code_b"])

    # Connect ALL members of the group to each other
    group = sorted(group)
    for i, a in enumerate(group):
        for b in group[i+1:]:
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, tenant_id, created_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                a, b, tenant_id, now,
            )
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, tenant_id, created_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                b, a, tenant_id, now,
            )

    return {"success": True, "group": group}


@app.delete("/api/pairs/{pair_id}")
async def delete_pair(pair_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT code_a, code_b FROM product_pairs WHERE id = $1 AND tenant_id = $2",
        pair_id, tenant_id,
    )
    if row:
        await pool.execute(
            "DELETE FROM product_pairs WHERE tenant_id = $1 AND "
            "((code_a=$2 AND code_b=$3) OR (code_a=$3 AND code_b=$2))",
            tenant_id, row["code_a"], row["code_b"],
        )
    return {"success": True}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agents": ["support_sales", "marketing"], "version": "0.2.0"}


# ── Sale (discounted products) ─────────────

@app.get("/api/sale")
async def list_sale(tenant_id: str = Depends(get_tenant_id)):
    """Return every product currently marked on sale, across all categories.
    Powers the admin Sale tab and the bot when customers ask about discounts."""
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT * FROM inventory WHERE tenant_id = $1 AND on_sale = true AND stock > 0 "
        "ORDER BY category, model, size, code",
        tenant_id,
    )
    return {"sale": [dict(r) for r in rows]}


# ── Dashboard (admin home) ──────────────────

@app.get("/api/dashboard")
async def dashboard(tenant_id: str = Depends(get_tenant_id)):
    """At-a-glance summary for the admin home: today's orders / revenue,
    low-stock alerts, a 7-day sales sparkline, and a recent-activity feed.
    All counts come from the existing tables — no new writes or migrations."""
    pool = await get_db()
    now = datetime.now(timezone.utc)
    today_prefix = now.date().isoformat()
    seven_days_ago = (now - timedelta(days=6)).date().isoformat()

    today_orders = await pool.fetch(
        "SELECT id, total FROM orders WHERE tenant_id = $1 AND created_at LIKE $2",
        tenant_id, f"{today_prefix}%",
    )
    today_revenue = sum(float(o["total"] or 0) for o in today_orders)

    low_stock = await pool.fetch(
        "SELECT id, code, model, size, stock, price, image_url FROM inventory "
        "WHERE tenant_id = $1 AND stock <= 1 AND image_url != '' "
        "ORDER BY stock ASC, code ASC LIMIT 12",
        tenant_id,
    )

    # Totals (lifetime-ish) for headline cards — all scoped to this tenant.
    totals = await pool.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM orders WHERE tenant_id = $1) AS total_orders, "
        "  (SELECT COALESCE(SUM(total), 0) FROM orders WHERE tenant_id = $1) AS total_revenue, "
        "  (SELECT COUNT(*) FROM inventory WHERE tenant_id = $1 AND stock > 0) AS in_stock, "
        "  (SELECT COUNT(DISTINCT conversation_id) FROM messages "
        "    WHERE tenant_id = $1 AND conversation_id NOT LIKE 'debug_%') AS conversations",
        tenant_id,
    )

    recent_orders = await pool.fetch(
        "SELECT id, total, created_at FROM orders "
        "WHERE tenant_id = $1 AND created_at >= $2 ORDER BY created_at",
        tenant_id, seven_days_ago,
    )
    by_day: dict[str, dict[str, float]] = {}
    for o in recent_orders:
        day = (o["created_at"] or "")[:10]
        if not day:
            continue
        slot = by_day.setdefault(day, {"count": 0, "revenue": 0.0})
        slot["count"] += 1
        slot["revenue"] += float(o["total"] or 0)

    sales_7d = []
    for i in range(7):
        d = (now - timedelta(days=6 - i)).date().isoformat()
        slot = by_day.get(d, {"count": 0, "revenue": 0.0})
        sales_7d.append({"date": d, "count": int(slot["count"]), "revenue": float(slot["revenue"])})

    recent_activity_rows = await pool.fetch(
        "SELECT id, customer_name, items, total, status, created_at "
        "FROM orders WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT 8",
        tenant_id,
    )
    recent_activity = [
        {
            "type": "order",
            "id": r["id"],
            "customer_name": r["customer_name"],
            "items": r["items"],
            "total": float(r["total"] or 0),
            "status": r["status"],
            "created_at": r["created_at"],
        }
        for r in recent_activity_rows
    ]

    return {
        "today": {"orders": len(today_orders), "revenue": today_revenue},
        "totals": {
            "orders": int(totals["total_orders"] or 0),
            "revenue": float(totals["total_revenue"] or 0),
            "in_stock": int(totals["in_stock"] or 0),
            "conversations": int(totals["conversations"] or 0),
        },
        "low_stock": [dict(r) for r in low_stock],
        "sales_7d": sales_7d,
        "recent_activity": recent_activity,
    }


# ── Conversation viewer (admin) ─────────────────────────────

@app.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Return every message in a conversation with the system-tag noise
    stripped from customer lines. Used by the admin conversation viewer to
    let the owner replay an entire chat without leaving the UI."""
    from src.insights import clean_user_message
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT id, role, content, created_at "
        "FROM messages WHERE conversation_id = $1 AND tenant_id = $2 "
        "ORDER BY created_at ASC, id ASC",
        conversation_id, tenant_id,
    )
    out = []
    for r in rows:
        content = r["content"] or ""
        role = r["role"]
        # Skip synthetic user-side "[customer sent a photo]" hints — those
        # are system notes we inject for the LLM, not real customer messages.
        if role == "user":
            cleaned = clean_user_message(content)
            if not cleaned:
                continue
            content = cleaned
        out.append({
            "id": r["id"],
            "role": role,
            "content": content,
            "created_at": r["created_at"],
        })
    return {"conversation_id": conversation_id, "messages": out}


# ── Insights (admin signals) ────────────────────────────────

@app.get("/api/insights/complaints")
async def insights_complaints(tenant_id: str = Depends(get_tenant_id)):
    """Customer messages that look like complaints (wrong match, wants operator)."""
    from src.insights import list_complaints
    return {"complaints": await list_complaints(tenant_id=tenant_id)}


@app.get("/api/insights/faqs")
async def insights_faqs(tenant_id: str = Depends(get_tenant_id)):
    """Most frequent customer questions — candidates for canned answers."""
    from src.insights import list_faq_candidates
    return {"faqs": await list_faq_candidates(tenant_id=tenant_id)}


@app.get("/api/insights/product-requests")
async def insights_product_requests(tenant_id: str = Depends(get_tenant_id)):
    """Customer asks for categories we don't carry — stocking hints for owner."""
    from src.insights import list_product_requests
    return {"requests": await list_product_requests(tenant_id=tenant_id)}


@app.post("/api/reindex")
async def reindex_catalog(full: bool = False, tenant_id: str = Depends(get_tenant_id)):
    """Re-index product catalog embeddings. Use full=true to wipe and re-embed all
    products; otherwise only new/missing ones are indexed."""
    from src.image_match import index_all_products
    pool = await get_db()
    if full:
        # Only wipe this tenant's embeddings so a second shop's embeddings
        # keep working during a reindex of another.
        try:
            await pool.execute(
                "DELETE FROM product_embeddings WHERE tenant_id = $1",
                tenant_id,
            )
        except Exception:
            # Fallback if product_embeddings doesn't have tenant_id yet
            # (pre-migration state) — delete all. Safe on single-tenant.
            await pool.execute("DELETE FROM product_embeddings")
    result = await index_all_products()
    return result


@app.post("/api/clear-conversation/{conversation_id}")
async def clear_conversation(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Wipe messages, AI hints, and tokens for one conversation so the bot
    starts fresh. Useful for re-testing the photo flow as a known customer."""
    pool = await get_db()
    stats: dict = {}
    stats["messages"] = await pool.execute(
        "DELETE FROM messages WHERE conversation_id = $1 AND tenant_id = $2",
        conversation_id, tenant_id,
    )
    try:
        stats["ai_photo_hints"] = await pool.execute(
            "DELETE FROM ai_photo_hints WHERE conversation_id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        # ai_photo_hints may not have tenant_id yet on older schemas.
        stats["ai_photo_hints"] = await pool.execute(
            "DELETE FROM ai_photo_hints WHERE conversation_id = $1",
            conversation_id,
        )
    try:
        stats["confirm_tokens"] = await pool.execute(
            "DELETE FROM confirm_tokens WHERE conversation_id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        pass
    try:
        stats["conversation"] = await pool.execute(
            "DELETE FROM conversations WHERE id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        pass
    return {"ok": True, "conversation_id": conversation_id, "stats": stats}


@app.post("/api/reindex/{code}")
async def reindex_one(code: str, tenant_id: str = Depends(get_tenant_id)):
    """Re-index a single product by code (useful when stored embedding is wrong)."""
    from src.image_match import index_product
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT id, code, model, size, image_url, image_url_back FROM inventory "
        "WHERE tenant_id = $1 AND UPPER(code) = UPPER($2) LIMIT 1",
        tenant_id, code,
    )
    if not row:
        return {"ok": False, "message": f"Code {code} not found"}
    try:
        await pool.execute(
            "DELETE FROM product_embeddings WHERE tenant_id = $1 AND UPPER(code) = UPPER($2)",
            tenant_id, code,
        )
    except Exception:
        await pool.execute(
            "DELETE FROM product_embeddings WHERE UPPER(code) = UPPER($1)",
            code,
        )
    ok = await index_product(row["id"], row["code"], row["model"], row["size"], row["image_url"], row.get("image_url_back", ""))
    return {"ok": ok, "code": row["code"]}


# ── Extra Photos (lifestyle/marketing) ──────────────────────

@app.get("/api/extra-photos")
async def list_extra_photos(code: str = "", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    if code:
        rows = await pool.fetch(
            "SELECT * FROM product_extra_photos WHERE tenant_id = $1 AND code = $2 "
            "ORDER BY created_at DESC",
            tenant_id, code,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM product_extra_photos WHERE tenant_id = $1 "
            "ORDER BY code, created_at DESC",
            tenant_id,
        )
    return {"photos": [dict(r) for r in rows]}


@app.post("/api/extra-photos")
async def add_extra_photo(
    code: str = Form(...),
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload a lifestyle/marketing photo for a product."""
    image_url = save_uploaded_image(image, f"extra_{code}")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO product_extra_photos (inventory_id, code, image_url, photo_type, tenant_id, created_at) "
        "VALUES ((SELECT id FROM inventory WHERE tenant_id = $1 AND UPPER(code) = UPPER($2) LIMIT 1), "
        "$2, $3, 'lifestyle', $1, $4) RETURNING id",
        tenant_id, code.upper(), image_url, now,
    )

    # Auto-index the new photo for AI matching
    try:
        from src.vision_match import index_extra_photo
        await index_extra_photo(code.upper(), image_url)
    except Exception as e:
        print(f"[EXTRA] Indexing failed: {e}")

    return {"id": row["id"], "image_url": image_url, "code": code}


@app.delete("/api/extra-photos/{photo_id}")
async def delete_extra_photo(photo_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute(
        "DELETE FROM product_extra_photos WHERE id = $1 AND tenant_id = $2",
        photo_id, tenant_id,
    )
    return {"success": True}


# ── Diagnostic endpoint ──────────────────────────────────────

@app.get("/api/test-vision")
async def test_vision():
    """Test Cloud Vision API + product matching on Railway."""
    results = {}

    # Step 1: Check credentials
    import os
    results["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "NOT SET")
    results["GOOGLE_CREDENTIALS_JSON_exists"] = bool(os.getenv("GOOGLE_CREDENTIALS_JSON"))

    # Step 2: Try Cloud Vision
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image()
        image.source.image_uri = "https://res.cloudinary.com/dw2yuqjrr/image/upload/v1774992584/tissu/tissu_strap_21.jpg"
        response = client.label_detection(image=image, max_results=3)
        results["vision_api"] = "OK"
        results["labels"] = [l.description for l in response.label_annotations[:3]]
    except Exception as e:
        results["vision_api"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

    # Step 3: Try analyze_and_match + save to ai_photo_hints
    try:
        import httpx as _httpx
        from src.vision_match import analyze_and_match
        from datetime import datetime as _dt, timezone as _tz
        url = "https://res.cloudinary.com/dw2yuqjrr/image/upload/v1774992584/tissu/tissu_strap_21.jpg"
        async with _httpx.AsyncClient(follow_redirects=True) as c:
            img = (await c.get(url)).content
        results["image_bytes"] = len(img)
        match = await analyze_and_match(img)
        results["analyze_and_match"] = "OK"
        results["matched"] = match.get("matched")
        results["code"] = match.get("code")
        results["score"] = match.get("score")
        # Try saving to ai_photo_hints (same as facebook.py does)
        if match.get("matched"):
            product = match.get("product", {})
            now = _dt.now(_tz.utc).isoformat()
            await pool.execute(
                """INSERT INTO ai_photo_hints (conversation_id, code, model, size, price, image_url, score, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (conversation_id) DO UPDATE SET code=$2""",
                "test_diagnostic", match["code"], product.get("model", ""),
                product.get("size", ""), float(product.get("price", 0)),
                product.get("image_url", ""), float(match["score"]), now,
            )
            # Read it back
            check = await pool.fetchrow("SELECT code, score FROM ai_photo_hints WHERE conversation_id = 'test_diagnostic'")
            results["hint_saved"] = check["code"] if check else "FAIL"
            await pool.execute("DELETE FROM ai_photo_hints WHERE conversation_id = 'test_diagnostic'")
    except Exception as e:
        import traceback as _tb
        results["analyze_and_match"] = f"FAIL: {type(e).__name__}: {str(e)[:300]}"
        results["traceback"] = _tb.format_exc()[-500:]

    # Step 4: Check fingerprints
    try:
        pool = await get_db()
        row = await pool.fetchrow("SELECT COUNT(*) as c FROM product_fingerprints")
        results["fingerprints_count"] = row["c"]
    except Exception as e:
        results["fingerprints_count"] = f"FAIL: {e}"

    return results


# ── Seed Data ────────────────────────────────────────────────

async def seed_knowledge_base():
    """Seed Tissu Shop knowledge base (always) and inventory (only missing codes)."""
    pool = await get_db()

    # Always reseed knowledge base (small, no user edits) for the
    # default tenant. New tenants seed their own KB separately.
    count_row = await pool.fetchrow(
        "SELECT COUNT(*) as c FROM knowledge_base WHERE tenant_id = $1",
        DEFAULT_TENANT_ID,
    )
    kb_count = count_row["c"]
    if kb_count > 0:
        await pool.execute(
            "DELETE FROM knowledge_base WHERE tenant_id = $1",
            DEFAULT_TENANT_ID,
        )

    now = datetime.now(timezone.utc).isoformat()

    articles = [
        ("რა ფასია?", "პატარა ზომა (33x25 სმ) — 69 ლარი. დიდი ზომა (37x27 სმ) — 74 ლარი.", "pricing"),
        ("რა ზომები გაქვთ?", "გვაქვს 2 ზომა: პატარა (33x25 სმ, 13-14 ინჩი ლეპტოპისთვის) და დიდი (37x27 სმ, 15-16 ინჩი ლეპტოპისთვის).", "products"),
        ("რა მოდელები გაქვთ?", "გვაქვს 2 მოდელი: თასმიანი (სახელურით) და ფხრიწიანი (zipper-ით).", "products"),
        ("როგორ ხდება მიწოდება?", "მიწოდება თბილისის მასშტაბით 6 ლარი. ღამის 12 საათამდე შეკვეთაზე მიწოდება მეორე დღეს. შაბათის შეკვეთა ორშაბათს. კვირას მიწოდება არ ხდება.", "delivery"),
        ("თუ არ ვიქნები მისამართზე?", "შეგიძლიათ მიუთითოთ მიმდებარე ადგილი სადაც დატოვებს კურიერი. თუ ვერ ჩაიბარებთ, კურიერი წაიღებს უკან და მეორე დღეს მოგაწვდით, რისთვისაც დამატებითი საკურიეროს გადახდა მოგიწევთ.", "delivery"),
        ("როგორ გადავიხადო?", "გადახდა ბანკის ანგარიშზე. თიბისი: GE58TB7085345064300066. საქართველოს ბანკი: GE65BG0000000358364200. გადახდის შემდეგ გვჭირდება: სქრინი/ქვითარი + მისამართი + ტელეფონის ნომერი.", "payment"),
        ("რისგან არის გაკეთებული?", "ლეპტოპის ქეისები ტილოსგან მზადდება, წყალგაუმტარია. ყოველი ცალი ხელნაკეთი და უნიკალური დიზაინისაა.", "products"),
    ]
    for q, a, cat in articles:
        await pool.execute(
            "INSERT INTO knowledge_base (question, answer, category, tenant_id, created_at) "
            "VALUES ($1, $2, $3, $4, $5)",
            q, a, cat, DEFAULT_TENANT_ID, now,
        )

    # Seed inventory — only add products that don't exist yet (by code)
    # for the default tenant.
    seed_file = Path(__file__).parent / "seed_inventory.json"
    if seed_file.exists():
        existing = await pool.fetch(
            "SELECT code FROM inventory WHERE tenant_id = $1",
            DEFAULT_TENANT_ID,
        )
        existing_codes = {row["code"] for row in existing}

        items = json.loads(seed_file.read_text())
        for item in items:
            code = item.get("code", "")
            if code and code not in existing_codes:
                await pool.execute(
                    "INSERT INTO inventory (product_name, model, size, color, style, code, tags, price, stock, image_url, image_url_back, tenant_id, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
                    item["product_name"], item["model"], item["size"], item.get("color", ""), item.get("style", ""),
                    code, item.get("tags", ""), item["price"], item["stock"],
                    item.get("image_url", ""), item.get("image_url_back", ""),
                    DEFAULT_TENANT_ID, now, now,
                )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", API_PORT))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("RAILWAY_ENVIRONMENT") is None)
