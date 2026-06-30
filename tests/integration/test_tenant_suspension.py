"""Phase 1 contract test: a suspended tenant is locked out of every
admin /api/* route, but their public storefront stays online so the
shop's customers don't see the site break.

Skipped by default — set ``TEST_INTEGRATION=1`` to run against the
real Supabase DB. Cleans up everything it creates regardless of
outcome.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest


RUN_INTEGRATION = os.environ.get("TEST_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set TEST_INTEGRATION=1 to run — hits a real Postgres DB",
)


async def _seed_active_tenant():
    """Create a brand-new tenant, activate the owner password, and
    return everything needed to log in as them."""
    import asyncpg

    from src.config import DATABASE_URL
    from src.db import (
        create_tenant, create_admin_user_pending,
        create_activation_token,
    )
    from src.passwords import hash_password
    from src.tokens import generate_reset_token

    tenant_id = f"__sus_{uuid.uuid4().hex[:8]}__"
    email = f"sus_{uuid.uuid4().hex[:6]}@test.invalid"
    password = "suspend-test-Pw99"
    now = datetime.now(timezone.utc).isoformat()

    await create_tenant(
        tenant_id, "Suspension Test", email,
        pricing_tier="start", feature_set="bot", status="active",
    )
    user_id = await create_admin_user_pending(tenant_id, email, role="owner")
    # Bypass the activation flow — write the hash directly so the
    # tenant's owner can log in via /admin/login from the start.
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            "UPDATE admin_users SET password_hash = $1 WHERE id = $2",
            hash_password(password), user_id,
        )
    finally:
        await conn.close()

    return {
        "tenant_id": tenant_id,
        "email": email,
        "password": password,
        "user_id": user_id,
    }


async def _cleanup(tenant_id: str) -> None:
    from src.db import delete_tenant_cascade
    try:
        await delete_tenant_cascade(tenant_id)
    except Exception as e:
        # We don't want a test bug to leak rows; complain loudly so a
        # failed run is obvious instead of silently piling up garbage.
        print(f"[test cleanup] failed to delete {tenant_id}: {e}", flush=True)


async def _run_suspension_assertions():
    """All four assertions inside one asyncio.run so we keep the
    fixture lifecycle and HTTP client lifetimes tight."""
    from httpx import AsyncClient, ASGITransport

    from server import app
    from src.db import update_tenant_status

    seeded = await _seed_active_tenant()
    tenant_id = seeded["tenant_id"]
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            # 1. While active, the tenant's owner can log in and reach
            #    /api/inventory.
            login = await c.post(
                "/admin/login",
                data={"email": seeded["email"], "password": seeded["password"]},
                follow_redirects=False,
            )
            assert login.status_code == 303, f"login should 303, got {login.status_code}"
            cookies = c.cookies

            inv_active = await c.get(
                "/api/inventory",
                cookies=cookies,
            )
            assert inv_active.status_code == 200, (
                f"active tenant /api/inventory should 200, got "
                f"{inv_active.status_code}: {inv_active.text}"
            )

        # 2. Suspend the tenant.
        await update_tenant_status(tenant_id, "suspended")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            # 3. Same cookies, but now /api/inventory returns 403 with
            #    error=account_suspended.
            inv_suspended = await c.get(
                "/api/inventory", cookies=cookies,
            )
            assert inv_suspended.status_code == 403, (
                f"suspended tenant /api/inventory should 403, got "
                f"{inv_suspended.status_code}: {inv_suspended.text}"
            )
            body = inv_suspended.json()
            assert body.get("error") == "account_suspended", body

            # 4. Storefront stays open (the shop's customers shouldn't
            #    see the website break over a missed bank transfer).
            sf_health = await c.get("/api/storefront/health")
            assert sf_health.status_code == 200, (
                f"storefront health should 200 even when tenant suspended, "
                f"got {sf_health.status_code}"
            )
    finally:
        await _cleanup(tenant_id)


def test_suspended_tenant_is_locked_out_of_admin_api():
    """End-to-end: create + activate + suspend + verify 403."""
    asyncio.run(_run_suspension_assertions())
