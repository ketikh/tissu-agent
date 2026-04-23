"""Cross-tenant isolation integration test.

Proves that a valid API key for one tenant cannot see or mutate another
tenant's data. Hits the real Postgres (Supabase) database so the
assertions run against the same query builder + SQL that production uses.

Skipped by default so CI doesn't touch Supabase. To run locally:

    TEST_INTEGRATION=1 DATABASE_URL=... ADMIN_API_KEY=... \
      .venv/bin/python -m pytest tests/integration/test_multitenant.py -v

The test seeds two tenants with disposable slugs, seeds one inventory
row per tenant, runs the admin HTTP endpoints through a TestClient with
each key, asserts isolation, and cleans up on teardown regardless of
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


async def _seed():
    """Create two throwaway tenants with one inventory row each."""
    import asyncpg

    from src.config import DATABASE_URL

    tenant_a = f"__iso_a_{uuid.uuid4().hex[:8]}__"
    tenant_b = f"__iso_b_{uuid.uuid4().hex[:8]}__"
    key_a = f"test-key-a-{uuid.uuid4().hex}"
    key_b = f"test-key-b-{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc).isoformat()

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        for key, tenant in ((key_a, tenant_a), (key_b, tenant_b)):
            await conn.execute(
                "INSERT INTO api_keys (key, tenant_id, label, created_at) "
                "VALUES ($1, $2, $3, $4)",
                key, tenant, "integration-test", now,
            )

        row_a = await conn.fetchrow(
            "INSERT INTO inventory (product_name, model, size, code, price, stock, "
            "image_url, tenant_id, created_at, updated_at) "
            "VALUES ($1, 'ფხრიწიანი', 'პატარა', 'TESTA1', 10, 5, '', $2, $3, $3) "
            "RETURNING id",
            f"Tenant-A-only-{uuid.uuid4().hex[:6]}", tenant_a, now,
        )
        row_b = await conn.fetchrow(
            "INSERT INTO inventory (product_name, model, size, code, price, stock, "
            "image_url, tenant_id, created_at, updated_at) "
            "VALUES ($1, 'ფხრიწიანი', 'პატარა', 'TESTB1', 20, 5, '', $2, $3, $3) "
            "RETURNING id",
            f"Tenant-B-only-{uuid.uuid4().hex[:6]}", tenant_b, now,
        )
    finally:
        await conn.close()

    return {
        "tenant_a": tenant_a,
        "tenant_b": tenant_b,
        "key_a": key_a,
        "key_b": key_b,
        "id_a": row_a["id"],
        "id_b": row_b["id"],
    }


async def _cleanup(t: dict) -> None:
    """Delete everything the seeder created, best-effort."""
    import asyncpg

    from src.config import DATABASE_URL

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            "DELETE FROM inventory WHERE tenant_id = ANY($1::text[])",
            [t["tenant_a"], t["tenant_b"]],
        )
        await conn.execute(
            "DELETE FROM api_keys WHERE key = ANY($1::text[])",
            [t["key_a"], t["key_b"]],
        )
    finally:
        await conn.close()


async def _run_isolation_checks():
    """The three assertions live here so a single asyncio.run() covers them.

    Keeping them in one function avoids pytest-asyncio fixture lifecycle
    quirks that interfered with an earlier multi-test version of this
    file — the point of the test is end-to-end proof, not test count.
    """
    import asyncpg
    from httpx import AsyncClient, ASGITransport

    from server import app
    from src.config import DATABASE_URL

    t = await _seed()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # 1. SELECT isolation — each tenant only sees its own rows.
            resp_a = await c.get("/api/inventory", headers={"X-API-Key": t["key_a"]})
            resp_b = await c.get("/api/inventory", headers={"X-API-Key": t["key_b"]})
            assert resp_a.status_code == 200, f"tenant A /api/inventory: {resp_a.text}"
            assert resp_b.status_code == 200, f"tenant B /api/inventory: {resp_b.text}"
            ids_a = {row["id"] for row in resp_a.json()["inventory"]}
            ids_b = {row["id"] for row in resp_b.json()["inventory"]}
            assert t["id_a"] in ids_a, "tenant A should see its own row"
            assert t["id_b"] not in ids_a, "tenant A must NOT see tenant B's row"
            assert t["id_b"] in ids_b, "tenant B should see its own row"
            assert t["id_a"] not in ids_b, "tenant B must NOT see tenant A's row"

            # 2. DELETE isolation — tenant B cannot delete tenant A's row
            #    by ID, even though the endpoint happily returns 200.
            del_resp = await c.delete(
                f"/api/inventory/{t['id_a']}",
                headers={"X-API-Key": t["key_b"]},
            )
            assert del_resp.status_code == 200

            # 3. Dashboard totals are per-tenant (each tenant has exactly 1).
            dash_a = await c.get("/api/dashboard", headers={"X-API-Key": t["key_a"]})
            dash_b = await c.get("/api/dashboard", headers={"X-API-Key": t["key_b"]})
            assert dash_a.json()["totals"]["in_stock"] == 1
            assert dash_b.json()["totals"]["in_stock"] == 1

        # 4. Confirm tenant A's row survived tenant B's delete attempt
        #    by reading directly from the DB.
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            still_there = await conn.fetchval(
                "SELECT COUNT(*) FROM inventory WHERE id = $1 AND tenant_id = $2",
                t["id_a"], t["tenant_a"],
            )
        finally:
            await conn.close()
        assert still_there == 1, "tenant B's DELETE should not have touched tenant A"
    finally:
        await _cleanup(t)


def test_cross_tenant_isolation():
    """End-to-end: two tenants + inventory + admin endpoints + DB check."""
    asyncio.run(_run_isolation_checks())
