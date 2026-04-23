from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import asyncpg

from src.config import DATABASE_URL

# Every tenant-scoped table carries a tenant_id so we can host multiple
# shops out of one database + one deployment. The single-shop Tissu
# install stays on 'default'; new tenants get their own slug when an
# operator seeds a row into api_keys.
DEFAULT_TENANT_ID = "default"

TENANT_SCOPED_TABLES = (
    "conversations",
    "messages",
    "leads",
    "tickets",
    "content",
    "knowledge_base",
    "inventory",
    "orders",
    "confirm_tokens",
    "categories",
    "product_extra_photos",
    # These tables are created lazily by other modules (image_match,
    # vision_match, the Facebook webhook's photo-hint pipeline). ADD
    # COLUMN will fail if the table isn't there yet, so init_db wraps
    # each statement in its own try/except to keep going.
    "product_pairs",
    "product_embeddings",
    "product_fingerprints",
    "ai_photo_hints",
)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def get_db() -> asyncpg.Pool:
    """Return the connection pool. Callers use `pool.fetch()`, `pool.execute()`, etc."""
    return await get_pool()


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                created_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT,
                company TEXT,
                phone TEXT,
                source TEXT,
                score INTEGER DEFAULT 0,
                status TEXT DEFAULT 'new',
                notes TEXT,
                conversation_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id SERIAL PRIMARY KEY,
                subject TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'open',
                priority TEXT DEFAULT 'medium',
                customer_email TEXT,
                conversation_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS content (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                content_type TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                tags TEXT DEFAULT '[]',
                scheduled_at TEXT,
                published_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                created_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id SERIAL PRIMARY KEY,
                product_name TEXT NOT NULL,
                model TEXT NOT NULL,
                size TEXT NOT NULL,
                color TEXT,
                style TEXT,
                code TEXT,
                tags TEXT DEFAULT '',
                price REAL NOT NULL,
                stock INTEGER DEFAULT 0,
                image_url TEXT,
                image_url_back TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                customer_name TEXT NOT NULL,
                customer_phone TEXT,
                customer_address TEXT,
                items TEXT NOT NULL,
                total REAL NOT NULL,
                delivery_fee REAL DEFAULT 6,
                payment_method TEXT,
                payment_confirmed INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                conversation_id TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS confirm_tokens (
                token TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                action TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                used_at TEXT
            );
        """)
        # ── Migrations ─────────────────────────────
        # Add a product category column so inventory can hold more than just
        # bags. Existing rows default to 'bag' so every earlier bot query keeps
        # returning the same results — necklaces (and future categories) live
        # under category != 'bag' and are filtered out of the bag-only queries.
        await conn.execute(
            "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'bag'"
        )
        # Same column on product_embeddings so the photo matcher can scope by
        # category at query time. Defaults preserve existing bag rows.
        await conn.execute(
            "ALTER TABLE product_embeddings ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'bag'"
        )
        # Free-form attribute bag per product (size/colour/material for a
        # necklace, any future-category fields, etc.). JSONB so we can query
        # into it later without a new column per attribute.
        await conn.execute(
            "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS attrs JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        # Sale flags — products marked on_sale surface in the ფასდაკლება admin
        # tab and in the bot when a customer asks about discounts. Sale_price
        # is optional: if null, the regular price applies with a visual flag.
        await conn.execute(
            "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS on_sale BOOLEAN NOT NULL DEFAULT false"
        )
        await conn.execute(
            "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS sale_price REAL"
        )
        # Speed up the admin's angles lookup and the orders-by-conversation
        # lookups the insights tab runs on every open.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_extra_photos_inventory ON product_extra_photos (inventory_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages (conversation_id, created_at)"
        )
        # Categories registry — lets the owner define new product categories
        # (e.g. ქამრები, ყუთები) from the admin UI with their own custom
        # field list (ფერი, სიგრძე, მასალა, …). Seeded below with the
        # two built-in categories we already ship with.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                emoji TEXT NOT NULL DEFAULT '📦',
                fields JSONB NOT NULL DEFAULT '[]'::jsonb,
                sort_order INTEGER NOT NULL DEFAULT 100,
                created_at TEXT NOT NULL
            )
        """)
        now_iso = datetime.now(timezone.utc).isoformat()
        # Note: tenant_id is supplied by the DEFAULT in the column spec,
        # so these seed rows land in the DEFAULT_TENANT_ID tenant without
        # us having to mention it explicitly. We keep them row-locked with
        # ON CONFLICT (slug) DO NOTHING — slugs are treated as globally
        # unique for now; we'll revisit if we ever allow two tenants to
        # reuse the same category slug.
        await conn.execute(
            """INSERT INTO categories (slug, name, emoji, fields, sort_order, created_at)
               VALUES ($1, $2, $3, $4::jsonb, $5, $6)
               ON CONFLICT (slug) DO NOTHING""",
            'bag', 'ჩანთები', '💼',
            json.dumps([{"key": "model", "label": "მოდელი"}, {"key": "size", "label": "ზომა"}]),
            10, now_iso,
        )
        await conn.execute(
            """INSERT INTO categories (slug, name, emoji, fields, sort_order, created_at)
               VALUES ($1, $2, $3, $4::jsonb, $5, $6)
               ON CONFLICT (slug) DO NOTHING""",
            'necklace', 'ყელსაბამები', '📿',
            json.dumps([]),  # owner adds fields as they need them
            20, now_iso,
        )
        # One-shot: if necklace is still carrying the original seeded
        # schema (length + material), clear it — the owner prefers to
        # define fields themselves. No-op once they've edited.
        await conn.execute(
            """UPDATE categories SET fields = '[]'::jsonb
               WHERE slug = 'necklace'
               AND fields = '[{"key": "length", "label": "სიგრძე"}, {"key": "material", "label": "მასალა"}]'::jsonb"""
        )

        # ── Multi-tenant migration ─────────────────────────────
        # Every tenant-scoped table gets a tenant_id column with the default
        # 'default'. This is idempotent: ADD COLUMN IF NOT EXISTS is a no-op
        # on re-run, and the DEFAULT 'default' backfills existing rows on
        # the very first migration so the single-shop Tissu deployment keeps
        # working without any data update.
        for table in TENANT_SCOPED_TABLES:
            try:
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                    f"tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT_ID}'"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_tenant "
                    f"ON {table} (tenant_id)"
                )
            except asyncpg.exceptions.UndefinedTableError:
                # Lazy tables (product_embeddings, ai_photo_hints, …) get
                # their tenant_id when their owning module creates them.
                print(f"[migration] skipping {table}: not created yet", flush=True)

        # api_keys maps each X-API-Key value to the tenant that owns the
        # data it can see. The middleware looks up tenant_id here; if the
        # table is empty (first boot after migration) we seed it from the
        # ADMIN_API_KEY env var so the existing admin UI keeps working.
        #
        # `scope` decides which /api/* paths the key is allowed to hit:
        #   - 'admin'      — everything under /api/* (the shop owner)
        #   - 'storefront' — only /api/storefront/* (the public website)
        # Defaults to 'admin' so the existing bootstrap key keeps working.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                label TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await conn.execute(
            "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS "
            "scope TEXT NOT NULL DEFAULT 'admin'"
        )
        admin_key = os.environ.get("ADMIN_API_KEY", "").strip()
        if admin_key:
            await conn.execute(
                """INSERT INTO api_keys (key, tenant_id, label, scope, created_at)
                   VALUES ($1, $2, $3, 'admin', $4)
                   ON CONFLICT (key) DO NOTHING""",
                admin_key, DEFAULT_TENANT_ID, "bootstrap-admin",
                datetime.now(timezone.utc).isoformat(),
            )

        # Read-only storefront key — seeded from the STOREFRONT_API_KEY
        # env var so the public website can hit /api/storefront/* without
        # sharing the admin master key.
        storefront_key = os.environ.get("STOREFRONT_API_KEY", "").strip()
        if storefront_key:
            await conn.execute(
                """INSERT INTO api_keys (key, tenant_id, label, scope, created_at)
                   VALUES ($1, $2, $3, 'storefront', $4)
                   ON CONFLICT (key) DO NOTHING""",
                storefront_key, DEFAULT_TENANT_ID, "bootstrap-storefront",
                datetime.now(timezone.utc).isoformat(),
            )

        # Product description — per-row copy the storefront renders on
        # the /product/[id] detail page. Nullable text, defaults to empty
        # so existing rows don't need a backfill.
        await conn.execute(
            "ALTER TABLE inventory ADD COLUMN IF NOT EXISTS description TEXT"
        )

        # ── Admin users (owner / staff login) ──────────────────
        # Replaces the browser prompt() hack — /admin/login now takes a
        # real email + password and verifies an argon2id hash from here.
        # tenant_id keeps the table multi-tenant-ready; each tenant has
        # their own owner seeded from ADMIN_OWNER_EMAIL / PASSWORD.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'owner',
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
        """)
        # Email is unique per tenant — two different shops can each have
        # owner@tissu.com without clashing.
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_users_tenant_email "
            "ON admin_users (tenant_id, LOWER(email))"
        )

        # Password-reset tokens. We store only the hash of the token so
        # a DB dump doesn't hand out active reset links. used_at gets
        # stamped the moment the token is redeemed, and a successful
        # reset also invalidates every other unused token for the user.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_password_resets (
                token_hash TEXT PRIMARY KEY,
                admin_user_id INTEGER NOT NULL REFERENCES admin_users(id),
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_password_resets_user "
            "ON admin_password_resets (admin_user_id)"
        )

        # Seed an owner account from the env vars if both are set AND
        # no account exists for that tenant yet. This mirrors the
        # bootstrap-admin pattern for api_keys — first-boot convenience
        # that becomes a no-op after rotation.
        owner_email = os.environ.get("ADMIN_OWNER_EMAIL", "").strip().lower()
        owner_password = os.environ.get("ADMIN_OWNER_PASSWORD", "")
        if owner_email and owner_password:
            existing = await conn.fetchval(
                "SELECT id FROM admin_users WHERE tenant_id = $1 "
                "AND LOWER(email) = $2",
                DEFAULT_TENANT_ID, owner_email,
            )
            if not existing:
                # Import locally so the rest of the module doesn't pay
                # the argon2-cffi import cost when the seed is a no-op.
                from src.passwords import hash_password
                try:
                    pw_hash = hash_password(owner_password)
                except Exception as e:
                    print(f"[admin-seed] refusing to seed owner: {e}", flush=True)
                else:
                    await conn.execute(
                        "INSERT INTO admin_users "
                        "(tenant_id, email, password_hash, role, created_at) "
                        "VALUES ($1, $2, $3, 'owner', $4)",
                        DEFAULT_TENANT_ID, owner_email, pw_hash,
                        datetime.now(timezone.utc).isoformat(),
                    )
                    print(f"[admin-seed] seeded owner {owner_email}", flush=True)


async def resolve_tenant_id(api_key: str) -> str | None:
    """Return the tenant_id for a given api_key, or None if unknown.

    Thin wrapper around :func:`resolve_api_key` that only hands back the
    tenant portion. Kept for existing callers (tests, legacy imports).
    """
    hit = await resolve_api_key(api_key)
    return hit[0] if hit else None


async def resolve_api_key(api_key: str) -> tuple[str, str] | None:
    """Return ``(tenant_id, scope)`` for a given api_key, or None.

    Called by the auth middleware on every /api/* request. Scope is one
    of 'admin' (everything under /api/*) or 'storefront' (only
    /api/storefront/*). Kept DB-hit-per-call for now — a process cache
    is worth adding once we have real multi-tenant traffic.
    """
    if not api_key:
        return None
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT tenant_id, scope FROM api_keys WHERE key = $1",
        api_key,
    )
    if not row:
        return None
    return (row["tenant_id"], row["scope"] or "admin")


async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    tool_calls: list | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
):
    """Insert a message and bump the parent conversation's updated_at.

    tenant_id defaults to the single-shop value — the bot webhooks don't
    carry a tenant yet (one Facebook page per deployment), so every
    message lands in the default tenant. Admin callers can pass tenant_id
    explicitly if they ever need to write on behalf of a different shop.
    """
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_calls, tenant_id, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            conversation_id, role, content,
            json.dumps(tool_calls) if tool_calls else None,
            tenant_id, now,
        )
        await conn.execute(
            "UPDATE conversations SET updated_at = $1 WHERE id = $2",
            now, conversation_id,
        )


async def get_conversation_messages(conversation_id: str) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT role, content, tool_calls FROM messages WHERE conversation_id = $1 ORDER BY created_at",
        conversation_id,
    )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def find_admin_user_by_email(
    email: str, tenant_id: str = DEFAULT_TENANT_ID,
) -> dict | None:
    """Return the admin_users row matching ``email`` (case-insensitive)
    for the given tenant, or None. Used on the login path."""
    if not email:
        return None
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, tenant_id, email, password_hash, role, "
        "created_at, last_login_at FROM admin_users "
        "WHERE tenant_id = $1 AND LOWER(email) = LOWER($2)",
        tenant_id, email,
    )
    return dict(row) if row else None


async def find_admin_user_by_id(user_id: int) -> dict | None:
    """Return the admin_users row for ``user_id`` or None. Used by the
    session middleware to load the current user on every admin request.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, tenant_id, email, password_hash, role, "
        "created_at, last_login_at FROM admin_users WHERE id = $1",
        user_id,
    )
    return dict(row) if row else None


async def mark_admin_user_logged_in(user_id: int) -> None:
    """Stamp last_login_at on successful login."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE admin_users SET last_login_at = $1 WHERE id = $2",
        datetime.now(timezone.utc).isoformat(), user_id,
    )


async def update_admin_user_password(user_id: int, new_hash: str) -> None:
    """Replace the password hash for ``user_id`` (used by rehash on
    login and by the password-reset flow)."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE admin_users SET password_hash = $1 WHERE id = $2",
        new_hash, user_id,
    )


async def ensure_conversation(
    conversation_id: str,
    agent_type: str,
    tenant_id: str = DEFAULT_TENANT_ID,
):
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "INSERT INTO conversations (id, agent_type, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO NOTHING",
        conversation_id, agent_type, tenant_id, now, now,
    )
