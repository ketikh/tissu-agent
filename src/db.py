from __future__ import annotations

import json
from datetime import datetime, timezone

import asyncpg

from src.config import DATABASE_URL

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


async def save_message(conversation_id: str, role: str, content: str, tool_calls: list | None = None):
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_calls, created_at) VALUES ($1, $2, $3, $4, $5)",
            conversation_id, role, content, json.dumps(tool_calls) if tool_calls else None, now,
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


async def ensure_conversation(conversation_id: str, agent_type: str):
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "INSERT INTO conversations (id, agent_type, created_at, updated_at) VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO NOTHING",
        conversation_id, agent_type, now, now,
    )
