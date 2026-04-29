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

# How long a fresh tenant stays in 'trial' before the platform
# auto-suspends them. Operator can mark-paid at any point during
# the trial to flip status='active' immediately.
TRIAL_DURATION_DAYS = 10

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

        # ── Tenants registry ───────────────────────────────────
        # Central table of all customers (shops) on the platform. The
        # tenant_id column on every other table points here. Status /
        # plan / payment_due_date drive the super-admin UI and the
        # "your account is suspended" lock-out.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                shop_name TEXT NOT NULL,
                owner_email TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'trial',
                plan TEXT NOT NULL DEFAULT 'start',
                payment_due_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Encrypted Facebook / WhatsApp credentials per tenant — filled
        # in by the super-admin when onboarding a new customer. The
        # webhook looks up a tenant by fb_page_id so it knows which
        # token to use when replying.
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS fb_page_id TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS fb_page_token_encrypted TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS fb_verify_token TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS wa_phone_id TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS wa_token_encrypted TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS owner_whatsapp TEXT"
        )

        # ── Phase 1: SaaS feature flags + plan separation ──────
        # The owner sells the platform on two orthogonal axes:
        #   feature_set  ∈ {bot, site, combo}  — what the tenant gets
        #   pricing_tier ∈ {start, grow, pro}  — the price band
        # bot_enabled / site_enabled are stored explicitly so a future
        # admin can flip one off without changing the headline plan.
        # suspended_at parallels status='suspended' for callers that
        # want a timestamp instead of an enum.
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS feature_set TEXT NOT NULL DEFAULT 'bot'"
        )
        # Rename existing 'plan' → 'pricing_tier' (metadata-only on Postgres,
        # safe re-run because we catch the "column doesn't exist" error
        # the second time around).
        try:
            await conn.execute(
                "ALTER TABLE tenants RENAME COLUMN plan TO pricing_tier"
            )
        except asyncpg.exceptions.UndefinedColumnError:
            pass  # already renamed on a prior boot
        except asyncpg.exceptions.DuplicateColumnError:
            pass  # both columns exist somehow — leave alone
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS bot_enabled BOOLEAN NOT NULL DEFAULT false"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS site_enabled BOOLEAN NOT NULL DEFAULT false"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS suspended_at TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS cloudinary_folder TEXT"
        )
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS storefront_subdomain TEXT"
        )
        # When this trial expires, the auth middleware auto-suspends
        # the tenant. NULL means "no trial deadline" — used for
        # 'active' tenants whose subscription is paid through
        # payment_due_date instead.
        await conn.execute(
            "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS trial_ends_at TEXT"
        )

        # Quick fb_page_id -> tenant lookup for the webhook dispatcher.
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tenants_fb_page "
            "ON tenants (fb_page_id) WHERE fb_page_id IS NOT NULL"
        )
        # Same for storefront_subdomain — Phase 7 will route by Host header.
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tenants_subdomain "
            "ON tenants (storefront_subdomain) WHERE storefront_subdomain IS NOT NULL"
        )
        # Seed the existing default tenant row for Tissu so the shop
        # shows up in the super-admin list with a known-good status.
        # This INSERT is idempotent — ON CONFLICT skips once seeded.
        now_iso = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            """INSERT INTO tenants
               (tenant_id, shop_name, owner_email, status, pricing_tier,
                feature_set, bot_enabled, site_enabled, cloudinary_folder,
                payment_due_date, created_at, updated_at)
               VALUES ($1, $2, $3, 'active', 'pro', 'combo',
                       true, true, $4, NULL, $5, $5)
               ON CONFLICT (tenant_id) DO NOTHING""",
            DEFAULT_TENANT_ID, "Tissu Shop",
            os.environ.get("ADMIN_OWNER_EMAIL", "").strip().lower() or "owner@tissu.shop",
            DEFAULT_TENANT_ID,  # cloudinary_folder
            now_iso,
        )

        # ── Phase 1 backfill ───────────────────────────────────
        # Existing rows from previous migrations are missing the new
        # flags. Each UPDATE is guarded by WHERE so it only touches
        # rows that haven't been backfilled yet — safe to re-run.
        # Default tenant: owner platform = combo + cloudinary folder.
        await conn.execute(
            "UPDATE tenants SET feature_set = 'combo', "
            "                   bot_enabled = true, "
            "                   site_enabled = true, "
            "                   cloudinary_folder = $1 "
            "WHERE tenant_id = $1 "
            "  AND (feature_set != 'combo' OR cloudinary_folder IS NULL)",
            DEFAULT_TENANT_ID,
        )
        # Other tenants (created before Phase 1): they pay for the bot
        # today, so feature_set='bot' + bot_enabled=true. cloudinary_
        # folder defaults to their own tenant_id so uploads stay
        # isolated when Phase 4 wires Cloudinary per-tenant.
        await conn.execute(
            "UPDATE tenants SET feature_set = 'bot', bot_enabled = true "
            "WHERE tenant_id != $1 "
            "  AND feature_set NOT IN ('bot', 'site', 'combo')",
            DEFAULT_TENANT_ID,
        )
        await conn.execute(
            "UPDATE tenants SET cloudinary_folder = tenant_id "
            "WHERE cloudinary_folder IS NULL"
        )
        # Sync the bot/site enabled flags with feature_set for any
        # legacy row that was created before the flag columns existed:
        # feature_set was filled by DEFAULT 'bot' but bot_enabled was
        # still DB DEFAULT false, so the entitlement check would lock
        # them out of the bot they're paying for. Idempotent — once
        # the flags match feature_set, the WHERE condition is false.
        await conn.execute(
            "UPDATE tenants SET bot_enabled = true "
            "WHERE feature_set IN ('bot', 'combo') AND bot_enabled = false"
        )
        await conn.execute(
            "UPDATE tenants SET site_enabled = true "
            "WHERE feature_set IN ('site', 'combo') AND site_enabled = false"
        )
        # Sync suspended_at with status — for any historical row whose
        # status was flipped to 'suspended' before the column existed.
        await conn.execute(
            "UPDATE tenants SET suspended_at = updated_at "
            "WHERE status = 'suspended' AND suspended_at IS NULL"
        )
        # Backfill trial_ends_at: for any tenant currently in 'trial'
        # without a trial_ends_at, set it to created_at + TRIAL_DURATION
        # so the new auto-suspend logic has a deadline to enforce.
        await conn.execute(
            "UPDATE tenants SET trial_ends_at = "
            "  TO_CHAR(("
            "    (created_at::timestamptz) + (INTERVAL '1 day' * $1)"
            "  ) AT TIME ZONE 'UTC', "
            "  'YYYY-MM-DD\"T\"HH24:MI:SS.US+00:00') "
            "WHERE status = 'trial' AND trial_ends_at IS NULL "
            "AND tenant_id != $2",
            TRIAL_DURATION_DAYS, DEFAULT_TENANT_ID,
        )

        # ── Phase 1: pricing_plans table ───────────────────────
        # Owner-editable price matrix. 9 cells (3 feature_sets × 3
        # pricing_tiers), all NULL until owner fills them in via the
        # super-admin "💰 ფასები" page. Lari (GEL) only for now —
        # add a currency column when we onboard a non-Georgian shop.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pricing_plans (
                id SERIAL PRIMARY KEY,
                feature_set TEXT NOT NULL,
                pricing_tier TEXT NOT NULL,
                price_gel NUMERIC(10, 2),
                setup_fee_gel NUMERIC(10, 2),
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (feature_set, pricing_tier)
            )
        """)
        for fset in ("bot", "site", "combo"):
            for tier in ("start", "grow", "pro"):
                await conn.execute(
                    """INSERT INTO pricing_plans
                       (feature_set, pricing_tier, created_at, updated_at)
                       VALUES ($1, $2, $3, $3)
                       ON CONFLICT (feature_set, pricing_tier) DO NOTHING""",
                    fset, tier, now_iso,
                )

        # ── Phase 2: super-admin audit log ─────────────────────
        # Every "super only" mutation gets a row here so the operator
        # has a paper trail of who created/suspended/impersonated/
        # rotated keys for which tenant. Cannot be edited from the UI;
        # it's append-only by design.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS super_admin_actions (
                id BIGSERIAL PRIMARY KEY,
                super_admin_id INTEGER REFERENCES admin_users(id),
                action TEXT NOT NULL,
                target_tenant_id TEXT,
                payload_json JSONB,
                ip TEXT,
                at TEXT NOT NULL
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_super_actions_at "
            "ON super_admin_actions (at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_super_actions_target "
            "ON super_admin_actions (target_tenant_id)"
        )

        # ── Phase 1: foreign keys from tenant-scoped tables ─────
        # Promote tenant_id from "free-form TEXT" to a real FK pointing
        # at tenants(tenant_id). Catches orphan rows (any tenant_id not
        # in tenants) at boot — if your migration ever fails here,
        # something seeded a tenant_id without a matching tenants row,
        # and the cascade-delete would have left things in a bad state.
        for table in TENANT_SCOPED_TABLES:
            constraint_name = f"fk_{table}_tenant"
            try:
                await conn.execute(
                    f"ALTER TABLE {table} "
                    f"ADD CONSTRAINT {constraint_name} "
                    f"FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) "
                    f"ON DELETE NO ACTION"
                )
            except asyncpg.exceptions.DuplicateObjectError:
                pass  # constraint already in place
            except asyncpg.exceptions.UndefinedTableError:
                pass  # lazy table not created yet
            except asyncpg.exceptions.UndefinedColumnError:
                pass  # tenant_id column not yet added (lazy table)
            except asyncpg.exceptions.ForeignKeyViolationError as e:
                print(
                    f"[migration] WARNING: orphan tenant_id rows in {table}; "
                    f"FK skipped — clean up before relying on referential "
                    f"integrity. Detail: {e}",
                    flush=True,
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
        # Bumped on password change or "logout all devices" — every
        # session token carries the epoch it was issued at, and the
        # middleware refuses any token whose epoch is older than the
        # user's current value. That gives us server-side revocation
        # without a session table.
        await conn.execute(
            "ALTER TABLE admin_users ADD COLUMN IF NOT EXISTS "
            "session_epoch INTEGER NOT NULL DEFAULT 0"
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

        # ── Bot persona config (white-label) ───────────────────
        # Each tenant configures their bot identity here.
        # The system prompt is built dynamically from these fields
        # so a different company can rebrand the bot without touching code.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_configs (
                tenant_id TEXT PRIMARY KEY REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                company_name TEXT NOT NULL DEFAULT '',
                bot_greeting TEXT NOT NULL DEFAULT '',
                product_catalog TEXT NOT NULL DEFAULT '',
                size_guide TEXT NOT NULL DEFAULT '',
                delivery_info TEXT NOT NULL DEFAULT '',
                payment_accounts JSONB NOT NULL DEFAULT '[]',
                currency_symbol TEXT NOT NULL DEFAULT '₾',
                off_topic_reply TEXT NOT NULL DEFAULT '',
                confidential_reply TEXT NOT NULL DEFAULT '',
                unavailable_reply TEXT NOT NULL DEFAULT '',
                custom_order_reply TEXT NOT NULL DEFAULT '',
                corporate_info TEXT NOT NULL DEFAULT '',
                tone TEXT NOT NULL DEFAULT 'friendly',
                emoji_enabled BOOLEAN NOT NULL DEFAULT true,
                custom_instructions TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )
        """)

        # Seed default Tissu config if not present.
        # These values mirror what was previously hardcoded in the system prompt.
        existing_cfg = await conn.fetchval(
            "SELECT tenant_id FROM bot_configs WHERE tenant_id = $1",
            DEFAULT_TENANT_ID,
        )
        if not existing_cfg:
            import json as _json
            tissu_accounts = _json.dumps([
                {"bank_name": "თიბისი", "account_number": "GE58TB7085345064300066"},
                {"bank_name": "საქართველოს ბანკი", "account_number": "GE65BG0000000358364200"},
            ])
            now_iso = datetime.now(timezone.utc).isoformat()
            await conn.execute(
                """INSERT INTO bot_configs
                   (tenant_id, company_name, bot_greeting, product_catalog,
                    size_guide, delivery_info, payment_accounts, currency_symbol,
                    off_topic_reply, confidential_reply, unavailable_reply,
                    custom_order_reply, corporate_info, tone, emoji_enabled,
                    custom_instructions, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)""",
                DEFAULT_TENANT_ID,
                "Tissu Shop",
                "გამარჯობა ✨ Tissu Shop-ის ასისტენტი ვარ. რით შემიძლია დაგეხმაროთ?",
                (
                    "ლეპტოპის ქეისები — ტილოსგან, წყალგაუმტარი, ორმხრივად გამოყენებადი "
                    "(ერთი მხარე ერთფეროვანი, მეორე ჭრელი)\n"
                    "ორი ზომა: პატარა (33x25სმ) — 69₾ | დიდი (37x27სმ) — 74₾\n"
                    "ორი სტილი: ფხრიწიანი | თასმიანი (strap)\n"
                    "ელვა შესაკრავიანი მოდელები ᲐᲠ გვაქვს! მხოლოდ ფხრიწიანი და თასმიანი.\n"
                    "კოდები: FP=ფხრიწიანი პატარა, TP=თასმიანი პატარა, "
                    "FD=ფხრიწიანი დიდი, TD=თასმიანი დიდი\n"
                    "ქართულადაც: ტდ1=TD1, ფპ5=FP5"
                ),
                (
                    "MacBook 13\", 13.3\", 13.6\", 14\" → პატარა ზომა (33x25სმ) მოერგება\n"
                    "MacBook 15\", 15.6\", 16\" → დიდი ზომა (37x27სმ) მოერგება\n"
                    "სხვა ბრენდი → ზომები გადაამოწმოთ სანტიმეტრებში"
                ),
                "6₾ თბილისში. ღამის 12-მდე შეკვეთა → მეორე დღეს (კვირის გარდა). რეგიონებში — ფასი ინდივიდუალური.",
                tissu_accounts,
                "₾",
                "მხოლოდ Tissu Shop-ის შესახებ გვაქვს ინფორმაცია ✨",
                "ეს კონფიდენციალური ინფორმაციაა ✨",
                "სამწუხაროდ ამჟამად არ გვაქვს ✨ ლეპტოპის ქეისები გვაქვს მხოლოდ.",
                "მხოლოდ მზა მოდელებს ვყიდით ✨ ჩვენი დიზაინებია. ქასთომ დიზაინების პრინტს არ ვაკეთებთ.",
                "კორპორატიული შეკვეთები შეგვიძლია (10+ ცალი) ✨ დეტალებისთვის მოგვწერეთ.",
                "friendly",
                True,
                "",
                now_iso,
            )
            print(f"[bot-config-seed] seeded default Tissu bot config", flush=True)

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
                    # First seeded owner on the default tenant gets the
                    # 'super' role — they are the platform operator who
                    # can create / suspend other tenants. Everyone else
                    # is plain 'owner' of their own tenant.
                    await conn.execute(
                        "INSERT INTO admin_users "
                        "(tenant_id, email, password_hash, role, created_at) "
                        "VALUES ($1, $2, $3, 'super', $4)",
                        DEFAULT_TENANT_ID, owner_email, pw_hash,
                        datetime.now(timezone.utc).isoformat(),
                    )
                    print(f"[admin-seed] seeded super-admin {owner_email}", flush=True)

        # One-shot: if the seeded owner on the default tenant is still
        # role='owner' from a previous migration, promote them to
        # 'super'. Keeps a single source of truth for "who is the
        # platform operator" without a separate table.
        await conn.execute(
            "UPDATE admin_users SET role = 'super' "
            "WHERE tenant_id = $1 "
            "AND LOWER(email) = $2 "
            "AND role = 'owner'",
            DEFAULT_TENANT_ID, owner_email or "",
        )


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
    """Return the admin_users row matching ``email`` for a SPECIFIC
    tenant. Used by code paths that already know which tenant they
    care about (e.g. seeding the default-tenant owner on first boot).
    For the login flow, prefer ``find_admin_users_by_email_global`` —
    callers there don't know the tenant up front."""
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


async def find_admin_users_by_email_global(email: str) -> list[dict]:
    """Find every admin_user with this email across all tenants.

    The login form has a single email field — we don't ask the
    customer which shop they belong to. So we search globally and
    let the login handler verify the password against each match
    (in practice there's only one row per email; the loop just
    handles the edge case where two tenants share an owner email).
    """
    if not email:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, tenant_id, email, password_hash, role, "
        "created_at, last_login_at FROM admin_users "
        "WHERE LOWER(email) = LOWER($1) ORDER BY created_at",
        email,
    )
    return [dict(r) for r in rows]


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


async def bump_admin_session_epoch(user_id: int) -> int:
    """Increment session_epoch and return the new value.

    Every active session token carries the epoch it was issued at; the
    middleware refuses any token whose epoch is < the user's current
    value. Bumping therefore invalidates every existing session for
    this user atomically. Used by change-password and logout-all-
    devices."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "UPDATE admin_users SET session_epoch = session_epoch + 1 "
        "WHERE id = $1 RETURNING session_epoch",
        user_id,
    )
    return int(row["session_epoch"]) if row else 0


async def get_admin_session_epoch(user_id: int) -> int:
    """Read the current session_epoch for ``user_id``. Used by the
    middleware on every authenticated request to decide whether the
    presented token is still valid. Returns 0 if the user is missing —
    the caller will reject the token regardless because find_admin_
    user_by_id will already have returned None."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT session_epoch FROM admin_users WHERE id = $1", user_id,
    )
    return int(row["session_epoch"]) if row else 0


async def create_password_reset(
    user_id: int, token_hash: str, ttl_seconds: int = 30 * 60,
) -> None:
    """Record a password-reset token's hash + expiry. Called when the
    user hits POST /admin/forgot-password."""
    from datetime import timedelta
    pool = await get_pool()
    now = datetime.now(timezone.utc)
    await pool.execute(
        "INSERT INTO admin_password_resets "
        "(token_hash, admin_user_id, expires_at, created_at) "
        "VALUES ($1, $2, $3, $4)",
        token_hash, user_id,
        (now + timedelta(seconds=ttl_seconds)).isoformat(),
        now.isoformat(),
    )


async def create_activation_token(
    user_id: int, token_hash: str, ttl_seconds: int = 7 * 24 * 60 * 60,
) -> None:
    """Activation tokens live in the same table as password-reset
    tokens (they do the same thing mechanically: let a user set a
    password). We just give them a 7-day TTL instead of 30 minutes
    because the super-admin sends the link over WhatsApp and the
    customer may take a few days to act on it."""
    await create_password_reset(user_id, token_hash, ttl_seconds=ttl_seconds)


async def consume_password_reset(token_hash: str) -> int | None:
    """Verify a reset token and mark it used. Returns the owning
    admin_user_id on success, None if the token is missing, already
    used, or expired.

    The update + check is done in one SQL statement so a concurrent
    replay of the same token can only succeed once — whichever call
    wins the ``used_at IS NULL`` race updates the row; the other
    walks away with None.
    """
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "UPDATE admin_password_resets SET used_at = $1 "
        "WHERE token_hash = $2 AND used_at IS NULL "
        "AND expires_at > $1 "
        "RETURNING admin_user_id",
        now, token_hash,
    )
    return row["admin_user_id"] if row else None


async def list_tenants() -> list[dict]:
    """Return every tenant with its headline fields + admin_user count.
    Used by the super-admin list page. The legacy ``plan`` field is
    aliased from the new ``pricing_tier`` column so any caller that
    still reads ``plan`` keeps working."""
    pool = await get_pool()
    rows = await pool.fetch("""
        SELECT
            t.tenant_id, t.shop_name, t.owner_email, t.status,
            t.pricing_tier, t.pricing_tier AS plan,
            t.feature_set, t.bot_enabled, t.site_enabled,
            t.suspended_at, t.trial_ends_at,
            t.cloudinary_folder, t.storefront_subdomain,
            t.payment_due_date, t.notes, t.created_at, t.updated_at,
            t.fb_page_id IS NOT NULL AS has_fb,
            (SELECT COUNT(*) FROM admin_users WHERE tenant_id = t.tenant_id) AS admin_count,
            (SELECT COUNT(*) FROM inventory WHERE tenant_id = t.tenant_id) AS product_count,
            (SELECT COUNT(*) FROM orders WHERE tenant_id = t.tenant_id) AS order_count
        FROM tenants t
        ORDER BY t.created_at DESC
    """)
    return [dict(r) for r in rows]


async def get_tenant(tenant_id: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM tenants WHERE tenant_id = $1", tenant_id,
    )
    return dict(row) if row else None


async def get_tenant_by_fb_page_id(page_id: str) -> dict | None:
    """Reverse lookup: which tenant owns this Facebook page? Called by
    the /webhook dispatcher on every incoming Messenger/IG event."""
    if not page_id:
        return None
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM tenants WHERE fb_page_id = $1", page_id,
    )
    return dict(row) if row else None


async def delete_tenant_cascade(tenant_id: str) -> dict:
    """Wipe an entire tenant and every row tied to it.

    The platform operator calls this from /admin/super when a customer
    churns. Refuses to nuke the default tenant — that's where the
    super-admin's own account lives, and it carries the bootstrap
    api_keys row.

    Runs inside a transaction so a partial failure rolls back. The
    "lazy" tables (product_pairs, ai_photo_hints, product_embeddings,
    product_fingerprints) are wrapped in try/except because they may
    not exist on every install.

    Returns a dict of {table: rows_deleted} so the UI can show a
    receipt — useful when the operator wants to verify the cleanup
    landed.
    """
    if tenant_id == DEFAULT_TENANT_ID:
        raise ValueError("cannot delete the default tenant")
    if not tenant_id:
        raise ValueError("tenant_id required")

    pool = await get_pool()
    receipt: dict[str, int] = {}

    # The tables we always touch. Order doesn't matter inside the CTE
    # because the data-modifying CTEs all see the same snapshot — but
    # FK-dependent rows still need to be removed before their parents,
    # so we split into two batches.
    fk_dependent_sql = """
        WITH d_msg AS (
            DELETE FROM messages WHERE tenant_id = $1 RETURNING 1
        ),
        d_conv AS (
            DELETE FROM conversations WHERE tenant_id = $1 RETURNING 1
        ),
        d_pwreset AS (
            DELETE FROM admin_password_resets
             WHERE admin_user_id IN (
                 SELECT id FROM admin_users WHERE tenant_id = $1
             )
             RETURNING 1
        ),
        d_admin AS (
            DELETE FROM admin_users WHERE tenant_id = $1 RETURNING 1
        )
        SELECT
            (SELECT COUNT(*) FROM d_msg)     AS messages,
            (SELECT COUNT(*) FROM d_conv)    AS conversations,
            (SELECT COUNT(*) FROM d_pwreset) AS admin_password_resets,
            (SELECT COUNT(*) FROM d_admin)   AS admin_users
    """

    flat_sql = """
        WITH
            d_inv  AS (DELETE FROM inventory             WHERE tenant_id = $1 RETURNING 1),
            d_ord  AS (DELETE FROM orders                WHERE tenant_id = $1 RETURNING 1),
            d_lead AS (DELETE FROM leads                 WHERE tenant_id = $1 RETURNING 1),
            d_tic  AS (DELETE FROM tickets               WHERE tenant_id = $1 RETURNING 1),
            d_cnt  AS (DELETE FROM content               WHERE tenant_id = $1 RETURNING 1),
            d_kb   AS (DELETE FROM knowledge_base        WHERE tenant_id = $1 RETURNING 1),
            d_cat  AS (DELETE FROM categories            WHERE tenant_id = $1 RETURNING 1),
            d_pep  AS (DELETE FROM product_extra_photos  WHERE tenant_id = $1 RETURNING 1),
            d_ct   AS (DELETE FROM confirm_tokens        WHERE tenant_id = $1 RETURNING 1),
            d_keys AS (DELETE FROM api_keys              WHERE tenant_id = $1 RETURNING 1),
            d_ten  AS (DELETE FROM tenants               WHERE tenant_id = $1 RETURNING 1)
        SELECT
            (SELECT COUNT(*) FROM d_inv)  AS inventory,
            (SELECT COUNT(*) FROM d_ord)  AS orders,
            (SELECT COUNT(*) FROM d_lead) AS leads,
            (SELECT COUNT(*) FROM d_tic)  AS tickets,
            (SELECT COUNT(*) FROM d_cnt)  AS content,
            (SELECT COUNT(*) FROM d_kb)   AS knowledge_base,
            (SELECT COUNT(*) FROM d_cat)  AS categories,
            (SELECT COUNT(*) FROM d_pep)  AS product_extra_photos,
            (SELECT COUNT(*) FROM d_ct)   AS confirm_tokens,
            (SELECT COUNT(*) FROM d_keys) AS api_keys,
            (SELECT COUNT(*) FROM d_ten)  AS tenants
    """

    async with pool.acquire() as conn:
        async with conn.transaction():
            row1 = await conn.fetchrow(fk_dependent_sql, tenant_id)
            for k, v in dict(row1).items():
                receipt[k] = int(v or 0)

            row2 = await conn.fetchrow(flat_sql, tenant_id)
            for k, v in dict(row2).items():
                receipt[k] = int(v or 0)

            # Lazy / optional tables — these may not exist on every
            # install, so each gets its own try/except. We still
            # batch each one as a CTE so it costs one round-trip.
            for table in (
                "product_pairs", "product_embeddings",
                "product_fingerprints", "ai_photo_hints",
            ):
                try:
                    n = await conn.fetchval(
                        f"WITH d AS (DELETE FROM {table} WHERE tenant_id = $1 RETURNING 1) "
                        "SELECT COUNT(*) FROM d",
                        tenant_id,
                    )
                    receipt[table] = int(n or 0)
                except asyncpg.exceptions.UndefinedTableError:
                    pass
                except asyncpg.exceptions.UndefinedColumnError:
                    pass

    return receipt


async def update_tenant_fb_credentials(
    tenant_id: str,
    fb_page_id: str,
    fb_page_token_encrypted: str,
    fb_verify_token: str,
) -> None:
    """Store a tenant's Facebook webhook credentials — called from the
    super-admin form when onboarding. fb_page_token MUST arrive
    already encrypted via ``src.secrets_vault.encrypt_secret``."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE tenants SET "
        "  fb_page_id = $1, "
        "  fb_page_token_encrypted = $2, "
        "  fb_verify_token = $3, "
        "  updated_at = $4 "
        "WHERE tenant_id = $5",
        fb_page_id.strip() or None,
        fb_page_token_encrypted or None,
        fb_verify_token.strip() or None,
        datetime.now(timezone.utc).isoformat(),
        tenant_id,
    )


async def create_tenant(
    tenant_id: str,
    shop_name: str,
    owner_email: str,
    pricing_tier: str = "start",
    feature_set: str = "bot",
    status: str = "trial",
) -> None:
    """Insert a new tenant row. The two plan axes are independent:
    feature_set picks bot/site/combo, pricing_tier picks the price
    band. bot_enabled and site_enabled are derived from feature_set
    here so callers don't have to keep them in sync — they can still
    be flipped per-tenant later via update_tenant_features().
    cloudinary_folder defaults to the tenant_id so the Phase 4 CMS
    can write `tissu/<tenant_id>/...` without collision.

    A new tenant starts on a TRIAL_DURATION_DAYS-long trial — after
    that the auth middleware auto-suspends them until the operator
    presses "გადახდა მიიღე" (mark-paid).
    """
    from datetime import timedelta as _td
    if feature_set not in ("bot", "site", "combo"):
        raise ValueError(f"feature_set must be bot/site/combo (got {feature_set!r})")
    if pricing_tier not in ("start", "grow", "pro"):
        raise ValueError(f"pricing_tier must be start/grow/pro (got {pricing_tier!r})")
    bot_enabled = feature_set in ("bot", "combo")
    site_enabled = feature_set in ("site", "combo")
    pool = await get_pool()
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    trial_ends = (
        (now_dt + _td(days=TRIAL_DURATION_DAYS)).isoformat()
        if status == "trial" else None
    )
    await pool.execute(
        """INSERT INTO tenants
           (tenant_id, shop_name, owner_email, status, pricing_tier,
            feature_set, bot_enabled, site_enabled, cloudinary_folder,
            trial_ends_at, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $1, $9, $10, $10)""",
        tenant_id, shop_name, owner_email.strip().lower(),
        status, pricing_tier, feature_set, bot_enabled, site_enabled,
        trial_ends, now,
    )


async def update_tenant_features(
    tenant_id: str,
    bot_enabled: bool | None = None,
    site_enabled: bool | None = None,
) -> None:
    """Flip individual feature flags without changing the headline
    feature_set. Useful for "this tenant's bot has temporary issues,
    turn it off but keep the subscription". Pass None to leave a
    flag alone."""
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    if bot_enabled is None and site_enabled is None:
        return
    sets = []
    params: list = []
    idx = 1
    if bot_enabled is not None:
        sets.append(f"bot_enabled = ${idx}")
        params.append(bool(bot_enabled))
        idx += 1
    if site_enabled is not None:
        sets.append(f"site_enabled = ${idx}")
        params.append(bool(site_enabled))
        idx += 1
    sets.append(f"updated_at = ${idx}")
    params.append(now)
    idx += 1
    params.append(tenant_id)
    await pool.execute(
        f"UPDATE tenants SET {', '.join(sets)} WHERE tenant_id = ${idx}",
        *params,
    )


async def update_tenant_status(
    tenant_id: str,
    status: str,
    payment_due_date: str | None = None,
) -> None:
    """Flip a tenant's status (trial / active / suspended) and keep
    suspended_at synced. Going to 'active' also clears
    ``trial_ends_at`` — once paid, the trial deadline is irrelevant
    and we don't want it to come back to bite us if the tenant later
    flips back to 'trial' for any reason."""
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    suspended_at = now if status == "suspended" else None
    if status == "active":
        if payment_due_date is None:
            await pool.execute(
                "UPDATE tenants SET status = $1, suspended_at = NULL, "
                "                   trial_ends_at = NULL, updated_at = $2 "
                "WHERE tenant_id = $3",
                status, now, tenant_id,
            )
        else:
            await pool.execute(
                "UPDATE tenants SET status = $1, suspended_at = NULL, "
                "                   trial_ends_at = NULL, "
                "                   payment_due_date = $2, updated_at = $3 "
                "WHERE tenant_id = $4",
                status, payment_due_date, now, tenant_id,
            )
        return
    # Non-active transitions keep trial_ends_at untouched (suspending
    # a trial tenant before the deadline shouldn't lose the
    # deadline; resuming with mark-paid is the only way out).
    if payment_due_date is None:
        await pool.execute(
            "UPDATE tenants SET status = $1, suspended_at = $2, "
            "                   updated_at = $3 "
            "WHERE tenant_id = $4",
            status, suspended_at, now, tenant_id,
        )
    else:
        await pool.execute(
            "UPDATE tenants SET status = $1, suspended_at = $2, "
            "                   payment_due_date = $3, updated_at = $4 "
            "WHERE tenant_id = $5",
            status, suspended_at, payment_due_date, now, tenant_id,
        )


async def is_bot_enabled(tenant_id: str) -> bool:
    """True iff the tenant currently has bot access. Used by
    request-time entitlement checks (Phase 3)."""
    pool = await get_pool()
    row = await pool.fetchval(
        "SELECT bot_enabled FROM tenants WHERE tenant_id = $1", tenant_id,
    )
    return bool(row) if row is not None else False


async def is_site_enabled(tenant_id: str) -> bool:
    """True iff the tenant currently has website CMS access."""
    pool = await get_pool()
    row = await pool.fetchval(
        "SELECT site_enabled FROM tenants WHERE tenant_id = $1", tenant_id,
    )
    return bool(row) if row is not None else False


async def log_super_action(
    super_admin_id: int | None,
    action: str,
    target_tenant_id: str | None = None,
    payload: dict | None = None,
    ip: str | None = None,
) -> None:
    """Append a row to ``super_admin_actions`` for the audit log.
    NEVER pass raw secrets in ``payload`` — token previews / hashes
    only. The caller is responsible for redacting before calling this.
    """
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "INSERT INTO super_admin_actions "
        "(super_admin_id, action, target_tenant_id, payload_json, ip, at) "
        "VALUES ($1, $2, $3, $4::jsonb, $5, $6)",
        super_admin_id, action, target_tenant_id,
        json.dumps(payload) if payload else None,
        ip, now,
    )


async def list_super_actions(limit: int = 100) -> list[dict]:
    """Return the most recent audit rows for the /admin/super/audit
    page. Joins on admin_users + tenants so the UI doesn't need a
    second round-trip per row."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT a.id, a.action, a.target_tenant_id, a.payload_json, "
        "       a.ip, a.at, "
        "       u.email AS super_admin_email, "
        "       t.shop_name AS target_shop_name "
        "FROM super_admin_actions a "
        "LEFT JOIN admin_users u ON u.id = a.super_admin_id "
        "LEFT JOIN tenants t ON t.tenant_id = a.target_tenant_id "
        "ORDER BY a.at DESC "
        "LIMIT $1",
        limit,
    )
    out = []
    for r in rows:
        d = dict(r)
        # asyncpg returns JSONB as a Python dict already (or str depending
        # on version) — normalize to dict.
        pj = d.get("payload_json")
        if isinstance(pj, str):
            try:
                d["payload_json"] = json.loads(pj)
            except Exception:
                d["payload_json"] = None
        out.append(d)
    return out


async def regenerate_tenant_api_key(tenant_id: str) -> str:
    """Delete the tenant's existing api_keys row(s) and seed a fresh
    one. Returns the new raw key (only chance the caller has to read
    it — we never store the raw, just record the row in api_keys)."""
    import secrets as _secrets
    new_key = f"tissu_admin_{_secrets.token_urlsafe(24)}"
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM api_keys WHERE tenant_id = $1 AND scope = 'admin'",
                tenant_id,
            )
            await conn.execute(
                "INSERT INTO api_keys (key, tenant_id, label, scope, created_at) "
                "VALUES ($1, $2, $3, 'admin', $4)",
                new_key, tenant_id, "rotated-admin", now,
            )
    return new_key


async def list_pricing_plans() -> list[dict]:
    """Return all 9 pricing rows for the super-admin pricing page."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, feature_set, pricing_tier, price_gel, "
        "       setup_fee_gel, description, updated_at "
        "FROM pricing_plans "
        "ORDER BY feature_set, "
        "         CASE pricing_tier "
        "              WHEN 'start' THEN 0 "
        "              WHEN 'grow'  THEN 1 "
        "              WHEN 'pro'   THEN 2 "
        "              ELSE 9 END"
    )
    out = []
    for r in rows:
        d = dict(r)
        # NUMERIC columns come back as Decimal — JSON-encodable as str.
        for k in ("price_gel", "setup_fee_gel"):
            v = d.get(k)
            if v is not None:
                d[k] = float(v)
        out.append(d)
    return out


async def update_pricing_plan(
    feature_set: str,
    pricing_tier: str,
    price_gel: float | None,
    setup_fee_gel: float | None,
    description: str | None,
) -> None:
    """Owner edits a cell in the pricing matrix. Each call updates
    one (feature_set, pricing_tier) row — pass None to clear a value."""
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE pricing_plans SET "
        "  price_gel = $1, setup_fee_gel = $2, description = $3, updated_at = $4 "
        "WHERE feature_set = $5 AND pricing_tier = $6",
        price_gel, setup_fee_gel, description, now,
        feature_set, pricing_tier,
    )


async def create_admin_user_pending(
    tenant_id: str, email: str, role: str = "owner",
) -> int:
    """Create an admin_user with no usable password yet — they'll set
    one via the activation link. Returns the new user's id."""
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    # Use a sentinel hash that no argon2 verify can match. We mark it
    # with a prefix so "pending" is visible in the DB at a glance.
    sentinel = "pending:set-on-activation:" + now
    row = await pool.fetchrow(
        "INSERT INTO admin_users "
        "(tenant_id, email, password_hash, role, created_at) "
        "VALUES ($1, $2, $3, $4, $5) RETURNING id",
        tenant_id, email.strip().lower(), sentinel, role, now,
    )
    return row["id"]


async def invalidate_all_password_resets_for(user_id: int) -> None:
    """Mark every unused reset token for a user as used. Called after
    a successful reset so any other outstanding links in the user's
    inbox stop working."""
    pool = await get_pool()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE admin_password_resets SET used_at = $1 "
        "WHERE admin_user_id = $2 AND used_at IS NULL",
        now, user_id,
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


async def get_bot_config(tenant_id: str) -> dict | None:
    """Return the bot_configs row for ``tenant_id``, or None if not found."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM bot_configs WHERE tenant_id = $1", tenant_id,
    )
    if not row:
        return None
    result = dict(row)
    import json as _json
    if isinstance(result.get("payment_accounts"), str):
        try:
            result["payment_accounts"] = _json.loads(result["payment_accounts"])
        except Exception:
            result["payment_accounts"] = []
    return result


async def upsert_bot_config(tenant_id: str, data: dict) -> None:
    """Insert or update the bot_configs row for ``tenant_id``.
    Only keys present in ``data`` are updated — missing keys keep
    their current DB value (partial update semantics)."""
    pool = await get_pool()
    import json as _json
    now = datetime.now(timezone.utc).isoformat()

    existing = await pool.fetchval(
        "SELECT tenant_id FROM bot_configs WHERE tenant_id = $1", tenant_id,
    )
    payment_accounts = data.get("payment_accounts")
    if payment_accounts is not None and not isinstance(payment_accounts, str):
        payment_accounts = _json.dumps(payment_accounts)

    if not existing:
        await pool.execute(
            """INSERT INTO bot_configs
               (tenant_id, company_name, bot_greeting, product_catalog,
                size_guide, delivery_info, payment_accounts, currency_symbol,
                off_topic_reply, confidential_reply, unavailable_reply,
                custom_order_reply, corporate_info, tone, emoji_enabled,
                custom_instructions, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)""",
            tenant_id,
            data.get("company_name", ""),
            data.get("bot_greeting", ""),
            data.get("product_catalog", ""),
            data.get("size_guide", ""),
            data.get("delivery_info", ""),
            payment_accounts or "[]",
            data.get("currency_symbol", "₾"),
            data.get("off_topic_reply", ""),
            data.get("confidential_reply", ""),
            data.get("unavailable_reply", ""),
            data.get("custom_order_reply", ""),
            data.get("corporate_info", ""),
            data.get("tone", "friendly"),
            bool(data.get("emoji_enabled", True)),
            data.get("custom_instructions", ""),
            now,
        )
    else:
        fields = []
        params: list = []
        idx = 1
        for col in (
            "company_name", "bot_greeting", "product_catalog", "size_guide",
            "delivery_info", "currency_symbol", "off_topic_reply",
            "confidential_reply", "unavailable_reply", "custom_order_reply",
            "corporate_info", "tone", "emoji_enabled", "custom_instructions",
        ):
            if col in data:
                fields.append(f"{col} = ${idx}")
                params.append(data[col])
                idx += 1
        if payment_accounts is not None:
            fields.append(f"payment_accounts = ${idx}")
            params.append(payment_accounts)
            idx += 1
        fields.append(f"updated_at = ${idx}")
        params.append(now)
        idx += 1
        params.append(tenant_id)
        await pool.execute(
            f"UPDATE bot_configs SET {', '.join(fields)} WHERE tenant_id = ${idx}",
            *params,
        )
