"""Public read API for the Tissu website (and any other external storefront).

The Tissu storefront pulls its product grid from here. The API key that
comes in via ``X-API-Key`` resolves to a tenant (via the api_keys table)
and the endpoints only return products owned by that tenant — so when a
second shop plugs their own site into this backend, each storefront
automatically sees only its own catalog.

The response shape is the source of truth and the website depends on it
verbatim. Keep changes backward-compatible: add new optional fields
rather than renaming or removing existing ones.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.db import DEFAULT_TENANT_ID, get_db


router = APIRouter(prefix="/api/storefront", tags=["storefront"])

# The Tissu website renders six sections. We map every internal category
# slug into one of these six so the frontend can filter by stable names
# without caring about legacy slugs. Unknown categories pass through
# unchanged — the storefront falls back to a "Other" section for those.
CATEGORY_ALIASES: dict[str, str] = {
    "bag": "pouch",          # Tissu bags are laptop pouches
    "pouch": "pouch",
    "laptop": "laptop",
    "tote": "tote",
    "kidsbackpack": "kidsbackpack",
    "apron": "apron",
    "necklace": "necklace",
}

PUBLIC_CATEGORIES: frozenset[str] = frozenset(CATEGORY_ALIASES.values())

STOREFRONT_CACHE = "s-maxage=60, stale-while-revalidate=300"
CURRENCY = "GEL"


def _tenant_id(request: Request) -> str:
    """Every /api/* request comes through APIKeyMiddleware which stashes
    the resolved tenant on request.state. Fall back to the default
    tenant for defensiveness — the middleware always sets this."""
    return getattr(request.state, "tenant_id", DEFAULT_TENANT_ID)


def _map_category(slug: str | None) -> str:
    """Normalize an internal category slug to the public enum. Unknown
    slugs pass through so new categories added by the owner still surface
    to the storefront without a code change."""
    if not slug:
        return "pouch"
    return CATEGORY_ALIASES.get(slug, slug)


def _parse_tags(raw: Any) -> list[str]:
    """Inventory.tags is a free-form text column — in practice either a
    comma-separated string, a JSON array, or empty. Normalize to a list
    of trimmed, non-empty strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # Try JSON first — handles '["new", "sale"]'.
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
    # Fallback: comma-separated.
    return [t.strip() for t in s.split(",") if t.strip()]


def _effective_price(row: dict) -> float:
    """Return the price the customer actually pays: sale_price when the
    product is on sale, otherwise the regular price."""
    if row.get("on_sale") and row.get("sale_price"):
        try:
            sp = float(row["sale_price"])
            if sp > 0:
                return sp
        except (TypeError, ValueError):
            pass
    try:
        return float(row.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0


def _serialize(row: dict) -> dict:
    """Turn an inventory row into the public response shape."""
    stock = int(row.get("stock") or 0)
    effective = _effective_price(row)
    # original_price exposes the pre-discount number so the storefront
    # can render a strike-through. Null when the product isn't on sale
    # (so the UI can simply check for null to decide whether to render).
    original_price = None
    if row.get("on_sale"):
        try:
            base = float(row.get("price") or 0)
            if base > 0 and base != effective:
                original_price = base
        except (TypeError, ValueError):
            pass
    description = row.get("description")
    if description is not None:
        description = str(description).strip() or None
    return {
        "id": str(row["id"]),
        "code": row.get("code") or "",
        "name": row.get("product_name") or "",
        "model": row.get("model") or "",
        "size": row.get("size") or "",
        "color": row.get("color") or "",
        "description": description,
        "price": effective,
        "original_price": original_price,
        "currency": CURRENCY,
        "stock": stock,
        "in_stock": stock > 0,
        "image_front": row.get("image_url") or "",
        "image_back": row.get("image_url_back") or "",
        "category": _map_category(row.get("category")),
        "tags": _parse_tags(row.get("tags")),
        "updated_at": row.get("updated_at") or row.get("created_at") or "",
    }


@router.get("/health")
async def storefront_health(response: Response):
    """Cheap liveness check the storefront can hit without a DB round-trip.
    Returns no tenant-scoped data so it's safe to cache aggressively at
    the edge."""
    response.headers["Cache-Control"] = STOREFRONT_CACHE
    return {"ok": True}


@router.get("/products")
async def list_products(
    request: Request,
    response: Response,
    include_out_of_stock: bool = False,
    category: str = "",
    tenant_id: str = Depends(_tenant_id),
):
    """Return every product available on the storefront for this tenant.

    Defaults to in-stock only; pass ``?include_out_of_stock=true`` to
    include sold-out rows (useful for a "back-in-soon" section). The
    optional ``?category=`` filter uses the *public* enum (pouch,
    laptop, …), not the internal slug.
    """
    pool = await get_db()

    sql = "SELECT * FROM inventory WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if not include_out_of_stock:
        sql += " AND stock > 0"
    if category:
        # Translate the public enum back to the internal slug(s) so
        # tenant-specific legacy slugs (e.g. 'bag' → 'pouch') match.
        internal_slugs = [
            slug for slug, public in CATEGORY_ALIASES.items()
            if public == category
        ] or [category]
        sql += f" AND category = ANY(${idx}::text[])"
        params.append(internal_slugs)
        idx += 1
    sql += " ORDER BY category, code, id"

    rows = await pool.fetch(sql, *params)
    products = [_serialize(dict(r)) for r in rows]
    response.headers["Cache-Control"] = STOREFRONT_CACHE
    return {"products": products, "count": len(products)}


@router.get("/products/{product_id}")
async def get_product(
    product_id: str,
    request: Request,
    response: Response,
    tenant_id: str = Depends(_tenant_id),
):
    """Return a single product by its stable id. 404s across tenants so a
    storefront operator can't probe other shops' inventory by guessing
    numeric ids."""
    try:
        id_int = int(product_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found")

    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT * FROM inventory WHERE id = $1 AND tenant_id = $2",
        id_int, tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="not found")

    response.headers["Cache-Control"] = STOREFRONT_CACHE
    return _serialize(dict(row))
