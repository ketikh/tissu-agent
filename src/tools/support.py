"""Tools for Tissu Shop Support + Sales agent."""
from __future__ import annotations

import os
import json
from typing import Any

import httpx
from datetime import datetime, timezone
from src.db import get_db
from src.engine import Tool


async def check_inventory(model: str = "", size: str = "", search: str = "", on_sale: bool = False) -> dict:
    """Check what's in stock. Can filter by model, size, tags, or sale status.

    When on_sale=True we return only products that the owner has marked as
    discounted in the admin panel — used when a customer asks whether we
    have any current promotions.
    """
    pool = await get_db()
    # Bag-only guard: the master bot is currently bag-aware only. We never
    # return necklaces (or any future category) from this tool so the bot
    # can't accidentally recommend a non-bag product to a customer asking
    # about ჩანთები.
    query = "SELECT * FROM inventory WHERE stock > 0 AND category = 'bag'"
    params = []
    idx = 1
    if on_sale:
        query += " AND on_sale = true"
    if model:
        query += f" AND model ILIKE ${idx}"
        params.append(f"%{model}%")
        idx += 1
    if size:
        query += f" AND size ILIKE ${idx}"
        params.append(f"%{size}%")
        idx += 1
    if search:
        # Check if search looks like product code(s) — single (FP1) or multiple (FP1,TP3,FD2)
        search_upper = search.strip().upper()
        # Multiple codes separated by comma
        if "," in search_upper:
            codes = [c.strip() for c in search_upper.split(",") if c.strip()]
            placeholders = ",".join(f"${idx + i}" for i in range(len(codes)))
            query += f" AND UPPER(code) IN ({placeholders})"
            params.extend(codes)
            idx += len(codes)
        elif any(search_upper.startswith(p) for p in ("FP", "TP", "FD", "TD")) and any(c.isdigit() for c in search_upper):
            query += f" AND UPPER(code) = ${idx}"
            params.append(search_upper)
            idx += 1
        else:
            query += f" AND (tags ILIKE ${idx} OR color ILIKE ${idx+1} OR style ILIKE ${idx+2})"
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
            idx += 3
    rows = await pool.fetch(query, *params)
    if not rows:
        return {"found": False, "message": "ვერ მოიძებნა."}
    items = []
    for r in rows:
        row = dict(r)
        item = {
            "code": row.get("code", ""),
            "model": row["model"],
            "size": row["size"],
            "price": row["price"],
        }
        if row.get("on_sale"):
            item["on_sale"] = True
            sp = row.get("sale_price")
            if sp is not None and sp > 0:
                item["sale_price"] = sp
        if row.get("image_url"):
            item["image_url"] = row["image_url"]
        if row.get("image_url_back"):
            item["image_url_back"] = row["image_url_back"]
        items.append(item)
    return {"found": True, "items": items, "count": len(items)}


async def save_lead(name: str, phone: str = "", notes: str = "", score: int = 0, conversation_id: str = "") -> dict:
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO leads (name, phone, source, score, notes, conversation_id, created_at, updated_at) VALUES ($1, $2, 'messenger', $3, $4, $5, $6, $7) RETURNING id",
        name, phone, score, notes, conversation_id, now, now,
    )
    return {"success": True, "lead_id": row["id"]}


async def create_order(customer_name: str, customer_phone: str, customer_address: str, items: str, total: float, payment_method: str = "", notes: str = "") -> dict:
    import re as _re
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO orders (customer_name, customer_phone, customer_address, items, total, payment_method, notes, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
        customer_name, customer_phone, customer_address, items, total, payment_method, notes, now, now,
    )
    order_id = row["id"]

    # Stock stays untouched here — the owner adjusts it manually from the admin
    # panel when changing order status, so auto-decrement would double-count.
    codes = sorted(set(c.upper() for c in _re.findall(r'(?i)(?:FP|TP|FD|TD)\d+', items)))

    from src.notifications import notify_owner_text
    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    parts = [
        f"🛒 ახალი შეკვეთა #{order_id}!",
        f"👤 {customer_name}",
        f"📱 {customer_phone}",
        f"📍 {customer_address}",
    ]
    if codes:
        parts.append(f"🏷️ {', '.join(codes)}")
    parts.extend([
        f"📦 {items}",
        f"💰 {total}₾",
        "",
        f"📋 ადმინ პანელი:\n{public_url}/admin",
    ])
    await notify_owner_text("\n".join(parts))

    return {
        "success": True,
        "order_id": order_id,
        "codes": codes,
    }


def _extract_product_code(text: str) -> str:
    """Find a product code such as TD3 / ფდ3 in free text; '' if none.
    Georgian letters that spell model/size codes are transliterated."""
    ka_to_lat = {"ტ": "T", "ფ": "F", "პ": "P", "დ": "D"}
    for word in text.split():
        word_clean = word.strip(".,!?\"'()[]")
        if not word_clean:
            continue
        converted = "".join(ka_to_lat.get(ch, ch) for ch in word_clean).upper()
        if (len(converted) <= 5
                and any(c.isalpha() for c in converted)
                and any(c.isdigit() for c in converted)):
            return converted
    return ""


async def _load_product_image_bytes(pool: Any, product_code: str) -> bytes | None:
    """Return the product's main photo as bytes (Cloudinary URL or local /static path)."""
    row = await pool.fetchrow("SELECT image_url FROM inventory WHERE UPPER(code) = $1", product_code)
    image_url = (row["image_url"] if row else "") or ""
    if not image_url:
        return None
    try:
        if image_url.startswith("http"):
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(image_url)
                return resp.content if resp.status_code == 200 else None
        if image_url.startswith("/static/"):
            static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
            path = os.path.join(static_dir, image_url.replace("/static/", "", 1))
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
    except (httpx.HTTPError, OSError) as e:
        print(f"[NOTIFY] product image load failed for {product_code}: {e}", flush=True)
    return None


async def _build_owner_message(
    reason: str, customer_name: str, customer_phone: str, details: str, conversation_id: str,
) -> str:
    """Compose the owner's notification. Order/payment reasons get one-time
    confirm/deny links; every notification ends with the admin panel link."""
    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    parts = [f"🔔 {reason}"]
    if customer_name:
        parts.append(f"👤 {customer_name}")
    if customer_phone:
        parts.append(f"📱 {customer_phone}")
    if details:
        parts.append(f"📝 {details}")
    needs_confirm = any(kw in reason.lower() for kw in ("შეკვეთა", "გადახდა", "ჩარიცხ", "order", "payment"))
    if conversation_id and needs_confirm:
        from src.webhooks.whatsapp import build_confirm_url, create_confirm_token
        confirm_url = build_confirm_url("owner-confirm", await create_confirm_token(conversation_id, "owner-confirm"))
        deny_url = build_confirm_url("owner-deny", await create_confirm_token(conversation_id, "owner-deny"))
        parts.extend(["", f"✅ ვადასტურებ:\n{confirm_url}", f"❌ არ ვადასტურებ:\n{deny_url}"])
    parts.extend(["", f"📋 ადმინ პანელი:\n{public_url}/admin"])
    return "\n".join(parts)


async def notify_owner(reason: str, customer_name: str = "", customer_phone: str = "", details: str = "", conversation_id: str = "") -> dict:
    """Notify the shop owner (Telegram first, WhatsApp fallback) and log a ticket.

    The ticket is written first so the request shows up in the admin panel
    even if every messaging channel is down."""
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    notification = {"reason": reason, "customer_name": customer_name, "customer_phone": customer_phone, "details": details}
    await pool.execute(
        "INSERT INTO tickets (subject, description, status, priority, customer_email, conversation_id, created_at, updated_at) VALUES ($1, $2, 'open', 'urgent', $3, $4, $5, $6)",
        f"[NOTIFICATION] {reason}", json.dumps(notification, ensure_ascii=False), customer_phone, conversation_id, now, now,
    )

    from src.notifications import notify_owner_image, notify_owner_text

    product_code = _extract_product_code(f"{details} {reason}")
    if product_code:
        image_bytes = await _load_product_image_bytes(pool, product_code)
        if image_bytes:
            await notify_owner_image(image_bytes, caption=f"📌 {product_code}")

    message = await _build_owner_message(reason, customer_name, customer_phone, details, conversation_id)
    try:
        sent = await notify_owner_text(message)
    except Exception as e:  # a messaging failure must never break the customer's reply
        print(f"[NOTIFY] owner text failed: {type(e).__name__}: {e}", flush=True)
        sent = False
    if not sent:
        print(f"[NOTIFY] owner NOT reached — ticket logged only: {reason[:60]}", flush=True)
    return {
        "notified": sent,
        "message": "მფლობელს ეცნობა" if sent else "მფლობელს შეტყობინება ვერ მიუვიდა, მოთხოვნა ადმინ პანელში ჩაიწერა",
    }


async def search_knowledge(query: str) -> dict:
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT question, answer, category FROM knowledge_base WHERE question ILIKE $1 OR answer ILIKE $2 LIMIT 5",
        f"%{query}%", f"%{query}%",
    )
    if not rows:
        return {"found": False}
    return {"found": True, "results": [dict(r) for r in rows]}


# Pending customer photos: conversation_id -> image_bytes
_pending_photos: dict[str, bytes] = {}
# AI match hints: conversation_id -> hint text
_ai_hints: dict[str, str] = {}


async def forward_photo_to_owner(size: str, conversation_id: str = "") -> dict:
    """Forward customer's pending photo to owner via WhatsApp with AI match + confirm/deny."""
    from src.notifications import notify_owner_image

    print(f"[PHOTO] forward_photo_to_owner called: size={size}, conv_id={conversation_id}")
    print(f"[PHOTO] Pending photos keys: {list(_pending_photos.keys())}")

    photo_bytes = _pending_photos.get(conversation_id)
    if not photo_bytes:
        print(f"[PHOTO] ERROR: Photo NOT FOUND for conversation_id={conversation_id}")
        return {"forwarded": False, "message": "ფოტო ვერ მოიძებნა"}

    print(f"[PHOTO] Photo found: {len(photo_bytes)} bytes")

    from src.webhooks.whatsapp import build_confirm_url, create_confirm_token
    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    confirm_url = build_confirm_url("photo-confirm", await create_confirm_token(conversation_id, "photo-confirm"))
    deny_url = build_confirm_url("photo-deny", await create_confirm_token(conversation_id, "photo-deny"))
    admin_url = f"{public_url}/admin"

    # Extract AI hint (now structured dict)
    ai_data = _ai_hints.pop(conversation_id, None)
    ai_hint_text = ""
    ai_product_url = ""
    if isinstance(ai_data, dict):
        ai_hint_text = ai_data.get("text", "")
        ai_product_url = ai_data.get("image_url", "")
    elif isinstance(ai_data, str):
        ai_hint_text = ai_data

    # 1. Send customer's photo with AI recommendation + confirm/deny
    caption = (
        f"📷 კლიენტი ეძებს ამ მოდელს, {size} ზომაში.{ai_hint_text}\n\n"
        f"✅ გვაქვს:\n{confirm_url}\n\n"
        f"❌ არ გვაქვს:\n{deny_url}\n\n"
        f"📋 ადმინ პანელი:\n{admin_url}"
    )
    sent = await notify_owner_image(photo_bytes, caption=caption)
    print(f"[PHOTO] owner photo sent: {sent}")

    # 2. If AI matched, also send the AI-recommended product photo so owner can visually compare
    if ai_product_url:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(ai_product_url)
                if resp.status_code == 200:
                    await notify_owner_image(
                        resp.content,
                        caption=f"🤖 AI-ის რეკომენდაცია ↑ შეადარე კლიენტის ფოტოს",
                    )
                    print(f"[PHOTO] AI recommendation photo also sent")
        except Exception as e:
            print(f"[PHOTO] Failed to send AI recommendation photo: {e}")

    # Clean up
    _pending_photos.pop(conversation_id, None)

    return {"forwarded": sent, "message": "მფლობელს გადაეგზავნა" if sent else "მფლობელს გაგზავნა ვერ მოხერხდა"}


SUPPORT_TOOLS = [
    Tool(
        name="check_inventory",
        description="მარაგის შემოწმება. ფილტრავს მოდელით, ზომით, ან ტეგებით/ფერით. on_sale=true — მხოლოდ ფასდაკლებული პროდუქტები.",
        parameters={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "'ფხრიწიანი' ან 'თასმიანი'"},
                "size": {"type": "string", "description": "'პატარა' ან 'დიდი'"},
                "search": {"type": "string", "description": "ძიება ფერით ან აღწერით"},
                "on_sale": {"type": "boolean", "description": "true — მხოლოდ ფასდაკლებული პროდუქტები (აქცია)"},
            },
            "required": [],
        },
        handler=check_inventory,
    ),
    Tool(
        name="save_lead",
        description="პოტენციური მყიდველის შენახვა.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
                "notes": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["name"],
        },
        handler=save_lead,
    ),
    Tool(
        name="create_order",
        description="შეკვეთის შექმნა.",
        parameters={
            "type": "object",
            "properties": {
                "customer_name": {"type": "string"},
                "customer_phone": {"type": "string"},
                "customer_address": {"type": "string"},
                "items": {"type": "string"},
                "total": {"type": "number"},
                "payment_method": {"type": "string"},
            },
            "required": ["customer_name", "customer_phone", "customer_address", "items", "total"],
        },
        handler=create_order,
    ),
    Tool(
        name="notify_owner",
        description="მფლობელის გაფრთხილება WhatsApp-ზე.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "customer_name": {"type": "string"},
                "customer_phone": {"type": "string"},
                "details": {"type": "string"},
            },
            "required": ["reason"],
        },
        handler=notify_owner,
    ),
    Tool(
        name="search_knowledge",
        description="ცოდნის ბაზაში ძიება.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
        handler=search_knowledge,
    ),
    # DISABLED — AI photo matching handles this automatically now.
    # Payment screenshot confirmation still works via send_whatsapp_image in facebook.py.
    # Tool(
    #     name="forward_photo_to_owner",
    #     description="კლიენტის ფოტო მფლობელს გადაუგზავნე WhatsApp-ზე. მფლობელი გადაწყვეტს მარაგშია თუ არა. გამოიძახე მხოლოდ მას შემდეგ რაც კლიენტმა ზომა აირჩია.",
    #     parameters={
    #         "type": "object",
    #         "properties": {
    #             "size": {"type": "string", "description": "'პატარა' ან 'დიდი'"},
    #         },
    #         "required": ["size"],
    #     },
    #     handler=forward_photo_to_owner,
    # ),
]
