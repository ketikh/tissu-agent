"""Tools for Tissu Shop Support + Sales agent."""
from __future__ import annotations

import os
import json
import httpx
from datetime import datetime, timezone
from src.db import get_db
from src.engine import Tool


async def check_inventory(model: str = "", size: str = "", search: str = "") -> dict:
    """Check what's in stock. Can filter by model, size, or search by tags/description."""
    pool = await get_db()
    query = "SELECT * FROM inventory WHERE stock > 0"
    params = []
    idx = 1
    if model:
        query += f" AND model ILIKE ${idx}"
        params.append(f"%{model}%")
        idx += 1
    if size:
        query += f" AND size ILIKE ${idx}"
        params.append(f"%{size}%")
        idx += 1
    if search:
        # Check if search looks like a product code (FP1, TP15, etc.)
        search_upper = search.strip().upper()
        if any(search_upper.startswith(p) for p in ("FP", "TP", "FD", "TD")) and any(c.isdigit() for c in search_upper):
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
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO orders (customer_name, customer_phone, customer_address, items, total, payment_method, notes, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
        customer_name, customer_phone, customer_address, items, total, payment_method, notes, now, now,
    )
    order_id = row["id"]

    # Notify owner via WhatsApp about new order
    from src.notifications import send_whatsapp_text
    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    wa_msg = (
        f"🛒 ახალი შეკვეთა #{order_id}!\n"
        f"👤 {customer_name}\n"
        f"📱 {customer_phone}\n"
        f"📍 {customer_address}\n"
        f"📦 {items}\n"
        f"💰 {total}₾\n\n"
        f"📋 ადმინ პანელი:\n{public_url}/admin"
    )
    await send_whatsapp_text(wa_msg)

    return {"success": True, "order_id": order_id}


async def notify_owner(reason: str, customer_name: str = "", customer_phone: str = "", details: str = "", conversation_id: str = "") -> dict:
    """Notify shop owner via WhatsApp with product photo and confirm/deny."""
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    notification = {"reason": reason, "customer_name": customer_name, "customer_phone": customer_phone, "details": details}
    await pool.execute(
        "INSERT INTO tickets (subject, description, status, priority, customer_email, conversation_id, created_at, updated_at) VALUES ($1, $2, 'open', 'urgent', $3, $4, $5, $6)",
        f"[NOTIFICATION] {reason}", json.dumps(notification, ensure_ascii=False), customer_phone, conversation_id, now, now,
    )

    wa_phone_id = os.getenv("WA_PHONE_ID", "")
    wa_token = os.getenv("WA_TOKEN", "")
    owner_number = os.getenv("OWNER_WHATSAPP", "")
    public_url = os.getenv("PUBLIC_URL", "")

    if wa_phone_id and wa_token and owner_number:
        headers = {"Authorization": f"Bearer {wa_token}", "Content-Type": "application/json"}
        wa_url = f"https://graph.facebook.com/v21.0/{wa_phone_id}/messages"

        # Find product code in details/reason — support Georgian letters
        _ka_to_lat = {"ტ": "T", "ფ": "F", "პ": "P", "დ": "D"}
        product_code = ""
        product_image = ""
        for word in (details + " " + reason).split():
            word_clean = word.strip(".,!?\"'()[]")
            if not word_clean:
                continue
            converted = ""
            for ch in word_clean:
                converted += _ka_to_lat.get(ch, ch)
            converted = converted.upper()
            if len(converted) <= 5 and any(c.isalpha() for c in converted) and any(c.isdigit() for c in converted):
                product_code = converted
                break

        if product_code:
            row = await pool.fetchrow("SELECT image_url FROM inventory WHERE UPPER(code) = $1", product_code)
            if row and row["image_url"]:
                product_image = row["image_url"]

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                if product_image:
                    _static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
                    _img_path = product_image
                    if _img_path.startswith("/static/"):
                        _img_path = os.path.join(_static_dir, _img_path.replace("/static/", ""))
                    if os.path.exists(_img_path):
                        with open(_img_path, "rb") as _f:
                            _img_bytes = _f.read()
                        _upload = await client.post(
                            f"https://graph.facebook.com/v21.0/{wa_phone_id}/media",
                            headers={"Authorization": f"Bearer {wa_token}"},
                            data={"messaging_product": "whatsapp", "type": "image/jpeg"},
                            files={"file": ("photo.jpg", _img_bytes, "image/jpeg")},
                        )
                        _media_id = _upload.json().get("id", "")
                        if _media_id:
                            await client.post(wa_url, headers=headers, json={
                                "messaging_product": "whatsapp", "to": owner_number,
                                "type": "image",
                                "image": {"id": _media_id, "caption": f"📌 {product_code}"},
                            })

                msg_parts = [f"🔔 {reason}"]
                if customer_name:
                    msg_parts.append(f"👤 {customer_name}")
                if customer_phone:
                    msg_parts.append(f"📱 {customer_phone}")
                if details:
                    msg_parts.append(f"📝 {details}")
                # Only add confirm/deny links for order-related notifications
                _needs_confirm = any(kw in reason.lower() for kw in ("შეკვეთა", "გადახდა", "ჩარიცხ", "order", "payment"))
                if public_url and conversation_id and _needs_confirm:
                    msg_parts.append("")
                    msg_parts.append(f"✅ ვადასტურებ:\n{public_url}/api/owner-confirm/{conversation_id}")
                    msg_parts.append(f"❌ არ ვადასტურებ:\n{public_url}/api/owner-deny/{conversation_id}")

                await client.post(wa_url, headers=headers, json={
                    "messaging_product": "whatsapp", "to": owner_number,
                    "type": "text", "text": {"body": "\n".join(msg_parts)},
                })
        except Exception:
            pass

    return {"notified": True, "message": "მფლობელს ეცნობა"}


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
    from src.notifications import send_whatsapp_image, send_whatsapp_text

    print(f"[PHOTO] forward_photo_to_owner called: size={size}, conv_id={conversation_id}")
    print(f"[PHOTO] Pending photos keys: {list(_pending_photos.keys())}")

    photo_bytes = _pending_photos.get(conversation_id)
    if not photo_bytes:
        print(f"[PHOTO] ERROR: Photo NOT FOUND for conversation_id={conversation_id}")
        return {"forwarded": False, "message": "ფოტო ვერ მოიძებნა"}

    print(f"[PHOTO] Photo found: {len(photo_bytes)} bytes")

    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    confirm_url = f"{public_url}/api/photo-confirm/{conversation_id}"
    deny_url = f"{public_url}/api/photo-deny/{conversation_id}"
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
    sent = await send_whatsapp_image(photo_bytes, caption=caption)
    print(f"[PHOTO] WhatsApp customer photo sent: {sent}")

    # 2. If AI matched, also send the AI-recommended product photo so owner can visually compare
    if ai_product_url:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(ai_product_url)
                if resp.status_code == 200:
                    await send_whatsapp_image(
                        resp.content,
                        caption=f"🤖 AI-ის რეკომენდაცია ↑ შეადარე კლიენტის ფოტოს",
                    )
                    print(f"[PHOTO] AI recommendation photo also sent")
        except Exception as e:
            print(f"[PHOTO] Failed to send AI recommendation photo: {e}")

    # Clean up
    _pending_photos.pop(conversation_id, None)

    return {"forwarded": sent, "message": "მფლობელს გადაეგზავნა" if sent else "WhatsApp გაგზავნა ვერ მოხერხდა"}


SUPPORT_TOOLS = [
    Tool(
        name="check_inventory",
        description="მარაგის შემოწმება. ფილტრავს მოდელით, ზომით, ან ტეგებით/ფერით.",
        parameters={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "'ფხრიწიანი' ან 'თასმიანი'"},
                "size": {"type": "string", "description": "'პატარა' ან 'დიდი'"},
                "search": {"type": "string", "description": "ძიება ფერით ან აღწერით"},
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
