"""Tissu Agent Server.

Local-first AI agent API that serves two business agents:
- /api/support — Customer Support + Sales Hybrid
- /api/marketing — Marketing + Content + Ads Intelligence

N8N connects to these endpoints to orchestrate workflows.
"""

import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import shutil
import io
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.config import API_HOST, API_PORT
from src.db import init_db, get_db
from src.models import ChatRequest, ChatResponse, LeadCreate, TicketCreate, ContentCreate
from src.engine import run_agent
from src.agents.support_sales import get_support_sales_agent
from src.agents.marketing import get_marketing_agent
from src.channels import get_adapter, ADAPTERS
import httpx


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_knowledge_base()
    yield


app = FastAPI(
    title="Tissu Agents",
    description="Local-first AI agent system for business",
    version="0.1.0",
    lifespan=lifespan,
)

def save_uploaded_image(upload: UploadFile, prefix: str) -> str:
    """Save uploaded image, converting HEIC/HEIF to JPEG automatically."""
    filename_base = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    save_dir = Path(__file__).parent / "static" / "products"

    ext = Path(upload.filename).suffix.lower()
    if ext in ('.heic', '.heif'):
        img = Image.open(upload.file)
        filename = f"{filename_base}.jpg"
        img.save(save_dir / filename, "JPEG", quality=85)
    else:
        filename = f"{filename_base}{ext}"
        with open(save_dir / filename, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    return f"/static/products/{filename}"


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Chat UI ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    html_path = Path(__file__).parent / "chat.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    html_path = Path(__file__).parent / "admin.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Agent Chat Endpoints ──────────────────────────────────────

# ── Facebook / Instagram Webhook ───────────────────────────────

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://endurant-hyped-johnna.ngrok-free.dev")
PAGE_ID = "447377388462459"  # Tissu Shop page ID

# Anti-loop: track recently processed messages
import time as _time
_processed_mids = {}  # mid -> timestamp

@app.get("/webhook")
async def webhook_verify(request: Request):
    """Facebook webhook verification (GET request)."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


async def _notify_owner_whatsapp(message: str):
    """Send WhatsApp notification to owner."""
    wa_phone_id = os.getenv("WA_PHONE_ID", "")
    wa_token = os.getenv("WA_TOKEN", "")
    owner = os.getenv("OWNER_WHATSAPP", "")
    if wa_phone_id and wa_token and owner:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(
                    f"https://graph.facebook.com/v21.0/{wa_phone_id}/messages",
                    headers={"Authorization": f"Bearer {wa_token}", "Content-Type": "application/json"},
                    json={"messaging_product": "whatsapp", "to": owner, "type": "text", "text": {"body": message}},
                )
        except Exception:
            logging.getLogger(__name__).error(f"Failed to notify owner: {message}")


async def _process_message(sender_id: str, text: str, conversation_id: str, channel: str, customer_name: str = "", image_url: str = ""):
    """Process a message in the background — agent + reply + images."""
    import logging
    logger = logging.getLogger(__name__)

    # Forward customer image to owner via WhatsApp
    if image_url:
        try:
            _wa_phone_id = os.getenv("WA_PHONE_ID", "")
            _wa_token = os.getenv("WA_TOKEN", "")
            _owner = os.getenv("OWNER_WHATSAPP", "")
            if _wa_phone_id and _wa_token and _owner:
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as _c:
                    _img_resp = await _c.get(image_url)
                    if _img_resp.status_code == 200:
                        _cname = customer_name or sender_id
                        _upload = await _c.post(
                            f"https://graph.facebook.com/v21.0/{_wa_phone_id}/media",
                            headers={"Authorization": f"Bearer {_wa_token}"},
                            data={"messaging_product": "whatsapp", "type": "image/jpeg"},
                            files={"file": ("photo.jpg", _img_resp.content, "image/jpeg")},
                        )
                        _media_id = _upload.json().get("id", "")
                        if _media_id:
                            await _c.post(
                                f"https://graph.facebook.com/v21.0/{_wa_phone_id}/messages",
                                headers={"Authorization": f"Bearer {_wa_token}", "Content-Type": "application/json"},
                                json={"messaging_product": "whatsapp", "to": _owner,
                                      "type": "image", "image": {"id": _media_id, "caption": f"📷 {_cname}\n\nვადასტურებ / არ ვადასტურებ"}},
                            )
        except Exception as e:
            logger.error(f"WA image forward failed: {e}", exc_info=True)

    # Add customer name as hidden context (agent uses it internally, never shows to customer)
    if customer_name:
        text = f"[SYSTEM: customer_name={customer_name}]\n{text}"

    agent = get_support_sales_agent()
    try:
        result = await run_agent(agent, text, conversation_id)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        await _notify_owner_whatsapp(f"🚨 აგენტის შეცდომა!\nკლიენტის მესიჯი: {text[:200]}\nშეცდომა: {str(e)[:300]}")
        result = {"reply": "გადავამოწმებ და მოგწერთ ✨", "tool_calls_made": [], "tool_results_data": {}}

    # Send reply back via Facebook/Instagram API
    if not FB_PAGE_TOKEN:
        return

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            fb_api = "https://graph.facebook.com/v21.0/me/messages"
            fb_params = {"access_token": FB_PAGE_TOKEN}

            reply_text = result["reply"].strip()
            if not reply_text:
                if image_url:
                    reply_text = "მადლობა, გადავამოწმებ ✨"
                else:
                    reply_text = "გადავამოწმებ და მოგწერთ ✨"
                    await _notify_owner_whatsapp(f"⚠️ ბოტი ვერ უპასუხა!\nკლიენტის მესიჯი: {text[:200]}")

            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": reply_text[:2000]},
            })

            # Send product images: code text → front photo → back photo
            tool_data = result.get("tool_results_data", {})
            logger.info(f"tool_results_data keys: {list(tool_data.keys())}")
            inventory_data = tool_data.get("check_inventory")
            if inventory_data:
                logger.info(f"check_inventory found={inventory_data.get('found')}, items={len(inventory_data.get('items',[]))}")
            if inventory_data and inventory_data.get("found"):
                items = inventory_data.get("items", [])
                for item in items:
                    code = item.get("code", "")
                    front = item.get("image_url", "")
                    back = item.get("image_url_back", "")
                    if not front:
                        continue
                    # 1. Send code as text
                    if code:
                        try:
                            await client.post(fb_api, params=fb_params, json={
                                "recipient": {"id": sender_id},
                                "message": {"text": f"📌 {code}"},
                            })
                        except Exception:
                            pass
                    # 2. Send front photo
                    try:
                        img_url = PUBLIC_URL + front if front.startswith("/") else front
                        await client.post(fb_api, params=fb_params, json={
                            "recipient": {"id": sender_id},
                            "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                        })
                    except Exception:
                        pass
                    # 3. Send back photo
                    if back:
                        try:
                            img_url = PUBLIC_URL + back if back.startswith("/") else back
                            await client.post(fb_api, params=fb_params, json={
                                "recipient": {"id": sender_id},
                                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                            })
                        except Exception:
                            pass
    except Exception as e:
        logger.error(f"Failed to send FB reply: {e}", exc_info=True)
        await _notify_owner_whatsapp(f"⚠️ პასუხის გაგზავნა ვერ მოხერხდა!\nკლიენტი: {sender_id}\nშეცდომა: {str(e)[:300]}")


@app.post("/webhook")
async def webhook_receive(request: Request):
    """Receive messages from Facebook Messenger / Instagram DM."""
    import asyncio
    body = await request.json()

    if body.get("object") not in ("page", "instagram"):
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            if event.get("delivery") or event.get("read"):
                continue

            sender_id = event.get("sender", {}).get("id", "")
            message = event.get("message", {})
            text = message.get("text", "")
            mid = message.get("mid", "")

            if message.get("is_echo") or sender_id == PAGE_ID or not sender_id:
                continue

            attachments = message.get("attachments", [])
            image_url = ""
            for att in attachments:
                if att.get("type") == "image":
                    image_url = att.get("payload", {}).get("url", "")
                    break

            if not text and not image_url:
                continue

            # If customer sent image — tell bot + forward will happen in _process_message
            if image_url:
                text = (text or "") + "\n[კლიენტმა გამოგზავნა ფოტო (შესაძლოა გადახდის სქრინი). უთხარი 'მადლობა, გადავამოწმებ ✨' და გამოიძახე notify_owner 'კლიენტმა ფოტო გამოგზავნა']"

            if mid and mid in _processed_mids:
                continue
            if mid:
                _processed_mids[mid] = _time.time()
                now = _time.time()
                for k in list(_processed_mids):
                    if now - _processed_mids[k] > 300:
                        del _processed_mids[k]

            channel = "instagram_dm" if body["object"] == "instagram" else "facebook_messenger"
            conversation_id = f"{channel}_{sender_id}"

            # Get customer name from Facebook profile
            customer_name = ""
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    resp = await c.get(
                        f"https://graph.facebook.com/v21.0/{sender_id}",
                        params={"fields": "first_name,last_name,name", "access_token": FB_PAGE_TOKEN},
                    )
                    profile = resp.json()
                    customer_name = profile.get("name", "") or f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
            except Exception:
                pass

            # Process in background — return 200 to Facebook immediately
            asyncio.create_task(_process_message(sender_id, text, conversation_id, channel, customer_name, image_url))

    return {"status": "ok"}


# ── WhatsApp Owner Response Webhook ──────────────────────────

@app.get("/wa-webhook")
async def wa_webhook_verify(request: Request):
    """WhatsApp webhook verification."""
    params = request.query_params
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == "tissu_wa_verify":
        return int(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


_wa_processed_mids = {}

@app.post("/wa-webhook")
async def wa_webhook_receive(request: Request):
    """Receive WhatsApp messages from owner — forward to customer or confirm/deny."""
    body = await request.json()

    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                sender = msg.get("from", "")
                owner_number = os.getenv("OWNER_WHATSAPP", "")
                if sender != owner_number:
                    continue

                # Anti-duplicate
                wa_mid = msg.get("id", "")
                if wa_mid and wa_mid in _wa_processed_mids:
                    continue
                if wa_mid:
                    _wa_processed_mids[wa_mid] = _time.time()
                    for k in list(_wa_processed_mids):
                        if _time.time() - _wa_processed_mids[k] > 300:
                            del _wa_processed_mids[k]

                text = msg.get("text", {}).get("body", "").strip()
                if not text:
                    continue

                # Find the latest conversation
                db = await get_db()
                try:
                    cursor = await db.execute(
                        "SELECT conversation_id FROM tickets ORDER BY created_at DESC LIMIT 1"
                    )
                    row = await cursor.fetchone()
                    conv_id = row["conversation_id"] if row else ""
                finally:
                    await db.close()

                if not conv_id:
                    continue

                sender_id = conv_id.replace("facebook_messenger_", "").replace("instagram_dm_", "")
                if not FB_PAGE_TOKEN or not sender_id:
                    continue

                # Check latest ticket status to determine which stage we're at
                db2 = await get_db()
                try:
                    _tcursor = await db2.execute(
                        "SELECT subject, status FROM tickets WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1", (conv_id,)
                    )
                    _ticket = await _tcursor.fetchone()
                    _ticket_status = _ticket["status"] if _ticket else "open"
                    _ticket_subject = _ticket["subject"] if _ticket else ""
                finally:
                    await db2.close()

                # Owner's message → forward as agent instruction to customer
                text_lower = text.lower()
                if "ვადასტურებ" in text_lower and "არ" not in text_lower:
                    # Payment confirmation — clean up uploaded photos
                    _uploads_dir = Path(__file__).parent / "static" / "uploads"
                    if _uploads_dir.exists():
                        for _f in _uploads_dir.iterdir():
                            if _f.suffix in ('.jpg', '.jpeg', '.png'):
                                _f.unlink(missing_ok=True)
                    agent = get_support_sales_agent()
                    result = await run_agent(agent, "[მფლობელმა დაადასტურა გადახდა. მოითხოვე მისამართი და ტელეფონი.]", conv_id)
                    reply = result["reply"].strip() or "გადახდა დადასტურებულია! ✨ მისამართი და ტელეფონის ნომერი მოგვწერეთ."
                elif "არ ვადასტურებ" in text_lower or ("არ" in text_lower and "ვადასტურებ" in text_lower):
                    # Deny
                    agent = get_support_sales_agent()
                    result = await run_agent(agent, "[მფლობელმა უარყო. თავაზიანად უთხარი რომ ეს პროდუქტი ამჟამად არ არის ხელმისაწვდომი და შესთავაზე სხვა ვარიანტი.]", conv_id)
                    reply = result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად არ არის ხელმისაწვდომი. გსურთ სხვა ვარიანტი ნახოთ?"
                elif text.startswith("უპასუხე:") or text.startswith("უპასუხე "):
                    # Owner dictates reply — send directly
                    reply = text.replace("უპასუხე:", "").replace("უპასუხე ", "").strip()
                else:
                    # Other text from owner — ignore, don't forward to customer
                    continue

                async with httpx.AsyncClient(timeout=30) as client:
                    await client.post(
                        "https://graph.facebook.com/v21.0/me/messages",
                        params={"access_token": FB_PAGE_TOKEN},
                        json={"recipient": {"id": sender_id}, "message": {"text": reply}},
                    )

    return {"status": "ok"}


@app.get("/api/owner-confirm/{conversation_id}")
async def owner_confirm(conversation_id: str):
    """Owner confirms order — agent continues with bank question."""
    sender_id = conversation_id.replace("facebook_messenger_", "").replace("instagram_dm_", "")

    if FB_PAGE_TOKEN and sender_id:
        async with httpx.AsyncClient(timeout=30) as client:
            # Owner confirmed payment — ask for address
            agent = get_support_sales_agent()
            result = await run_agent(agent, "[მფლობელმა დაადასტურა გადახდა. მოითხოვე მისამართი და ტელეფონი.]", conversation_id)
            reply = result["reply"].strip() or "გადახდა დადასტურებულია! ✨ მისამართი და ტელეფონის ნომერი მომწერეთ."

            await client.post(
                "https://graph.facebook.com/v21.0/me/messages",
                params={"access_token": FB_PAGE_TOKEN},
                json={"recipient": {"id": sender_id}, "message": {"text": reply}},
            )
    return HTMLResponse("<h1>✅ დადასტურებულია!</h1><p>კლიენტს ეცნობა.</p>")


@app.get("/api/owner-deny/{conversation_id}")
async def owner_deny(conversation_id: str):
    """Owner denies — agent tells customer."""
    sender_id = conversation_id.replace("facebook_messenger_", "").replace("instagram_dm_", "")

    if FB_PAGE_TOKEN and sender_id:
        async with httpx.AsyncClient(timeout=30) as client:
            agent = get_support_sales_agent()
            result = await run_agent(agent, "[მფლობელმა უარყო. თავაზიანად უთხარი რომ ეს პროდუქტი ამჟამად არ არის ხელმისაწვდომი და შესთავაზე სხვა ვარიანტი.]", conversation_id)
            reply = result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად არ არის ხელმისაწვდომი. გსურთ სხვა ვარიანტი ნახოთ?"

            await client.post(
                "https://graph.facebook.com/v21.0/me/messages",
                params={"access_token": FB_PAGE_TOKEN},
                json={"recipient": {"id": sender_id}, "message": {"text": reply}},
            )
    return HTMLResponse("<h1>❌ უარყოფილია</h1><p>კლიენტს ეცნობა.</p>")


# ── Agent Chat Endpoints ──────────────────────────────────────

@app.post("/api/support", response_model=ChatResponse)
async def chat_support(req: ChatRequest):
    agent = get_support_sales_agent()

    # Enrich the message with customer context if provided
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
    """Universal webhook endpoint for any channel.

    N8N routes platform webhooks here:
    - POST /api/webhook/facebook_messenger
    - POST /api/webhook/instagram_dm
    - POST /api/webhook/whatsapp

    The adapter parses the platform payload, runs the agent,
    and returns a platform-formatted response.
    """
    if channel not in ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown channel: {channel}. Available: {list(ADAPTERS.keys())}")

    adapter = get_adapter(channel)
    payload = await request.json()
    chat_request = adapter.parse_incoming(payload)
    agent = get_support_sales_agent()

    enriched_message = chat_request.message
    if chat_request.customer_context:
        ctx = chat_request.customer_context
        context_parts = [f"Channel: {channel}"]
        if ctx.name:
            context_parts.append(f"Customer name: {ctx.name}")
        if ctx.product_interest:
            context_parts.append(f"Product interest: {ctx.product_interest}")
        enriched_message = f"[Context: {'; '.join(context_parts)}]\n\n{chat_request.message}"

    result = await run_agent(agent, enriched_message, chat_request.conversation_id)
    return {"agent_response": result, "channel": channel}


@app.post("/api/marketing", response_model=ChatResponse)
async def chat_marketing(req: ChatRequest):
    agent = get_marketing_agent()
    result = await run_agent(agent, req.message, req.conversation_id)
    return ChatResponse(**result)


# ── Data Endpoints (for N8N and direct access) ────────────────

@app.get("/api/leads")
async def list_leads(status: str = "", limit: int = 50):
    db = await get_db()
    try:
        query = "SELECT * FROM leads WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"leads": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.post("/api/leads")
async def create_lead(lead: LeadCreate):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            "INSERT INTO leads (name, email, company, phone, source, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (lead.name, lead.email, lead.company, lead.phone, lead.source, lead.notes, now, now),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "message": "Lead created"}
    finally:
        await db.close()


@app.get("/api/tickets")
async def list_tickets(status: str = "", limit: int = 50):
    db = await get_db()
    try:
        query = "SELECT * FROM tickets WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"tickets": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.get("/api/content")
async def list_content(content_type: str = "", status: str = "", limit: int = 50):
    db = await get_db()
    try:
        query = "SELECT * FROM content WHERE 1=1"
        params = []
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"content": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.post("/api/content")
async def create_content(item: ContentCreate):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            "INSERT INTO content (title, body, content_type, tags, scheduled_at, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'draft', ?, ?)",
            (item.title, item.body, item.content_type, json.dumps(item.tags), item.scheduled_at, now, now),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "message": "Content created"}
    finally:
        await db.close()


@app.get("/api/conversations")
async def list_conversations(agent_type: str = "", limit: int = 20):
    db = await get_db()
    try:
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"conversations": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"conversation_id": conversation_id, "messages": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.get("/api/knowledge")
async def list_knowledge(category: str = ""):
    db = await get_db()
    try:
        query = "SELECT * FROM knowledge_base WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"articles": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.post("/api/knowledge")
async def add_knowledge(question: str, answer: str, category: str = "general"):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            "INSERT INTO knowledge_base (question, answer, category, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, category, now),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "message": "Knowledge article added"}
    finally:
        await db.close()


@app.get("/api/inventory")
async def list_inventory():
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM inventory ORDER BY model, size")
        rows = await cursor.fetchall()
        return {"inventory": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.post("/api/inventory")
async def add_inventory(
    product_name: str = Form(...),
    model: str = Form(...),
    size: str = Form(...),
    price: float = Form(...),
    stock: int = Form(...),
    color: str = Form(""),
    style: str = Form(""),
    image: UploadFile = File(None),
):
    image_url = ""
    if image and image.filename:
        prefix = f"{model}_{size}_{color}".replace(" ", "_")
        image_url = save_uploaded_image(image, prefix)

    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = await db.execute(
            "INSERT INTO inventory (product_name, model, size, color, style, price, stock, image_url, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product_name, model, size, color, style, price, stock, image_url, now, now),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "image_url": image_url, "message": "Product added"}
    finally:
        await db.close()


@app.put("/api/inventory/{item_id}")
async def update_inventory(item_id: int, stock: int = None, price: float = None, model: str = None, size: str = None, color: str = None, tags: str = None):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        if stock is not None:
            await db.execute("UPDATE inventory SET stock = ?, updated_at = ? WHERE id = ?", (stock, now, item_id))
        if price is not None:
            await db.execute("UPDATE inventory SET price = ?, updated_at = ? WHERE id = ?", (price, now, item_id))
        if model is not None:
            await db.execute("UPDATE inventory SET model = ?, updated_at = ? WHERE id = ?", (model, now, item_id))
        if size is not None:
            new_price = 74 if "დიდი" in size else 69
            await db.execute("UPDATE inventory SET size = ?, price = ?, updated_at = ? WHERE id = ?", (size, new_price, now, item_id))
        if color is not None:
            await db.execute("UPDATE inventory SET color = ?, updated_at = ? WHERE id = ?", (color, now, item_id))
        if tags is not None:
            await db.execute("UPDATE inventory SET tags = ?, updated_at = ? WHERE id = ?", (tags, now, item_id))
        await db.commit()
        return {"message": f"Item #{item_id} updated"}
    finally:
        await db.close()


@app.post("/api/inventory/{item_id}/image")
async def upload_product_image(item_id: int, image: UploadFile = File(...)):
    image_url = save_uploaded_image(image, f"product_{item_id}_front")
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute("UPDATE inventory SET image_url = ?, updated_at = ? WHERE id = ?", (image_url, now, item_id))
        await db.commit()
        return {"image_url": image_url}
    finally:
        await db.close()


@app.post("/api/inventory/swap-images")
async def swap_images(request: Request):
    """Swap images between two inventory slots (drag & drop support)."""
    data = await request.json()
    from_id = data["from_id"]
    from_side = data["from_side"]  # "front" or "back"
    to_id = data["to_id"]
    to_side = data["to_side"]

    db = await get_db()
    try:
        # Get current URLs
        c1 = await db.execute("SELECT image_url, image_url_back FROM inventory WHERE id = ?", (from_id,))
        r1 = await c1.fetchone()
        c2 = await db.execute("SELECT image_url, image_url_back FROM inventory WHERE id = ?", (to_id,))
        r2 = await c2.fetchone()
        if not r1 or not r2:
            raise HTTPException(status_code=404)

        from_url = r1["image_url"] if from_side == "front" else r1["image_url_back"]
        to_url = r2["image_url"] if to_side == "front" else r2["image_url_back"]

        now = datetime.now(timezone.utc).isoformat()
        from_col = "image_url" if from_side == "front" else "image_url_back"
        to_col = "image_url" if to_side == "front" else "image_url_back"

        await db.execute(f"UPDATE inventory SET {from_col} = ?, updated_at = ? WHERE id = ?", (to_url, now, from_id))
        await db.execute(f"UPDATE inventory SET {to_col} = ?, updated_at = ? WHERE id = ?", (from_url, now, to_id))
        await db.commit()
        return {"message": "Images swapped"}
    finally:
        await db.close()


@app.post("/api/inventory/{item_id}/image_back")
async def upload_back_image(item_id: int, image: UploadFile = File(...)):
    image_url = save_uploaded_image(image, f"product_{item_id}_back")
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute("UPDATE inventory SET image_url_back = ?, updated_at = ? WHERE id = ?", (image_url, now, item_id))
        await db.commit()
        return {"image_url": image_url}
    finally:
        await db.close()


@app.post("/api/inventory/{item_id}/clear-image")
async def clear_image(item_id: int, side: str = "back"):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        col = "image_url" if side == "front" else "image_url_back"
        await db.execute(f"UPDATE inventory SET {col} = '', updated_at = ? WHERE id = ?", (now, item_id))
        await db.commit()
        return {"message": f"{side} image cleared"}
    finally:
        await db.close()


@app.post("/api/inventory/{item_id}/remove-back")
async def remove_back_image(item_id: int):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute("UPDATE inventory SET image_url_back = '', updated_at = ? WHERE id = ?", (now, item_id))
        await db.commit()
        return {"message": "Back image removed"}
    finally:
        await db.close()


@app.delete("/api/inventory/{item_id}")
async def delete_inventory(item_id: int):
    db = await get_db()
    try:
        await db.execute("DELETE FROM inventory WHERE id = ?", (item_id,))
        await db.commit()
        return {"message": f"Item #{item_id} deleted"}
    finally:
        await db.close()


@app.get("/api/orders")
async def list_orders(status: str = "", limit: int = 50):
    db = await get_db()
    try:
        query = "SELECT * FROM orders WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {"orders": [dict(r) for r in rows]}
    finally:
        await db.close()


@app.put("/api/orders/{order_id}")
async def update_order(order_id: int, request: Request):
    data = await request.json()
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        for field in ("customer_phone", "customer_address", "status", "notes"):
            if field in data:
                await db.execute(f"UPDATE orders SET {field} = ?, updated_at = ? WHERE id = ?", (data[field], now, order_id))
        await db.commit()
        return {"success": True}
    finally:
        await db.close()


@app.delete("/api/orders/{order_id}")
async def delete_order(order_id: int):
    db = await get_db()
    try:
        await db.execute("DELETE FROM orders WHERE id = ?", (order_id,))
        await db.commit()
        return {"success": True}
    finally:
        await db.close()


@app.post("/api/orders/{order_id}/decrease-stock")
async def decrease_stock_for_order(order_id: int):
    """When order moves to ready_for_send, decrease stock for the product code."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT items FROM orders WHERE id = ?", (order_id,))
        order = await cursor.fetchone()
        if not order:
            raise HTTPException(status_code=404)
        item_code = order["items"].strip().upper()
        # Find product by code and decrease stock
        await db.execute(
            "UPDATE inventory SET stock = MAX(0, stock - 1), updated_at = ? WHERE UPPER(code) = ? AND stock > 0",
            (datetime.now(timezone.utc).isoformat(), item_code),
        )
        await db.commit()
        return {"success": True, "code": item_code}
    finally:
        await db.close()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agents": ["support_sales", "marketing"], "version": "0.1.0"}


# ── Seed Data ─────────────────────────────────────────────────

async def seed_knowledge_base():
    """Seed Tissu Shop knowledge base and starter inventory."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as c FROM knowledge_base")
        count = (await cursor.fetchone())["c"]
        if count > 0:
            return

        now = datetime.now(timezone.utc).isoformat()

        # Knowledge base — real Tissu Shop info
        articles = [
            ("რა ფასია?", "პატარა ზომა (33x25 სმ) — 69 ლარი. დიდი ზომა (37x27 სმ) — 74 ლარი.", "pricing"),
            ("რა ზომები გაქვთ?", "გვაქვს 2 ზომა: პატარა (33x25 სმ, 13-14 ინჩი ლეპტოპისთვის) და დიდი (37x27 სმ, 15-16 ინჩი ლეპტოპისთვის).", "products"),
            ("რა მოდელები გაქვთ?", "გვაქვს 2 მოდელი: თასმიანი (სახელურით) და ფხრიწიანი (zipper-ით).", "products"),
            ("როგორ ხდება მიწოდება?", "მიწოდება თბილისის მასშტაბით 6 ლარი. ღამის 12 საათამდე შეკვეთაზე მიწოდება მეორე დღეს. შაბათის შეკვეთა ორშაბათს. კვირას მიწოდება არ ხდება.", "delivery"),
            ("თუ არ ვიქნები მისამართზე?", "შეგიძლიათ მიუთითოთ მიმდებარე ადგილი სადაც დატოვებს კურიერი. თუ ვერ ჩაიბარებთ, კურიერი წაიღებს უკან და მეორე დღეს მოგაწვდით, რისთვისაც დამატებითი საკურიეროს გადახდა მოგიწევთ.", "delivery"),
            ("როგორ გადავიხადო?", "გადახდა ბანკის ანგარიშზე. თიბისი: GE58TB7085345064300066. საქართველოს ბანკი: GE65BG0000000358364200. გადახდის შემდეგ გვჭირდება: სქრინი/ქვითარი + მისამართი + ტელეფონის ნომერი.", "payment"),
            ("რისგან არის გაკეთებული?", "ლეპტოპის ქეისები ხელნაკეთია, ნაჭრისგან. ყოველი ცალი უნიკალური დიზაინისაა.", "products"),
        ]
        for q, a, cat in articles:
            await db.execute(
                "INSERT INTO knowledge_base (question, answer, category, created_at) VALUES (?, ?, ?, ?)",
                (q, a, cat, now),
            )

        # Load real inventory from seed_inventory.json
        seed_file = Path(__file__).parent / "seed_inventory.json"
        if seed_file.exists():
            import json as _json
            items = _json.loads(seed_file.read_text())
            for item in items:
                await db.execute(
                    "INSERT INTO inventory (product_name, model, size, color, style, code, tags, price, stock, image_url, image_url_back, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (item["product_name"], item["model"], item["size"], item.get("color", ""), item.get("style", ""),
                     item.get("code", ""), item.get("tags", ""), item["price"], item["stock"],
                     item.get("image_url", ""), item.get("image_url_back", ""), now, now),
                )

        await db.commit()
    finally:
        await db.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", API_PORT))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("RAILWAY_ENVIRONMENT") is None)
