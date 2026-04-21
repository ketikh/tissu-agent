"""Tissu Agent Server.

AI-powered sales agent for Tissu Shop. Facebook Messenger bot with
Gemini Vision, WhatsApp owner notifications, and admin panel.
"""
import os
import json
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import API_HOST, API_PORT
from src.db import init_db, get_db, close_pool
from src.models import ChatRequest, ChatResponse, LeadCreate, ContentCreate
from src.engine import run_agent
from src.agents.support_sales import get_support_sales_agent
from src.agents.marketing import get_marketing_agent
from src.channels import get_adapter, ADAPTERS
from src.webhooks.facebook import router as fb_router
from src.webhooks.whatsapp import router as wa_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_knowledge_base()
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

# Include webhook routers
app.include_router(fb_router)
app.include_router(wa_router)


# ── Pages ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    html_path = Path(__file__).parent / "chat.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    html_path = Path(__file__).parent / "admin.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


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

    agent = get_support_sales_agent()
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

    agent = get_support_sales_agent()
    result = await run_agent(agent, enriched_message, chat_request.conversation_id)
    return {"agent_response": result, "channel": channel}


@app.post("/api/marketing", response_model=ChatResponse)
async def chat_marketing(req: ChatRequest):
    agent = get_marketing_agent()
    result = await run_agent(agent, req.message, req.conversation_id)
    return ChatResponse(**result)


# ── Data Endpoints ───────────────────────────────────────────

@app.get("/api/leads")
async def list_leads(status: str = "", limit: int = 50):
    pool = await get_db()
    query = "SELECT * FROM leads WHERE 1=1"
    params = []
    idx = 1
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"leads": [dict(r) for r in rows]}


@app.post("/api/leads")
async def create_lead(lead: LeadCreate):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO leads (name, email, company, phone, source, notes, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id",
        lead.name, lead.email, lead.company, lead.phone, lead.source, lead.notes, now, now,
    )
    return {"id": row["id"], "message": "Lead created"}


@app.get("/api/tickets")
async def list_tickets(status: str = "", limit: int = 50):
    pool = await get_db()
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []
    idx = 1
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"tickets": [dict(r) for r in rows]}


@app.get("/api/content")
async def list_content(content_type: str = "", status: str = "", limit: int = 50):
    pool = await get_db()
    query = "SELECT * FROM content WHERE 1=1"
    params = []
    idx = 1
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
async def create_content(item: ContentCreate):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO content (title, body, content_type, tags, scheduled_at, status, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, 'draft', $6, $7) RETURNING id",
        item.title, item.body, item.content_type, json.dumps(item.tags), item.scheduled_at, now, now,
    )
    return {"id": row["id"], "message": "Content created"}


@app.get("/api/conversations")
async def list_conversations(agent_type: str = "", limit: int = 20):
    pool = await get_db()
    query = "SELECT * FROM conversations WHERE 1=1"
    params = []
    idx = 1
    if agent_type:
        query += f" AND agent_type = ${idx}"
        params.append(agent_type)
        idx += 1
    query += f" ORDER BY updated_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"conversations": [dict(r) for r in rows]}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT role, content, created_at FROM messages WHERE conversation_id = $1 ORDER BY created_at",
        conversation_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": [dict(r) for r in rows]}


@app.get("/api/knowledge")
async def list_knowledge(category: str = ""):
    pool = await get_db()
    query = "SELECT * FROM knowledge_base WHERE 1=1"
    params = []
    idx = 1
    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    rows = await pool.fetch(query, *params)
    return {"articles": [dict(r) for r in rows]}


@app.post("/api/knowledge")
async def add_knowledge(question: str, answer: str, category: str = "general"):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO knowledge_base (question, answer, category, created_at) VALUES ($1, $2, $3, $4) RETURNING id",
        question, answer, category, now,
    )
    return {"id": row["id"], "message": "Knowledge article added"}


# ── Inventory Endpoints ──────────────────────────────────────

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


@app.get("/api/inventory")
async def list_inventory():
    pool = await get_db()
    rows = await pool.fetch("SELECT * FROM inventory ORDER BY model, size")
    return {"inventory": [dict(r) for r in rows]}


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

    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO inventory (product_name, model, size, color, style, price, stock, image_url, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id",
        product_name, model, size, color, style, price, stock, image_url, now, now,
    )
    return {"id": row["id"], "image_url": image_url, "message": "Product added"}


@app.put("/api/inventory/{item_id}")
async def update_inventory(item_id: int, stock: int = None, price: float = None, model: str = None, size: str = None, color: str = None, tags: str = None):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    if stock is not None:
        await pool.execute("UPDATE inventory SET stock = $1, updated_at = $2 WHERE id = $3", stock, now, item_id)
    if price is not None:
        await pool.execute("UPDATE inventory SET price = $1, updated_at = $2 WHERE id = $3", price, now, item_id)
    if model is not None:
        await pool.execute("UPDATE inventory SET model = $1, updated_at = $2 WHERE id = $3", model, now, item_id)
    if size is not None:
        new_price = 74 if "დიდი" in size else 69
        await pool.execute("UPDATE inventory SET size = $1, price = $2, updated_at = $3 WHERE id = $4", size, new_price, now, item_id)
    if color is not None:
        await pool.execute("UPDATE inventory SET color = $1, updated_at = $2 WHERE id = $3", color, now, item_id)
    if tags is not None:
        await pool.execute("UPDATE inventory SET tags = $1, updated_at = $2 WHERE id = $3", tags, now, item_id)
    return {"message": f"Item #{item_id} updated"}


@app.post("/api/inventory/{item_id}/image")
async def upload_product_image(item_id: int, image: UploadFile = File(...)):
    image_url = save_uploaded_image(image, f"product_{item_id}_front")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url = $1, updated_at = $2 WHERE id = $3", image_url, now, item_id)
    return {"image_url": image_url}


@app.post("/api/inventory/{item_id}/image_back")
async def upload_back_image(item_id: int, image: UploadFile = File(...)):
    image_url = save_uploaded_image(image, f"product_{item_id}_back")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = $1, updated_at = $2 WHERE id = $3", image_url, now, item_id)
    return {"image_url": image_url}


@app.post("/api/inventory/swap-images")
async def swap_images(request: Request):
    """Swap images between two inventory slots (drag & drop support)."""
    data = await request.json()
    from_id = data["from_id"]
    from_side = data["from_side"]
    to_id = data["to_id"]
    to_side = data["to_side"]

    pool = await get_db()
    r1 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1", from_id)
    r2 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1", to_id)
    if not r1 or not r2:
        raise HTTPException(status_code=404)

    from_url = r1["image_url"] if from_side == "front" else r1["image_url_back"]
    to_url = r2["image_url"] if to_side == "front" else r2["image_url_back"]

    now = datetime.now(timezone.utc).isoformat()
    from_col = "image_url" if from_side == "front" else "image_url_back"
    to_col = "image_url" if to_side == "front" else "image_url_back"

    await pool.execute(f"UPDATE inventory SET {from_col} = $1, updated_at = $2 WHERE id = $3", to_url, now, from_id)
    await pool.execute(f"UPDATE inventory SET {to_col} = $1, updated_at = $2 WHERE id = $3", from_url, now, to_id)
    return {"message": "Images swapped"}


@app.post("/api/inventory/{item_id}/clear-image")
async def clear_image(item_id: int, side: str = "back"):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    col = "image_url" if side == "front" else "image_url_back"
    await pool.execute(f"UPDATE inventory SET {col} = '', updated_at = $1 WHERE id = $2", now, item_id)
    return {"message": f"{side} image cleared"}


@app.post("/api/inventory/{item_id}/remove-back")
async def remove_back_image(item_id: int):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = '', updated_at = $1 WHERE id = $2", now, item_id)
    return {"message": "Back image removed"}


@app.delete("/api/inventory/{item_id}")
async def delete_inventory(item_id: int):
    pool = await get_db()
    await pool.execute("DELETE FROM inventory WHERE id = $1", item_id)
    return {"message": f"Item #{item_id} deleted"}


# ── Orders ───────────────────────────────────────────────────

@app.get("/api/orders")
async def list_orders(status: str = "", limit: int = 50):
    pool = await get_db()
    query = "SELECT * FROM orders WHERE 1=1"
    params = []
    idx = 1
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"orders": [dict(r) for r in rows]}


@app.put("/api/orders/{order_id}")
async def update_order(order_id: int, request: Request):
    data = await request.json()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    allowed_fields = ("customer_phone", "customer_address", "status", "notes", "items", "total")
    for field in allowed_fields:
        if field in data:
            await pool.execute(f"UPDATE orders SET {field} = $1, updated_at = $2 WHERE id = $3", data[field], now, order_id)
    return {"success": True}


@app.delete("/api/orders/{order_id}")
async def delete_order(order_id: int):
    pool = await get_db()
    await pool.execute("DELETE FROM orders WHERE id = $1", order_id)
    return {"success": True}


@app.post("/api/orders/{order_id}/decrease-stock")
async def decrease_stock_for_order(order_id: int):
    """When order moves to ready_for_send, decrease stock for the product code."""
    pool = await get_db()
    order = await pool.fetchrow("SELECT items FROM orders WHERE id = $1", order_id)
    if not order:
        raise HTTPException(status_code=404)
    items_raw = order["items"].strip().upper()
    import re as _re
    code_match = _re.search(r'(FP|TP|FD|TD)\d+', items_raw)
    item_code = code_match.group(0) if code_match else items_raw
    await pool.execute(
        "UPDATE inventory SET stock = GREATEST(0, stock - 1), updated_at = $1 WHERE UPPER(code) = $2 AND stock > 0",
        datetime.now(timezone.utc).isoformat(), item_code,
    )
    return {"success": True, "code": item_code}


# ── Product Pairs (same print, different style) ─────────────

@app.get("/api/pairs")
async def list_pairs():
    pool = await get_db()
    rows = await pool.fetch("SELECT * FROM product_pairs ORDER BY code_a")
    return {"pairs": [dict(r) for r in rows]}


@app.post("/api/pairs")
async def add_pair(request: Request):
    data = await request.json()
    code_a = data["code_a"].upper().strip()
    code_b = data["code_b"].upper().strip()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Find all codes already connected to A or B (transitive group)
    group = set([code_a, code_b])
    to_check = [code_a, code_b]
    while to_check:
        current = to_check.pop(0)
        rows = await pool.fetch("SELECT code_b FROM product_pairs WHERE code_a = $1", current)
        for r in rows:
            if r["code_b"] not in group:
                group.add(r["code_b"])
                to_check.append(r["code_b"])

    # Connect ALL members of the group to each other
    group = sorted(group)
    for i, a in enumerate(group):
        for b in group[i+1:]:
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, created_at) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                a, b, now,
            )
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, created_at) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                b, a, now,
            )

    return {"success": True, "group": group}


@app.delete("/api/pairs/{pair_id}")
async def delete_pair(pair_id: int):
    pool = await get_db()
    row = await pool.fetchrow("SELECT code_a, code_b FROM product_pairs WHERE id = $1", pair_id)
    if row:
        await pool.execute("DELETE FROM product_pairs WHERE (code_a=$1 AND code_b=$2) OR (code_a=$2 AND code_b=$1)", row["code_a"], row["code_b"])
    return {"success": True}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agents": ["support_sales", "marketing"], "version": "0.2.0"}


# ── Dashboard (admin home) ──────────────────

@app.get("/api/dashboard")
async def dashboard():
    """At-a-glance summary for the admin home: today's orders / revenue,
    low-stock alerts, a 7-day sales sparkline, and a recent-activity feed.
    All counts come from the existing tables — no new writes or migrations."""
    pool = await get_db()
    now = datetime.now(timezone.utc)
    today_prefix = now.date().isoformat()
    seven_days_ago = (now - timedelta(days=6)).date().isoformat()

    today_orders = await pool.fetch(
        "SELECT id, total FROM orders WHERE created_at LIKE $1",
        f"{today_prefix}%",
    )
    today_revenue = sum(float(o["total"] or 0) for o in today_orders)

    low_stock = await pool.fetch(
        "SELECT id, code, model, size, stock, price, image_url FROM inventory "
        "WHERE stock <= 1 AND image_url != '' ORDER BY stock ASC, code ASC LIMIT 12"
    )

    # Totals (lifetime-ish) for headline cards
    totals = await pool.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM orders) AS total_orders, "
        "  (SELECT COALESCE(SUM(total), 0) FROM orders) AS total_revenue, "
        "  (SELECT COUNT(*) FROM inventory WHERE stock > 0) AS in_stock, "
        "  (SELECT COUNT(DISTINCT conversation_id) FROM messages WHERE conversation_id NOT LIKE 'debug_%') AS conversations"
    )

    recent_orders = await pool.fetch(
        "SELECT id, total, created_at FROM orders WHERE created_at >= $1 ORDER BY created_at",
        seven_days_ago,
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
        "FROM orders ORDER BY created_at DESC LIMIT 8"
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


# ── Insights (admin signals) ────────────────────────────────

@app.get("/api/insights/complaints")
async def insights_complaints():
    """Customer messages that look like complaints (wrong match, wants operator)."""
    from src.insights import list_complaints
    return {"complaints": await list_complaints()}


@app.get("/api/insights/faqs")
async def insights_faqs():
    """Most frequent customer questions — candidates for canned answers."""
    from src.insights import list_faq_candidates
    return {"faqs": await list_faq_candidates()}


@app.get("/api/insights/product-requests")
async def insights_product_requests():
    """Customer asks for categories we don't carry — stocking hints for owner."""
    from src.insights import list_product_requests
    return {"requests": await list_product_requests()}


@app.post("/api/reindex")
async def reindex_catalog(full: bool = False):
    """Re-index product catalog embeddings. Use full=true to wipe and re-embed all
    products; otherwise only new/missing ones are indexed."""
    from src.image_match import index_all_products
    pool = await get_db()
    if full:
        await pool.execute("DELETE FROM product_embeddings")
    result = await index_all_products()
    return result


@app.post("/api/clear-conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Wipe messages, AI hints, and tokens for one conversation so the bot
    starts fresh. Useful for re-testing the photo flow as a known customer."""
    pool = await get_db()
    stats: dict = {}
    stats["messages"] = await pool.execute("DELETE FROM messages WHERE conversation_id = $1", conversation_id)
    stats["ai_photo_hints"] = await pool.execute("DELETE FROM ai_photo_hints WHERE conversation_id = $1", conversation_id)
    try:
        stats["confirm_tokens"] = await pool.execute("DELETE FROM confirm_tokens WHERE conversation_id = $1", conversation_id)
    except Exception:
        pass
    try:
        stats["conversation"] = await pool.execute("DELETE FROM conversations WHERE id = $1", conversation_id)
    except Exception:
        pass
    return {"ok": True, "conversation_id": conversation_id, "stats": stats}


@app.post("/api/reindex/{code}")
async def reindex_one(code: str):
    """Re-index a single product by code (useful when stored embedding is wrong)."""
    from src.image_match import index_product
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT id, code, model, size, image_url, image_url_back FROM inventory WHERE UPPER(code) = UPPER($1) LIMIT 1",
        code,
    )
    if not row:
        return {"ok": False, "message": f"Code {code} not found"}
    await pool.execute("DELETE FROM product_embeddings WHERE UPPER(code) = UPPER($1)", code)
    ok = await index_product(row["id"], row["code"], row["model"], row["size"], row["image_url"], row.get("image_url_back", ""))
    return {"ok": ok, "code": row["code"]}


# ── Extra Photos (lifestyle/marketing) ──────────────────────

@app.get("/api/extra-photos")
async def list_extra_photos(code: str = ""):
    pool = await get_db()
    if code:
        rows = await pool.fetch("SELECT * FROM product_extra_photos WHERE code = $1 ORDER BY created_at DESC", code)
    else:
        rows = await pool.fetch("SELECT * FROM product_extra_photos ORDER BY code, created_at DESC")
    return {"photos": [dict(r) for r in rows]}


@app.post("/api/extra-photos")
async def add_extra_photo(
    code: str = Form(...),
    image: UploadFile = File(...),
):
    """Upload a lifestyle/marketing photo for a product."""
    image_url = save_uploaded_image(image, f"extra_{code}")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO product_extra_photos (inventory_id, code, image_url, photo_type, created_at) VALUES ((SELECT id FROM inventory WHERE UPPER(code) = UPPER($1) LIMIT 1), $1, $2, 'lifestyle', $3) RETURNING id",
        code.upper(), image_url, now,
    )

    # Auto-index the new photo for AI matching
    try:
        from src.vision_match import index_extra_photo
        await index_extra_photo(code.upper(), image_url)
    except Exception as e:
        print(f"[EXTRA] Indexing failed: {e}")

    return {"id": row["id"], "image_url": image_url, "code": code}


@app.delete("/api/extra-photos/{photo_id}")
async def delete_extra_photo(photo_id: int):
    pool = await get_db()
    await pool.execute("DELETE FROM product_extra_photos WHERE id = $1", photo_id)
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

    # Always reseed knowledge base (small, no user edits)
    count_row = await pool.fetchrow("SELECT COUNT(*) as c FROM knowledge_base")
    kb_count = count_row["c"]
    if kb_count > 0:
        await pool.execute("DELETE FROM knowledge_base")

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
            "INSERT INTO knowledge_base (question, answer, category, created_at) VALUES ($1, $2, $3, $4)",
            q, a, cat, now,
        )

    # Seed inventory — only add products that don't exist yet (by code)
    seed_file = Path(__file__).parent / "seed_inventory.json"
    if seed_file.exists():
        existing = await pool.fetch("SELECT code FROM inventory")
        existing_codes = {row["code"] for row in existing}

        items = json.loads(seed_file.read_text())
        for item in items:
            code = item.get("code", "")
            if code and code not in existing_codes:
                await pool.execute(
                    "INSERT INTO inventory (product_name, model, size, color, style, code, tags, price, stock, image_url, image_url_back, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
                    item["product_name"], item["model"], item["size"], item.get("color", ""), item.get("style", ""),
                    code, item.get("tags", ""), item["price"], item["stock"],
                    item.get("image_url", ""), item.get("image_url_back", ""), now, now,
                )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", API_PORT))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("RAILWAY_ENVIRONMENT") is None)
