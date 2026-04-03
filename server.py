"""Tissu Agent Server.

AI-powered sales agent for Tissu Shop. Facebook Messenger bot with
Gemini Vision, WhatsApp owner notifications, and admin panel.
"""
import os
import json
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import API_HOST, API_PORT
from src.db import init_db, get_db
from src.models import ChatRequest, ChatResponse, LeadCreate, ContentCreate
from src.engine import run_agent
from src.agents.support_sales import get_support_sales_agent
from src.agents.marketing import get_marketing_agent
from src.channels import get_adapter, ADAPTERS
from src.webhooks.facebook import router as fb_router
from src.webhooks.whatsapp import router as wa_router
from src.vision import preload_product_images


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_knowledge_base()
    # Download product images into memory for fast visual comparison
    await preload_product_images()
    yield


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
    agent = get_support_sales_agent()
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
    """Universal webhook endpoint for any channel (N8N routes here)."""
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


# ── Data Endpoints ───────────────────────────────────────────

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


@app.post("/api/inventory/swap-images")
async def swap_images(request: Request):
    """Swap images between two inventory slots (drag & drop support)."""
    data = await request.json()
    from_id = data["from_id"]
    from_side = data["from_side"]
    to_id = data["to_id"]
    to_side = data["to_side"]

    db = await get_db()
    try:
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


# ── Orders ───────────────────────────────────────────────────

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
    return {"status": "ok", "agents": ["support_sales", "marketing"], "version": "0.2.0"}


# ── Seed Data ────────────────────────────────────────────────

async def seed_knowledge_base():
    """Seed Tissu Shop knowledge base and starter inventory."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as c FROM knowledge_base")
        count = (await cursor.fetchone())["c"]
        if count > 0:
            return

        now = datetime.now(timezone.utc).isoformat()

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

        seed_file = Path(__file__).parent / "seed_inventory.json"
        if seed_file.exists():
            items = json.loads(seed_file.read_text())
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
