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

import asyncpg
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import API_HOST, API_PORT
from src.db import init_db, get_db, close_pool, DEFAULT_TENANT_ID


def get_tenant_id(request: Request) -> str:
    """Pull the tenant_id that the APIKeyMiddleware stashed on the request.
    Falls back to DEFAULT_TENANT_ID for non-/api paths and exempt endpoints,
    so internal callers and Meta webhooks get a sensible value."""
    return getattr(request.state, "tenant_id", DEFAULT_TENANT_ID)
from src.models import ChatRequest, ChatResponse, LeadCreate, ContentCreate
from src.engine import run_agent
from src.agents.support_sales import get_support_sales_agent
from src.agents.marketing import get_marketing_agent
from src.channels import get_adapter, ADAPTERS
from src.webhooks.facebook import router as fb_router
from src.webhooks.whatsapp import router as wa_router
from src.api.storefront import router as storefront_router
from src.auth import APIKeyMiddleware


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

# API key guard on /api/*. Must be added AFTER CORS so preflight (OPTIONS)
# requests still get the CORS headers before we check the key.
app.add_middleware(APIKeyMiddleware)

# Include webhook routers
app.include_router(fb_router)
app.include_router(wa_router)
# Public storefront read API — also under /api/* so it goes through the
# same X-API-Key gate; the router itself re-uses the tenant_id from the
# middleware to scope every query.
app.include_router(storefront_router)


# ── Pages ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    html_path = Path(__file__).parent / "chat.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    html_path = Path(__file__).parent / "admin.html"
    # Always serve fresh — the admin HTML is small and iterating on UX
    # feels broken when browsers cache the previous revision for hours.
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── Legal pages (public) ────────────────────────
# Required by Meta App Review and by customer-facing trust. These are
# plain static HTML served from /legal/. Browsers can cache them.

def _legal_page(filename: str) -> HTMLResponse:
    path = Path(__file__).parent / "legal" / filename
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    return _legal_page("privacy.html")


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    return _legal_page("terms.html")


@app.get("/data-deletion", response_class=HTMLResponse)
async def data_deletion_page():
    return _legal_page("data-deletion.html")


# ── Meta Data-Deletion Callback ─────────────────
# Meta calls this when a user removes the app through Facebook's
# "Apps and Websites" panel. We must respond with a URL + confirmation
# code so Meta can show the user a status page. The heavy lifting
# (actually deleting the rows) is queued via the tickets table and the
# owner processes it manually from the admin insights.

@app.post("/api/meta/data-deletion")
async def meta_data_deletion(request: Request):
    import base64
    import hashlib
    import hmac
    import urllib.parse

    form = await request.form()
    signed_request = form.get("signed_request", "")
    user_id = ""
    if signed_request and "." in signed_request:
        try:
            _sig, payload_b64 = signed_request.split(".", 1)
            # base64url-decode with padding fix
            pad = "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad).decode("utf-8"))
            user_id = str(payload.get("user_id", ""))
        except Exception as e:
            print(f"[DATA-DELETE] Could not parse signed_request: {e}", flush=True)

    # Log the request so the owner can act on it. Uses the tickets table
    # which already surfaces in the admin Insights section.
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    try:
        await pool.execute(
            "INSERT INTO tickets (subject, description, status, priority, customer_email, conversation_id, tenant_id, created_at, updated_at) "
            "VALUES ($1, $2, 'open', 'urgent', $3, $4, $5, $6, $6)",
            "[DATA-DELETION] Meta user requested deletion",
            json.dumps({"user_id": user_id, "source": "meta_callback"}, ensure_ascii=False),
            "", f"meta_user_{user_id}" if user_id else "meta_user_unknown",
            DEFAULT_TENANT_ID, now,
        )
    except Exception as e:
        print(f"[DATA-DELETE] Failed to log ticket: {e}", flush=True)

    # Per Meta spec, respond with a URL where the user can check status
    # plus an opaque confirmation code.
    code = user_id or "anon"
    public = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app").rstrip("/")
    return {
        "url": f"{public}/data-deletion?code={urllib.parse.quote(code)}",
        "confirmation_code": code,
    }


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
async def list_leads(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM leads WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"leads": [dict(r) for r in rows]}


@app.post("/api/leads")
async def create_lead(lead: LeadCreate, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO leads (name, email, company, phone, source, notes, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
        lead.name, lead.email, lead.company, lead.phone, lead.source, lead.notes, tenant_id, now, now,
    )
    return {"id": row["id"], "message": "Lead created"}


@app.get("/api/tickets")
async def list_tickets(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM tickets WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"tickets": [dict(r) for r in rows]}


@app.get("/api/content")
async def list_content(content_type: str = "", status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM content WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
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
async def create_content(item: ContentCreate, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO content (title, body, content_type, tags, scheduled_at, status, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, 'draft', $6, $7, $8) RETURNING id",
        item.title, item.body, item.content_type, json.dumps(item.tags), item.scheduled_at, tenant_id, now, now,
    )
    return {"id": row["id"], "message": "Content created"}


@app.get("/api/conversations")
async def list_conversations(agent_type: str = "", limit: int = 100, tenant_id: str = Depends(get_tenant_id)):
    """List recent conversations with message count and a snippet of the
    last customer message. Powers the admin ჩატები tab."""
    pool = await get_db()
    query = """
        SELECT c.id, c.agent_type, c.updated_at, c.created_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count,
               (SELECT m.content FROM messages m
                 WHERE m.conversation_id = c.id AND m.role = 'user'
                 ORDER BY m.created_at DESC LIMIT 1) AS last_user_msg,
               (SELECT m.content FROM messages m
                 WHERE m.conversation_id = c.id AND m.role IN ('assistant','model')
                 ORDER BY m.created_at DESC LIMIT 1) AS last_bot_msg
        FROM conversations c
        WHERE c.tenant_id = $1 AND c.id NOT LIKE 'debug_%'
    """
    params: list = [tenant_id]
    idx = 2
    if agent_type:
        query += f" AND c.agent_type = ${idx}"
        params.append(agent_type)
        idx += 1
    query += f" ORDER BY c.updated_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    from src.insights import clean_user_message
    out = []
    for r in rows:
        last_user = clean_user_message(r["last_user_msg"] or "")
        out.append({
            "id": r["id"],
            "agent_type": r["agent_type"],
            "updated_at": r["updated_at"],
            "created_at": r["created_at"],
            "msg_count": int(r["msg_count"] or 0),
            "last_user_msg": last_user[:140],
            "last_bot_msg": (r["last_bot_msg"] or "")[:140],
        })
    return {"conversations": out}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    # Verify the conversation belongs to this tenant before returning its
    # messages — prevents a tenant with a valid key from reading another
    # tenant's chat history by guessing IDs.
    owner = await pool.fetchrow(
        "SELECT tenant_id FROM conversations WHERE id = $1", conversation_id,
    )
    if not owner or owner["tenant_id"] != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = await pool.fetch(
        "SELECT role, content, created_at FROM messages "
        "WHERE conversation_id = $1 AND tenant_id = $2 ORDER BY created_at",
        conversation_id, tenant_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": [dict(r) for r in rows]}


@app.get("/api/knowledge")
async def list_knowledge(category: str = "", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM knowledge_base WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if category:
        query += f" AND category = ${idx}"
        params.append(category)
        idx += 1
    rows = await pool.fetch(query, *params)
    return {"articles": [dict(r) for r in rows]}


@app.post("/api/knowledge")
async def add_knowledge(question: str, answer: str, category: str = "general", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO knowledge_base (question, answer, category, tenant_id, created_at) "
        "VALUES ($1, $2, $3, $4, $5) RETURNING id",
        question, answer, category, tenant_id, now,
    )
    return {"id": row["id"], "message": "Knowledge article added"}


# ── Inventory Endpoints ──────────────────────────────────────

def _cloudinary_configured() -> bool:
    """Cloudinary wins when CLOUDINARY_URL (or CLOUD/API key triplet) is set.
    Railway's filesystem is ephemeral — without Cloudinary, every deploy
    would wipe admin-uploaded photos. Local dev falls back to /static."""
    return bool(os.getenv("CLOUDINARY_URL")) or (
        os.getenv("CLOUDINARY_CLOUD_NAME") and os.getenv("CLOUDINARY_API_KEY") and os.getenv("CLOUDINARY_API_SECRET")
    )


def _upload_to_cloudinary(upload: UploadFile, public_id: str):
    """Upload to Cloudinary and return the secure HTTPS URL. Returns None on failure."""
    try:
        import cloudinary
        import cloudinary.uploader
        if os.getenv("CLOUDINARY_URL"):
            cloudinary.config(secure=True)
        else:
            cloudinary.config(
                cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
                api_key=os.getenv("CLOUDINARY_API_KEY"),
                api_secret=os.getenv("CLOUDINARY_API_SECRET"),
                secure=True,
            )
        ext = Path(upload.filename or "").suffix.lower()
        file_obj = upload.file
        # HEIC from iOS needs converting to JPEG before Cloudinary accepts it.
        if ext in ('.heic', '.heif'):
            from io import BytesIO
            img = Image.open(upload.file)
            buf = BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=90)
            buf.seek(0)
            file_obj = buf
        result = cloudinary.uploader.upload(
            file_obj,
            folder="tissu/uploads",
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        return result.get("secure_url") or result.get("url")
    except Exception as e:
        print(f"[UPLOAD] Cloudinary failed, falling back to local: {e}", flush=True)
        return None


def save_uploaded_image(upload: UploadFile, prefix: str) -> str:
    """Persist an uploaded image and return a URL the frontend can render.

    Prefers Cloudinary when credentials are configured (prod / Railway) so
    images survive deploys; falls back to the local ``static/products``
    directory for dev environments without Cloudinary set up.
    """
    filename_base = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if _cloudinary_configured():
        url = _upload_to_cloudinary(upload, public_id=filename_base)
        if url:
            return url
        # Rewind the file pointer so the local fallback still works when
        # Cloudinary rejected the upload mid-stream.
        try:
            upload.file.seek(0)
        except Exception:
            pass

    save_dir = Path(__file__).parent / "static" / "products"
    save_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(upload.filename or "").suffix.lower()
    if ext in ('.heic', '.heif'):
        img = Image.open(upload.file)
        filename = f"{filename_base}.jpg"
        img.convert("RGB").save(save_dir / filename, "JPEG", quality=85)
    else:
        filename = f"{filename_base}{ext or '.jpg'}"
        with open(save_dir / filename, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    return f"/static/products/{filename}"


# ── Categories registry ────────────────────────────────────

@app.get("/api/categories")
async def list_categories(tenant_id: str = Depends(get_tenant_id)):
    """Return every catalog category with its field schema and live count.
    Driven by the `categories` table — new rows show up in the admin
    sidebar the moment they're inserted."""
    pool = await get_db()
    rows = await pool.fetch("""
        SELECT c.slug, c.name, c.emoji, c.fields, c.sort_order,
               COALESCE(cnt.n, 0) AS count
        FROM categories c
        LEFT JOIN (
            SELECT category, COUNT(*) AS n FROM inventory
            WHERE tenant_id = $1
            GROUP BY category
        ) cnt ON cnt.category = c.slug
        WHERE c.tenant_id = $1
        ORDER BY c.sort_order ASC, c.name ASC
    """, tenant_id)
    out = []
    for r in rows:
        fields = r["fields"]
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except Exception:
                fields = []
        out.append({
            "slug": r["slug"],
            "name": r["name"],
            "emoji": r["emoji"],
            "fields": fields,
            "sort_order": r["sort_order"],
            "count": int(r["count"] or 0),
        })
    return {"categories": out}


@app.post("/api/categories")
async def add_category(request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Register a new product category. Body: {slug, name, emoji, fields}.
    `fields` is a list of {key, label} objects describing the per-product
    custom attributes the admin UI should render."""
    import re as _re
    data = await request.json()
    slug = (data.get("slug") or "").strip().lower()
    name = (data.get("name") or "").strip()
    emoji = (data.get("emoji") or "📦").strip()[:8]
    fields = data.get("fields") or []
    if not slug or not name:
        raise HTTPException(status_code=400, detail="slug და name აუცილებელია")
    if not _re.match(r"^[a-z][a-z0-9_-]{1,31}$", slug):
        raise HTTPException(status_code=400, detail="slug არ ვარგა — მხოლოდ a-z, 0-9, _ ან -")
    cleaned_fields = []
    for f in fields if isinstance(fields, list) else []:
        if isinstance(f, dict) and f.get("key") and f.get("label"):
            cleaned_fields.append({"key": str(f["key"]).strip(), "label": str(f["label"]).strip()})
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    try:
        await pool.execute(
            """INSERT INTO categories (slug, name, emoji, fields, sort_order, tenant_id, created_at)
               VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)""",
            slug, name, emoji, json.dumps(cleaned_fields), 100, tenant_id, now,
        )
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=409, detail="ეს slug უკვე არსებობს")
    return {"ok": True, "slug": slug}


@app.delete("/api/categories/{slug}")
async def delete_category(slug: str, tenant_id: str = Depends(get_tenant_id)):
    """Delete a category — refuses when any inventory row still uses it,
    so the owner doesn't orphan products by accident."""
    if slug in ("bag", "necklace"):
        raise HTTPException(status_code=400, detail="ძირითადი კატეგორიები არ წაიშლება")
    pool = await get_db()
    in_use = await pool.fetchval(
        "SELECT COUNT(*) FROM inventory WHERE tenant_id = $1 AND category = $2",
        tenant_id, slug,
    )
    if int(in_use or 0) > 0:
        raise HTTPException(status_code=409, detail=f"ამ კატეგორიაში {in_use} პროდუქტია, ჯერ გადაიტანე სხვაგან")
    await pool.execute(
        "DELETE FROM categories WHERE slug = $1 AND tenant_id = $2",
        slug, tenant_id,
    )
    return {"ok": True}


@app.put("/api/categories/{slug}")
async def update_category(slug: str, request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Update the name / emoji / fields of an existing category."""
    data = await request.json()
    pool = await get_db()
    updates = []
    params: list = []
    idx = 1
    for key in ("name", "emoji"):
        if key in data and data[key] is not None:
            updates.append(f"{key} = ${idx}")
            params.append(str(data[key]).strip())
            idx += 1
    if "fields" in data and isinstance(data["fields"], list):
        cleaned = [
            {"key": str(f["key"]).strip(), "label": str(f["label"]).strip()}
            for f in data["fields"]
            if isinstance(f, dict) and f.get("key") and f.get("label")
        ]
        updates.append(f"fields = ${idx}::jsonb")
        params.append(json.dumps(cleaned))
        idx += 1
    if not updates:
        return {"ok": True, "updated": False}
    params.append(slug)
    params.append(tenant_id)
    await pool.execute(
        f"UPDATE categories SET {', '.join(updates)} "
        f"WHERE slug = ${idx} AND tenant_id = ${idx + 1}",
        *params,
    )
    return {"ok": True}


@app.put("/api/inventory/{item_id}/attr")
async def set_inventory_attr(item_id: int, request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Merge a single key/value into inventory.attrs — used by admin to
    save the per-field text inputs for category-driven products."""
    data = await request.json()
    key = (data.get("key") or "").strip()
    value = data.get("value", "")
    if not key:
        raise HTTPException(status_code=400, detail="key required")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE inventory SET attrs = attrs || jsonb_build_object($1::text, $2::text), updated_at = $3 "
        "WHERE id = $4 AND tenant_id = $5",
        key, str(value), now, item_id, tenant_id,
    )
    return {"ok": True}


@app.get("/api/inventory")
async def list_inventory(category: str = "", tenant_id: str = Depends(get_tenant_id)):
    """List inventory. Omit category to get everything (admin); pass
    `?category=bag` or `?category=necklace` to scope results — bot-facing
    callers should always pass category='bag' to avoid cross-category leakage.

    Each row gets an `angles` field with the product's extra angle photos
    (rows from product_extra_photos where photo_type='angle')."""
    pool = await get_db()
    if category:
        rows = await pool.fetch(
            "SELECT * FROM inventory WHERE tenant_id = $1 AND category = $2 ORDER BY model, size",
            tenant_id, category,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM inventory WHERE tenant_id = $1 ORDER BY model, size",
            tenant_id,
        )

    inv_ids = [r["id"] for r in rows]
    angles_by_id: dict = {}
    if inv_ids:
        try:
            angle_rows = await pool.fetch(
                "SELECT id, inventory_id, image_url FROM product_extra_photos "
                "WHERE tenant_id = $1 AND photo_type = 'angle' "
                "AND inventory_id = ANY($2::int[]) ORDER BY created_at ASC",
                tenant_id, inv_ids,
            )
            for ar in angle_rows:
                angles_by_id.setdefault(ar["inventory_id"], []).append(
                    {"id": ar["id"], "image_url": ar["image_url"]}
                )
        except Exception as e:
            # Table might not exist yet on very old schemas — degrade to no angles.
            print(f"[inventory] angles query skipped: {e}", flush=True)

    out = []
    for r in rows:
        d = dict(r)
        d["angles"] = angles_by_id.get(r["id"], [])
        out.append(d)
    return {"inventory": out}


@app.post("/api/inventory/{item_id}/angle")
async def add_product_angle(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    """Upload an extra angle photo for a product — goes into
    product_extra_photos with photo_type='angle' so it lives alongside the
    existing lifestyle photos but is clearly distinguishable. Returns the
    new row's id + image_url for the frontend to insert inline."""
    image_url = save_uploaded_image(image, f"angle_{item_id}")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "SELECT code FROM inventory WHERE id = $1 AND tenant_id = $2",
        item_id, tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    code = row["code"] or ""
    new = await pool.fetchrow(
        "INSERT INTO product_extra_photos (inventory_id, code, image_url, photo_type, tenant_id, created_at) "
        "VALUES ($1, $2, $3, 'angle', $4, $5) RETURNING id",
        item_id, code, image_url, tenant_id, now,
    )
    return {"id": new["id"], "image_url": image_url}


@app.delete("/api/inventory/angle/{photo_id}")
async def delete_product_angle(photo_id: int, tenant_id: str = Depends(get_tenant_id)):
    """Remove a single angle photo by its extra-photos row id."""
    pool = await get_db()
    await pool.execute(
        "DELETE FROM product_extra_photos "
        "WHERE id = $1 AND tenant_id = $2 AND photo_type = 'angle'",
        photo_id, tenant_id,
    )
    return {"ok": True}


@app.post("/api/inventory")
async def add_inventory(
    product_name: str = Form(...),
    model: str = Form(""),
    size: str = Form(""),
    price: float = Form(...),
    stock: int = Form(...),
    color: str = Form(""),
    style: str = Form(""),
    category: str = Form("bag"),
    attrs_json: str = Form("{}"),
    image: UploadFile = File(None),
    tenant_id: str = Depends(get_tenant_id),
):
    image_url = ""
    if image and image.filename:
        prefix = f"{model}_{size}_{color}".replace(" ", "_")
        image_url = save_uploaded_image(image, prefix)

    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Auto-generate a code for non-bag categories using the first two
    # uppercase letters of the slug (e.g. necklace → NE, belt → BE). Bag
    # codes stay on the legacy FP/TP/FD/TD convention and are not
    # generated here. Codes are unique per tenant.
    new_code = ""
    if category and category != "bag":
        prefix = (category[:2] or "XX").upper()
        max_n = await pool.fetchval(
            "SELECT COALESCE(MAX(CAST(SUBSTRING(code, 3) AS INTEGER)), 0) "
            "FROM inventory WHERE tenant_id = $1 AND category = $2 "
            "AND code ~ ('^' || $3 || '[0-9]+$')",
            tenant_id, category, prefix,
        )
        new_code = f"{prefix}{int(max_n or 0) + 1}"

    try:
        attrs_value = attrs_json if attrs_json else "{}"
        json.loads(attrs_value)  # validate
    except Exception:
        attrs_value = "{}"

    row = await pool.fetchrow(
        "INSERT INTO inventory (product_name, model, size, color, style, code, category, attrs, price, stock, image_url, tenant_id, created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13, $14) RETURNING id",
        product_name, model, size, color, style, new_code, category, attrs_value, price, stock, image_url, tenant_id, now, now,
    )
    return {"id": row["id"], "code": new_code, "category": category, "image_url": image_url, "message": "Product added"}


@app.put("/api/inventory/{item_id}")
async def update_inventory(
    item_id: int,
    stock: int = None, price: float = None, model: str = None,
    size: str = None, color: str = None, tags: str = None,
    on_sale: bool = None, sale_price: float = None,
    tenant_id: str = Depends(get_tenant_id),
):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    if stock is not None:
        await pool.execute("UPDATE inventory SET stock = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", stock, now, item_id, tenant_id)
    if price is not None:
        await pool.execute("UPDATE inventory SET price = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", price, now, item_id, tenant_id)
    if model is not None:
        await pool.execute("UPDATE inventory SET model = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", model, now, item_id, tenant_id)
    if size is not None:
        new_price = 74 if "დიდი" in size else 69
        await pool.execute("UPDATE inventory SET size = $1, price = $2, updated_at = $3 WHERE id = $4 AND tenant_id = $5", size, new_price, now, item_id, tenant_id)
    if color is not None:
        await pool.execute("UPDATE inventory SET color = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", color, now, item_id, tenant_id)
    if tags is not None:
        await pool.execute("UPDATE inventory SET tags = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", tags, now, item_id, tenant_id)
    if on_sale is not None:
        await pool.execute("UPDATE inventory SET on_sale = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", on_sale, now, item_id, tenant_id)
    if sale_price is not None:
        # Passing a literal 0 or negative clears the sale price back to null.
        sp = sale_price if sale_price > 0 else None
        await pool.execute("UPDATE inventory SET sale_price = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", sp, now, item_id, tenant_id)
    return {"message": f"Item #{item_id} updated"}


@app.post("/api/inventory/{item_id}/image")
async def upload_product_image(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    image_url = save_uploaded_image(image, f"product_{item_id}_front")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", image_url, now, item_id, tenant_id)
    return {"image_url": image_url}


@app.post("/api/inventory/{item_id}/image_back")
async def upload_back_image(item_id: int, image: UploadFile = File(...), tenant_id: str = Depends(get_tenant_id)):
    image_url = save_uploaded_image(image, f"product_{item_id}_back")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", image_url, now, item_id, tenant_id)
    return {"image_url": image_url}


@app.post("/api/inventory/swap-images")
async def swap_images(request: Request, tenant_id: str = Depends(get_tenant_id)):
    """Swap images between two inventory slots (drag & drop support)."""
    data = await request.json()
    from_id = data["from_id"]
    from_side = data["from_side"]
    to_id = data["to_id"]
    to_side = data["to_side"]

    pool = await get_db()
    r1 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1 AND tenant_id = $2", from_id, tenant_id)
    r2 = await pool.fetchrow("SELECT image_url, image_url_back FROM inventory WHERE id = $1 AND tenant_id = $2", to_id, tenant_id)
    if not r1 or not r2:
        raise HTTPException(status_code=404)

    from_url = r1["image_url"] if from_side == "front" else r1["image_url_back"]
    to_url = r2["image_url"] if to_side == "front" else r2["image_url_back"]

    now = datetime.now(timezone.utc).isoformat()
    from_col = "image_url" if from_side == "front" else "image_url_back"
    to_col = "image_url" if to_side == "front" else "image_url_back"

    await pool.execute(f"UPDATE inventory SET {from_col} = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", to_url, now, from_id, tenant_id)
    await pool.execute(f"UPDATE inventory SET {to_col} = $1, updated_at = $2 WHERE id = $3 AND tenant_id = $4", from_url, now, to_id, tenant_id)
    return {"message": "Images swapped"}


@app.post("/api/inventory/{item_id}/clear-image")
async def clear_image(item_id: int, side: str = "back", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    col = "image_url" if side == "front" else "image_url_back"
    await pool.execute(f"UPDATE inventory SET {col} = '', updated_at = $1 WHERE id = $2 AND tenant_id = $3", now, item_id, tenant_id)
    return {"message": f"{side} image cleared"}


@app.post("/api/inventory/{item_id}/remove-back")
async def remove_back_image(item_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute("UPDATE inventory SET image_url_back = '', updated_at = $1 WHERE id = $2 AND tenant_id = $3", now, item_id, tenant_id)
    return {"message": "Back image removed"}


@app.delete("/api/inventory/{item_id}")
async def delete_inventory(item_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute("DELETE FROM inventory WHERE id = $1 AND tenant_id = $2", item_id, tenant_id)
    return {"message": f"Item #{item_id} deleted"}


# ── Orders ───────────────────────────────────────────────────

@app.get("/api/orders")
async def list_orders(status: str = "", limit: int = 50, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    query = "SELECT * FROM orders WHERE tenant_id = $1"
    params: list = [tenant_id]
    idx = 2
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {"orders": [dict(r) for r in rows]}


@app.put("/api/orders/{order_id}")
async def update_order(order_id: int, request: Request, tenant_id: str = Depends(get_tenant_id)):
    data = await request.json()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    allowed_fields = ("customer_phone", "customer_address", "status", "notes", "items", "total")
    for field in allowed_fields:
        if field in data:
            await pool.execute(
                f"UPDATE orders SET {field} = $1, updated_at = $2 "
                f"WHERE id = $3 AND tenant_id = $4",
                data[field], now, order_id, tenant_id,
            )
    return {"success": True}


@app.delete("/api/orders/{order_id}")
async def delete_order(order_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute(
        "DELETE FROM orders WHERE id = $1 AND tenant_id = $2",
        order_id, tenant_id,
    )
    return {"success": True}


@app.post("/api/orders/{order_id}/decrease-stock")
async def decrease_stock_for_order(order_id: int, tenant_id: str = Depends(get_tenant_id)):
    """When order moves to ready_for_send, decrease stock for the product code."""
    pool = await get_db()
    order = await pool.fetchrow(
        "SELECT items FROM orders WHERE id = $1 AND tenant_id = $2",
        order_id, tenant_id,
    )
    if not order:
        raise HTTPException(status_code=404)
    items_raw = order["items"].strip().upper()
    import re as _re
    code_match = _re.search(r'(FP|TP|FD|TD)\d+', items_raw)
    item_code = code_match.group(0) if code_match else items_raw
    await pool.execute(
        "UPDATE inventory SET stock = GREATEST(0, stock - 1), updated_at = $1 "
        "WHERE tenant_id = $2 AND UPPER(code) = $3 AND stock > 0",
        datetime.now(timezone.utc).isoformat(), tenant_id, item_code,
    )
    return {"success": True, "code": item_code}


# ── Product Pairs (same print, different style) ─────────────

@app.get("/api/pairs")
async def list_pairs(tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT * FROM product_pairs WHERE tenant_id = $1 ORDER BY code_a",
        tenant_id,
    )
    return {"pairs": [dict(r) for r in rows]}


@app.post("/api/pairs")
async def add_pair(request: Request, tenant_id: str = Depends(get_tenant_id)):
    data = await request.json()
    code_a = data["code_a"].upper().strip()
    code_b = data["code_b"].upper().strip()
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Find all codes already connected to A or B (transitive group),
    # scoped to this tenant so different shops' pairs can't merge.
    group = set([code_a, code_b])
    to_check = [code_a, code_b]
    while to_check:
        current = to_check.pop(0)
        rows = await pool.fetch(
            "SELECT code_b FROM product_pairs WHERE tenant_id = $1 AND code_a = $2",
            tenant_id, current,
        )
        for r in rows:
            if r["code_b"] not in group:
                group.add(r["code_b"])
                to_check.append(r["code_b"])

    # Connect ALL members of the group to each other
    group = sorted(group)
    for i, a in enumerate(group):
        for b in group[i+1:]:
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, tenant_id, created_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                a, b, tenant_id, now,
            )
            await pool.execute(
                "INSERT INTO product_pairs (code_a, code_b, tenant_id, created_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                b, a, tenant_id, now,
            )

    return {"success": True, "group": group}


@app.delete("/api/pairs/{pair_id}")
async def delete_pair(pair_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT code_a, code_b FROM product_pairs WHERE id = $1 AND tenant_id = $2",
        pair_id, tenant_id,
    )
    if row:
        await pool.execute(
            "DELETE FROM product_pairs WHERE tenant_id = $1 AND "
            "((code_a=$2 AND code_b=$3) OR (code_a=$3 AND code_b=$2))",
            tenant_id, row["code_a"], row["code_b"],
        )
    return {"success": True}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "agents": ["support_sales", "marketing"], "version": "0.2.0"}


# ── Sale (discounted products) ─────────────

@app.get("/api/sale")
async def list_sale(tenant_id: str = Depends(get_tenant_id)):
    """Return every product currently marked on sale, across all categories.
    Powers the admin Sale tab and the bot when customers ask about discounts."""
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT * FROM inventory WHERE tenant_id = $1 AND on_sale = true AND stock > 0 "
        "ORDER BY category, model, size, code",
        tenant_id,
    )
    return {"sale": [dict(r) for r in rows]}


# ── Dashboard (admin home) ──────────────────

@app.get("/api/dashboard")
async def dashboard(tenant_id: str = Depends(get_tenant_id)):
    """At-a-glance summary for the admin home: today's orders / revenue,
    low-stock alerts, a 7-day sales sparkline, and a recent-activity feed.
    All counts come from the existing tables — no new writes or migrations."""
    pool = await get_db()
    now = datetime.now(timezone.utc)
    today_prefix = now.date().isoformat()
    seven_days_ago = (now - timedelta(days=6)).date().isoformat()

    today_orders = await pool.fetch(
        "SELECT id, total FROM orders WHERE tenant_id = $1 AND created_at LIKE $2",
        tenant_id, f"{today_prefix}%",
    )
    today_revenue = sum(float(o["total"] or 0) for o in today_orders)

    low_stock = await pool.fetch(
        "SELECT id, code, model, size, stock, price, image_url FROM inventory "
        "WHERE tenant_id = $1 AND stock <= 1 AND image_url != '' "
        "ORDER BY stock ASC, code ASC LIMIT 12",
        tenant_id,
    )

    # Totals (lifetime-ish) for headline cards — all scoped to this tenant.
    totals = await pool.fetchrow(
        "SELECT "
        "  (SELECT COUNT(*) FROM orders WHERE tenant_id = $1) AS total_orders, "
        "  (SELECT COALESCE(SUM(total), 0) FROM orders WHERE tenant_id = $1) AS total_revenue, "
        "  (SELECT COUNT(*) FROM inventory WHERE tenant_id = $1 AND stock > 0) AS in_stock, "
        "  (SELECT COUNT(DISTINCT conversation_id) FROM messages "
        "    WHERE tenant_id = $1 AND conversation_id NOT LIKE 'debug_%') AS conversations",
        tenant_id,
    )

    recent_orders = await pool.fetch(
        "SELECT id, total, created_at FROM orders "
        "WHERE tenant_id = $1 AND created_at >= $2 ORDER BY created_at",
        tenant_id, seven_days_ago,
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
        "FROM orders WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT 8",
        tenant_id,
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


# ── Conversation viewer (admin) ─────────────────────────────

@app.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Return every message in a conversation with the system-tag noise
    stripped from customer lines. Used by the admin conversation viewer to
    let the owner replay an entire chat without leaving the UI."""
    from src.insights import clean_user_message
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT id, role, content, created_at "
        "FROM messages WHERE conversation_id = $1 AND tenant_id = $2 "
        "ORDER BY created_at ASC, id ASC",
        conversation_id, tenant_id,
    )
    out = []
    for r in rows:
        content = r["content"] or ""
        role = r["role"]
        # Skip synthetic user-side "[customer sent a photo]" hints — those
        # are system notes we inject for the LLM, not real customer messages.
        if role == "user":
            cleaned = clean_user_message(content)
            if not cleaned:
                continue
            content = cleaned
        out.append({
            "id": r["id"],
            "role": role,
            "content": content,
            "created_at": r["created_at"],
        })
    return {"conversation_id": conversation_id, "messages": out}


# ── Insights (admin signals) ────────────────────────────────

@app.get("/api/insights/complaints")
async def insights_complaints(tenant_id: str = Depends(get_tenant_id)):
    """Customer messages that look like complaints (wrong match, wants operator)."""
    from src.insights import list_complaints
    return {"complaints": await list_complaints(tenant_id=tenant_id)}


@app.get("/api/insights/faqs")
async def insights_faqs(tenant_id: str = Depends(get_tenant_id)):
    """Most frequent customer questions — candidates for canned answers."""
    from src.insights import list_faq_candidates
    return {"faqs": await list_faq_candidates(tenant_id=tenant_id)}


@app.get("/api/insights/product-requests")
async def insights_product_requests(tenant_id: str = Depends(get_tenant_id)):
    """Customer asks for categories we don't carry — stocking hints for owner."""
    from src.insights import list_product_requests
    return {"requests": await list_product_requests(tenant_id=tenant_id)}


@app.post("/api/reindex")
async def reindex_catalog(full: bool = False, tenant_id: str = Depends(get_tenant_id)):
    """Re-index product catalog embeddings. Use full=true to wipe and re-embed all
    products; otherwise only new/missing ones are indexed."""
    from src.image_match import index_all_products
    pool = await get_db()
    if full:
        # Only wipe this tenant's embeddings so a second shop's embeddings
        # keep working during a reindex of another.
        try:
            await pool.execute(
                "DELETE FROM product_embeddings WHERE tenant_id = $1",
                tenant_id,
            )
        except Exception:
            # Fallback if product_embeddings doesn't have tenant_id yet
            # (pre-migration state) — delete all. Safe on single-tenant.
            await pool.execute("DELETE FROM product_embeddings")
    result = await index_all_products()
    return result


@app.post("/api/clear-conversation/{conversation_id}")
async def clear_conversation(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Wipe messages, AI hints, and tokens for one conversation so the bot
    starts fresh. Useful for re-testing the photo flow as a known customer."""
    pool = await get_db()
    stats: dict = {}
    stats["messages"] = await pool.execute(
        "DELETE FROM messages WHERE conversation_id = $1 AND tenant_id = $2",
        conversation_id, tenant_id,
    )
    try:
        stats["ai_photo_hints"] = await pool.execute(
            "DELETE FROM ai_photo_hints WHERE conversation_id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        # ai_photo_hints may not have tenant_id yet on older schemas.
        stats["ai_photo_hints"] = await pool.execute(
            "DELETE FROM ai_photo_hints WHERE conversation_id = $1",
            conversation_id,
        )
    try:
        stats["confirm_tokens"] = await pool.execute(
            "DELETE FROM confirm_tokens WHERE conversation_id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        pass
    try:
        stats["conversation"] = await pool.execute(
            "DELETE FROM conversations WHERE id = $1 AND tenant_id = $2",
            conversation_id, tenant_id,
        )
    except Exception:
        pass
    return {"ok": True, "conversation_id": conversation_id, "stats": stats}


@app.post("/api/reindex/{code}")
async def reindex_one(code: str, tenant_id: str = Depends(get_tenant_id)):
    """Re-index a single product by code (useful when stored embedding is wrong)."""
    from src.image_match import index_product
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT id, code, model, size, image_url, image_url_back FROM inventory "
        "WHERE tenant_id = $1 AND UPPER(code) = UPPER($2) LIMIT 1",
        tenant_id, code,
    )
    if not row:
        return {"ok": False, "message": f"Code {code} not found"}
    try:
        await pool.execute(
            "DELETE FROM product_embeddings WHERE tenant_id = $1 AND UPPER(code) = UPPER($2)",
            tenant_id, code,
        )
    except Exception:
        await pool.execute(
            "DELETE FROM product_embeddings WHERE UPPER(code) = UPPER($1)",
            code,
        )
    ok = await index_product(row["id"], row["code"], row["model"], row["size"], row["image_url"], row.get("image_url_back", ""))
    return {"ok": ok, "code": row["code"]}


# ── Extra Photos (lifestyle/marketing) ──────────────────────

@app.get("/api/extra-photos")
async def list_extra_photos(code: str = "", tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    if code:
        rows = await pool.fetch(
            "SELECT * FROM product_extra_photos WHERE tenant_id = $1 AND code = $2 "
            "ORDER BY created_at DESC",
            tenant_id, code,
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM product_extra_photos WHERE tenant_id = $1 "
            "ORDER BY code, created_at DESC",
            tenant_id,
        )
    return {"photos": [dict(r) for r in rows]}


@app.post("/api/extra-photos")
async def add_extra_photo(
    code: str = Form(...),
    image: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload a lifestyle/marketing photo for a product."""
    image_url = save_uploaded_image(image, f"extra_{code}")
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = await pool.fetchrow(
        "INSERT INTO product_extra_photos (inventory_id, code, image_url, photo_type, tenant_id, created_at) "
        "VALUES ((SELECT id FROM inventory WHERE tenant_id = $1 AND UPPER(code) = UPPER($2) LIMIT 1), "
        "$2, $3, 'lifestyle', $1, $4) RETURNING id",
        tenant_id, code.upper(), image_url, now,
    )

    # Auto-index the new photo for AI matching
    try:
        from src.vision_match import index_extra_photo
        await index_extra_photo(code.upper(), image_url)
    except Exception as e:
        print(f"[EXTRA] Indexing failed: {e}")

    return {"id": row["id"], "image_url": image_url, "code": code}


@app.delete("/api/extra-photos/{photo_id}")
async def delete_extra_photo(photo_id: int, tenant_id: str = Depends(get_tenant_id)):
    pool = await get_db()
    await pool.execute(
        "DELETE FROM product_extra_photos WHERE id = $1 AND tenant_id = $2",
        photo_id, tenant_id,
    )
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

    # Always reseed knowledge base (small, no user edits) for the
    # default tenant. New tenants seed their own KB separately.
    count_row = await pool.fetchrow(
        "SELECT COUNT(*) as c FROM knowledge_base WHERE tenant_id = $1",
        DEFAULT_TENANT_ID,
    )
    kb_count = count_row["c"]
    if kb_count > 0:
        await pool.execute(
            "DELETE FROM knowledge_base WHERE tenant_id = $1",
            DEFAULT_TENANT_ID,
        )

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
            "INSERT INTO knowledge_base (question, answer, category, tenant_id, created_at) "
            "VALUES ($1, $2, $3, $4, $5)",
            q, a, cat, DEFAULT_TENANT_ID, now,
        )

    # Seed inventory — only add products that don't exist yet (by code)
    # for the default tenant.
    seed_file = Path(__file__).parent / "seed_inventory.json"
    if seed_file.exists():
        existing = await pool.fetch(
            "SELECT code FROM inventory WHERE tenant_id = $1",
            DEFAULT_TENANT_ID,
        )
        existing_codes = {row["code"] for row in existing}

        items = json.loads(seed_file.read_text())
        for item in items:
            code = item.get("code", "")
            if code and code not in existing_codes:
                await pool.execute(
                    "INSERT INTO inventory (product_name, model, size, color, style, code, tags, price, stock, image_url, image_url_back, tenant_id, created_at, updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
                    item["product_name"], item["model"], item["size"], item.get("color", ""), item.get("style", ""),
                    code, item.get("tags", ""), item["price"], item["stock"],
                    item.get("image_url", ""), item.get("image_url_back", ""),
                    DEFAULT_TENANT_ID, now, now,
                )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", API_PORT))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=os.getenv("RAILWAY_ENVIRONMENT") is None)
