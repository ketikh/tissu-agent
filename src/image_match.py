"""Tissu Shop — AI Image Matching System.

Uses Gemini Vision to analyze product images and generate descriptive tags,
then Gemini Embeddings to create vectors for similarity search via pgvector.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import httpx
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY
from src.db import get_db

logger = logging.getLogger(__name__)

VISION_MODEL = "gemini-2.5-flash"
CLIP_SERVICE_URL = os.getenv("CLIP_SERVICE_URL", "")  # e.g. https://your-space.hf.space


def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


async def analyze_image(image_bytes: bytes) -> dict:
    """Analyze a product image with Gemini Vision. Returns detailed tags dict."""
    client = _get_client()
    try:
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part(text=(
                    'This is a laptop sleeve/case. Analyze it in extreme detail. Return JSON:\n'
                    '{\n'
                    '  "primary_color": "exact color name (e.g. navy blue, forest green, burgundy, beige)",\n'
                    '  "secondary_color": "second most visible color or none",\n'
                    '  "color_tone": "warm/cool/neutral",\n'
                    '  "brightness": "light/medium/dark",\n'
                    '  "pattern_type": "solid/striped/floral/geometric/abstract/checkered/paisley/animal/tropical",\n'
                    '  "pattern_detail": "describe the specific pattern in 5-10 words",\n'
                    '  "pattern_scale": "small/medium/large",\n'
                    '  "pattern_colors": ["color1", "color2", "color3"],\n'
                    '  "closure_type": "strap/zipper/magnetic/velcro/open",\n'
                    '  "surface_texture": "smooth/rough/woven/quilted/ribbed",\n'
                    '  "material_appearance": "canvas/denim/cotton/linen/synthetic",\n'
                    '  "overall_style": "minimalist/colorful/vintage/modern/bohemian/classic",\n'
                    '  "distinctive_features": "any unique visual element in 5-10 words",\n'
                    '  "visual_description": "detailed 20-30 word description of exact appearance"\n'
                    '}\n'
                    'Be very specific about colors and patterns — two bags that look different MUST have different descriptions.\n'
                    'Only valid JSON, no markdown, no explanation.'
                )),
            ])],
        )
        text = (resp.text or "").strip()
        # Clean markdown formatting if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {}


def _tags_to_text(tags: dict, code: str = "", model: str = "", size: str = "") -> str:
    """Convert tags dict to rich searchable text for embedding."""
    parts = []
    if code:
        parts.append(f"product code {code}")
    if model:
        parts.append(f"model {model}")
    if size:
        parts.append(f"size {size}")

    # Core visual identity
    for key in ("primary_color", "secondary_color", "color_tone", "brightness",
                "pattern_type", "pattern_detail", "pattern_scale",
                "closure_type", "surface_texture", "material_appearance",
                "overall_style", "distinctive_features", "visual_description"):
        val = tags.get(key)
        if val and val != "none":
            parts.append(str(val))

    # Pattern colors
    for c in tags.get("pattern_colors", []):
        parts.append(c)

    # Fallback for old format
    for key in ("color", "pattern", "style", "material_look", "description"):
        val = tags.get(key)
        if val and val not in parts:
            parts.append(str(val))
    for c in tags.get("dominant_colors", []):
        if c not in parts:
            parts.append(c)

    return " ".join(parts)


async def generate_embedding_from_image(image_url: str = "", image_bytes: bytes = b"") -> list[float]:
    """Generate CLIP image embedding via microservice."""
    if not CLIP_SERVICE_URL:
        # Fallback to Gemini text embedding if CLIP not available
        return await _gemini_text_embedding("laptop bag product")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if image_url:
                resp = await client.post(f"{CLIP_SERVICE_URL}/embed", json={"image_url": image_url})
            elif image_bytes:
                import base64
                b64 = base64.b64encode(image_bytes).decode()
                resp = await client.post(f"{CLIP_SERVICE_URL}/embed", json={"image_base64": b64})
            else:
                return []

            if resp.status_code == 200:
                return resp.json()["embedding"]
            else:
                logger.error(f"CLIP service error: {resp.status_code} {resp.text[:100]}")
                return []
    except Exception as e:
        logger.error(f"CLIP service failed: {e}")
        return []


async def _gemini_text_embedding(text: str) -> list[float]:
    """Fallback: Gemini text embedding."""
    try:
        client = _get_client()
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return list(result.embeddings[0].values)
    except Exception as e:
        logger.error(f"Gemini embedding failed: {e}")
        return []


async def index_product(inventory_id: int, code: str, model: str, size: str, image_url: str) -> bool:
    """Analyze and index a single product image."""
    try:
        # Download image
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(image_url)
            if resp.status_code != 200:
                logger.error(f"Failed to download {image_url}: {resp.status_code}")
                return False
            image_bytes = resp.content

        # Analyze with Vision
        tags = await analyze_image(image_bytes)
        if not tags:
            logger.error(f"Failed to analyze {code}")
            return False

        # Generate CLIP image embedding (or fallback to text)
        embedding = await generate_embedding_from_image(image_url=image_url)
        if not embedding:
            # Fallback to text embedding from tags
            text = _tags_to_text(tags, code=code, model=model, size=size)
            embedding = await _gemini_text_embedding(text)

        # Store in database
        pool = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags, ensure_ascii=False)

        await pool.execute(
            """INSERT INTO product_embeddings (inventory_id, code, tags, embedding, created_at)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (inventory_id) DO UPDATE SET tags = $3, embedding = $4, created_at = $5""",
            inventory_id, code, tags_json, str(embedding), now,
        )
        return True
    except Exception as e:
        logger.error(f"Indexing failed for {code}: {e}")
        return False


async def index_all_products() -> dict:
    """Index all products that have images but no embeddings yet."""
    pool = await get_db()

    # Get products with images that haven't been indexed
    rows = await pool.fetch("""
        SELECT i.id, i.code, i.model, i.size, i.image_url
        FROM inventory i
        LEFT JOIN product_embeddings pe ON pe.inventory_id = i.id
        WHERE i.image_url != '' AND i.image_url IS NOT NULL AND i.stock > 0 AND pe.id IS NULL
        ORDER BY i.code
    """)

    if not rows:
        return {"indexed": 0, "message": "All products already indexed"}

    success = 0
    failed = 0
    for row in rows:
        print(f"[INDEX] Processing {row['code']}...", flush=True)
        ok = await index_product(row["id"], row["code"], row["model"], row["size"], row["image_url"])
        if ok:
            success += 1
            print(f"[INDEX] ✅ {row['code']}", flush=True)
        else:
            failed += 1
            print(f"[INDEX] ❌ {row['code']}", flush=True)

    return {"indexed": success, "failed": failed, "total": len(rows)}


async def analyze_and_match(image_bytes: bytes, size: str = "") -> dict:
    """Analyze a user's photo and find the closest matching product.

    Returns:
        {"matched": True, "code": "TP15", "score": 0.92, "product": {...}} or
        {"matched": False, "message": "..."}
    """
    # Step 1: Analyze user's photo
    tags = await analyze_image(image_bytes)
    if not tags:
        return {"matched": False, "message": "ფოტოს ანალიზი ვერ მოხერხდა"}

    # Step 2: Generate CLIP embedding for user's photo
    user_embedding = await generate_embedding_from_image(image_bytes=image_bytes)
    if not user_embedding:
        # Fallback to text embedding
        text = _tags_to_text(tags)
        if size:
            text += f" size:{size}"
        user_embedding = await _gemini_text_embedding(text)

    # Step 3: Find closest match in database using pgvector cosine similarity
    pool = await get_db()

    query = """
        SELECT pe.code, pe.tags, pe.inventory_id,
               i.model, i.size, i.price, i.image_url, i.image_url_back, i.stock,
               1 - (pe.embedding <=> $1::vector) as similarity
        FROM product_embeddings pe
        JOIN inventory i ON i.id = pe.inventory_id
        WHERE i.stock > 0
    """
    params = [str(user_embedding)]
    idx = 2

    if size:
        query += f" AND i.size ILIKE ${idx}"
        params.append(f"%{size}%")
        idx += 1

    query += " ORDER BY similarity DESC LIMIT 3"

    rows = await pool.fetch(query, *params)

    if not rows:
        return {"matched": False, "message": "მარაგში ვერ მოიძებნა"}

    best = rows[0]
    score = float(best["similarity"])

    # Threshold: 0.75 = good match
    if score < 0.75:
        return {
            "matched": False,
            "message": "ზუსტი შესაბამისი ვერ მოიძებნა",
            "closest_code": best["code"],
            "score": round(score, 2),
        }

    return {
        "matched": True,
        "code": best["code"],
        "score": round(score, 2),
        "product": {
            "code": best["code"],
            "model": best["model"],
            "size": best["size"],
            "price": best["price"],
            "image_url": best["image_url"],
            "image_url_back": best.get("image_url_back", ""),
        },
        "alternatives": [
            {"code": r["code"], "score": round(float(r["similarity"]), 2)}
            for r in rows[1:]
            if float(r["similarity"]) > 0.70
        ],
    }
