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


async def crop_to_main_bag(image_bytes: bytes) -> bytes:
    """Ask Gemini where the main laptop bag is, then crop to that region.

    Designed for customer screenshots (Instagram / Facebook) that contain
    the product bag in the middle surrounded by UI chrome — profile avatars,
    reaction icons, headers, comments, a sliver of the next post, etc. CLIP
    and the Gemini re-ranker both work far better on a tight crop of just
    the product than on the full screenshot where the bag is a small
    fraction of the pixels.

    Returns the cropped JPEG bytes, or the original bytes if anything fails
    (no crop is always safer than returning nothing).
    """
    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        W, H = img.size

        client = _get_client()
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part(text=(
                    "You are looking at a customer photo. It may be a plain product "
                    "photo OR a screenshot from Instagram / Facebook / a website that "
                    "contains a lot of UI clutter around the product (profile avatars, "
                    "reaction icons, status bar, page header, comments, a partial image "
                    "of the next post at the edge).\n\n"
                    "Find the ONE largest, most prominent LAPTOP BAG / LAPTOP SLEEVE / "
                    "CLUTCH-STYLE FABRIC BAG in the image — the main subject. Ignore "
                    "tiny avatars, UI icons, cropped photos at the edges.\n\n"
                    "Return ONLY JSON, no prose, no markdown:\n"
                    "{\"x1\": <int 0-100>, \"y1\": <int 0-100>, "
                    "\"x2\": <int 0-100>, \"y2\": <int 0-100>}\n\n"
                    "Where x1,y1 is the top-left and x2,y2 the bottom-right of the "
                    "tightest box around the main bag, as PERCENTAGES of image width "
                    "and height. If no bag is visible, return "
                    "{\"x1\": 0, \"y1\": 0, \"x2\": 100, \"y2\": 100}."
                )),
            ])],
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        box = json.loads(text)

        x1 = max(0, int(box["x1"]) * W // 100)
        y1 = max(0, int(box["y1"]) * H // 100)
        x2 = min(W, int(box["x2"]) * W // 100)
        y2 = min(H, int(box["y2"]) * H // 100)

        # Sanity checks — ignore obviously bogus boxes and return original.
        if x2 - x1 < W * 0.15 or y2 - y1 < H * 0.15:
            return image_bytes
        # If Gemini just returned the whole image, skip the crop (nothing to do).
        if x1 == 0 and y1 == 0 and x2 == W and y2 == H:
            return image_bytes

        # Add a small 5% padding so we don't clip the edge of the bag.
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)

        cropped = img.crop((x1, y1, x2, y2))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=92)
        out = buf.getvalue()
        print(
            f"[CROP] Gemini box: ({x1},{y1})-({x2},{y2}) of ({W},{H}) → "
            f"{len(out)} bytes",
            flush=True,
        )
        return out
    except Exception as e:
        print(f"[CROP] Failed, using original: {e}", flush=True)
        return image_bytes


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


async def generate_embedding_from_image(image_url: str = "", image_bytes: bytes = b"", remove_bg: bool = True) -> list[float]:
    """Generate CLIP image embedding via microservice. Uses background
    removal by default so props, flowers, and backgrounds don't contaminate
    the product embedding — customer photos and catalog photos must be
    processed the same way or scores won't be comparable."""
    if not CLIP_SERVICE_URL:
        # Fallback to Gemini text embedding if CLIP not available
        return await _gemini_text_embedding("laptop bag product")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            payload: dict = {"remove_bg": remove_bg}
            if image_url:
                payload["image_url"] = image_url
            elif image_bytes:
                import base64
                payload["image_base64"] = base64.b64encode(image_bytes).decode()
            else:
                return []
            resp = await client.post(f"{CLIP_SERVICE_URL}/embed", json=payload)

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


async def index_product(inventory_id: int, code: str, model: str, size: str, image_url: str, image_url_back: str = "") -> bool:
    """Analyze and index a product — both front and back images."""
    try:
        # Generate CLIP embedding for front image
        embedding = await generate_embedding_from_image(image_url=image_url)
        if not embedding:
            logger.error(f"Failed to embed {code} front")
            return False

        # Analyze with Vision for tags
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(image_url)
            image_bytes = resp.content if resp.status_code == 200 else b""
        tags = await analyze_image(image_bytes) if image_bytes else {}

        # Store front embedding
        pool = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags, ensure_ascii=False)
        await pool.execute(
            """INSERT INTO product_embeddings (inventory_id, code, tags, embedding, created_at)
               VALUES ($1, $2, $3, $4, $5)""",
            inventory_id, code, tags_json, str(embedding), now,
        )

        # Index back image too (as separate row with negative inventory_id)
        if image_url_back:
            back_embedding = await generate_embedding_from_image(image_url=image_url_back)
            if back_embedding:
                await pool.execute(
                    """INSERT INTO product_embeddings (inventory_id, code, tags, embedding, created_at)
                       VALUES ($1, $2, $3, $4, $5)""",
                    -inventory_id, code, tags_json, str(back_embedding), now,
                )

        return True
    except Exception as e:
        logger.error(f"Indexing failed for {code}: {e}")
        return False


async def index_all_products() -> dict:
    """Index all products that have images but no embeddings yet.
    Includes sold-out items so we can still identify them from customer
    photos and say 'sorry, this one is out of stock'."""
    pool = await get_db()

    # Get products with images that haven't been indexed
    rows = await pool.fetch("""
        SELECT i.id, i.code, i.model, i.size, i.image_url, i.image_url_back
        FROM inventory i
        LEFT JOIN product_embeddings pe ON pe.inventory_id = i.id
        WHERE i.image_url != '' AND i.image_url IS NOT NULL AND pe.id IS NULL
        ORDER BY i.code
    """)

    if not rows:
        return {"indexed": 0, "message": "All products already indexed"}

    success = 0
    failed = 0
    for row in rows:
        print(f"[INDEX] Processing {row['code']}...", flush=True)
        ok = await index_product(row["id"], row["code"], row["model"], row["size"], row["image_url"], row.get("image_url_back", ""))
        if ok:
            success += 1
            print(f"[INDEX] ✅ {row['code']}", flush=True)
        else:
            failed += 1
            print(f"[INDEX] ❌ {row['code']}", flush=True)

    return {"indexed": success, "failed": failed, "total": len(rows)}


async def _gemini_visual_compare(user_image: bytes, candidate_urls: list[str]) -> str | None:
    """Use Gemini Vision to pick the best match from CLIP candidates."""
    if not candidate_urls:
        return None
    client = _get_client()
    try:
        # Download candidate images
        parts = [types.Part.from_bytes(data=user_image, mime_type="image/jpeg")]
        parts.append(types.Part(text="ეს არის კლიენტის ფოტო (პირველი სურათი). ქვემოთ არის კანდიდატი პროდუქტები:"))

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as http:
            for i, url in enumerate(candidate_urls):
                try:
                    resp = await http.get(url)
                    if resp.status_code == 200:
                        parts.append(types.Part.from_bytes(data=resp.content, mime_type="image/jpeg"))
                        parts.append(types.Part(text=f"კანდიდატი {i+1}"))
                except Exception:
                    pass

        parts.append(types.Part(text=(
            "TASK: Match the customer's laptop-bag photo to exactly one of the candidate reference photos. "
            "Some products have multiple reference photos (front + back) — treat each numbered reference as a "
            "separate candidate. The bag must be the SAME PRODUCT — same fabric, same pattern, same colors. "
            "A similar shape with different colors is NOT a match.\n\n"
            "FIRST — FIND THE MAIN SUBJECT IN THE CUSTOMER PHOTO:\n"
            "The customer photo is often a SCREENSHOT from Instagram / Facebook / a website. "
            "Such screenshots contain ONE large central bag (the main subject) plus lots of clutter: "
            "small round profile avatars, tiny reaction icons (hearts, likes), status bars, page headers, "
            "comments, usernames, and sometimes a sliver of the NEXT post's photo starting at the top or bottom edge. "
            "IGNORE ALL OF THAT. Look ONLY at the single largest, most prominent bag in the frame — the one "
            "that occupies the center and takes up the most pixels. Mentally crop away everything else "
            "(small circles, UI chrome, partial images at the edges, text) before you compare. "
            "If multiple bags appear, pick the BIGGEST one — that is the product the customer is asking about.\n\n"
            "COMPARE THE MAIN BAG ON ALL FOUR DIMENSIONS:\n"
            "1) DOMINANT COLORS — exact hue, not just 'blue'. Orange ≠ yellow. Navy ≠ royal blue. "
            "Burgundy ≠ red. Mint ≠ teal. If the customer photo is orange and the candidate is yellow, it's NOT a match.\n"
            "2) PATTERN TYPE — solid / floral / geometric / striped / plaid / tropical / abstract / animal. "
            "Different pattern type = NOT a match.\n"
            "3) PATTERN SCALE & COLOR ARRANGEMENT — same motifs arranged the same way, same accent colors.\n"
            "4) FABRIC TEXTURE — canvas / denim / quilted / smooth. Different texture = NOT a match.\n\n"
            "IGNORE: background, lighting, angle, hands, shadows, props, flowers, tables, "
            "and everything described above as 'screenshot clutter'. "
            "Judge only the surface of the single main bag. If the customer photo shows the bag at an odd angle, "
            "mentally rotate/unfold it before comparing.\n\n"
            "BE STRICT. When in doubt, answer 0. It is better to return 'no match' than to return "
            "a wrong product of a similar shape but different color.\n\n"
            "Answer with ONLY the candidate number (e.g. 1, 2, 3, ...) for the best match, or 0 if none match."
        )))

        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[types.Content(role="user", parts=parts)],
        )
        answer = (resp.text or "").strip()
        # Extract first contiguous number (supports 1, 2, ..., 10+ candidates)
        import re as _re
        m = _re.search(r"\d+", answer)
        if m:
            return int(m.group(0))
        return None
    except Exception as e:
        logger.error(f"Visual compare failed: {e}")
        return None


async def analyze_and_match(image_bytes: bytes, size: str = "") -> dict:
    """Analyze a user's photo and find the closest matching product.

    Uses CLIP for top-3 candidates, then Gemini Vision to pick the best match.

    Returns:
        {"matched": True, "code": "TP15", "score": 0.92, "product": {...}} or
        {"matched": False, "message": "..."}
    """
    # Step 1: Generate CLIP embedding for user's photo
    user_embedding = await generate_embedding_from_image(image_bytes=image_bytes)
    if not user_embedding:
        return {"matched": False, "message": "CLIP embedding ვერ მოხერხდა"}

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
    clip_score = float(best["similarity"])

    # If CLIP score is too low, no match
    if clip_score < 0.70:
        return {
            "matched": False,
            "message": "ზუსტი შესაბამისი ვერ მოიძებნა",
            "closest_code": best["code"],
            "score": round(clip_score, 2),
        }

    # Step 3: Gemini Vision picks the best from top-3 CLIP candidates
    candidate_urls = [r["image_url"] for r in rows if r.get("image_url")]
    gemini_pick = await _gemini_visual_compare(image_bytes, candidate_urls)

    if gemini_pick and 1 <= gemini_pick <= len(rows):
        # Gemini chose a specific candidate
        chosen = rows[gemini_pick - 1]
        print(f"[MATCH] Gemini picked #{gemini_pick}: {chosen['code']}", flush=True)
    elif gemini_pick == 0:
        # Gemini says none match — but if CLIP is very confident, use it
        if clip_score >= 0.85:
            chosen = best
            print(f"[MATCH] Gemini said 0 but CLIP confident ({clip_score}): {best['code']}", flush=True)
        else:
            return {"matched": False, "message": "ვიზუალური შედარებით ვერ მოიძებნა"}
    else:
        # Gemini failed — use CLIP's best
        chosen = best
        print(f"[MATCH] Gemini failed, using CLIP best: {best['code']}", flush=True)

    return {
        "matched": True,
        "code": chosen["code"],
        "score": round(float(chosen["similarity"]), 2),
        "product": {
            "code": chosen["code"],
            "model": chosen["model"],
            "size": chosen["size"],
            "price": chosen["price"],
            "image_url": chosen["image_url"],
            "image_url_back": chosen.get("image_url_back", ""),
        },
        "alternatives": [
            {"code": r["code"], "score": round(float(r["similarity"]), 2)}
            for r in rows
            if r["code"] != chosen["code"] and float(r["similarity"]) > 0.70
        ],
    }
