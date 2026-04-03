"""Tissu Shop — Image Analysis.

1. Receipt detection: Gemini Vision
2. Product matching: Google Cloud Vision API (dominant color extraction + comparison)
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import types

from src.db import get_db

logger = logging.getLogger(__name__)

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

# Cached product colors: code -> list of {r, g, b, score, pixelFraction}
_product_colors: dict[str, list[dict]] = {}
_colors_loaded = False


def _get_api_key() -> str:
    """Get API key for Cloud Vision API."""
    return os.getenv("GOOGLE_VISION_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


@dataclass
class ImageAnalysisResult:
    image_type: str  # "payment_receipt" | "product"
    description: str
    similar_codes: list[str]
    raw_text: str


# ── Cloud Vision API ───────────────────────────────────────────

async def _extract_colors_from_bytes(image_bytes: bytes) -> list[dict]:
    """Extract dominant colors from image bytes via Cloud Vision API."""
    api_key = _get_api_key()
    if not api_key:
        return []

    b64 = base64.b64encode(image_bytes).decode()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{VISION_API_URL}?key={api_key}",
                json={
                    "requests": [{
                        "image": {"content": b64},
                        "features": [{"type": "IMAGE_PROPERTIES", "maxResults": 10}],
                    }],
                },
            )
        data = resp.json()
        response = data.get("responses", [{}])[0]
        colors = (
            response.get("imagePropertiesAnnotation", {})
            .get("dominantColors", {})
            .get("colors", [])
        )
        return [
            {
                "r": int(c.get("color", {}).get("red", 0)),
                "g": int(c.get("color", {}).get("green", 0)),
                "b": int(c.get("color", {}).get("blue", 0)),
                "score": c.get("score", 0),
                "pixelFraction": c.get("pixelFraction", 0),
            }
            for c in colors
        ]
    except Exception as e:
        logger.error(f"Cloud Vision API failed: {e}")
        return []


async def _extract_colors_batch(image_urls: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Extract colors for multiple product images via Cloud Vision API.

    Downloads each image and sends as base64 (works with any URL type).
    """
    api_key = _get_api_key()
    if not api_key:
        return {}

    results: dict[str, list[dict]] = {}

    # Download all images first
    downloaded: list[tuple[str, bytes]] = []
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        for code, url in image_urls:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    downloaded.append((code, resp.content))
            except Exception as e:
                logger.warning(f"Failed to download {code}: {e}")

    if not downloaded:
        return {}

    # Send to Vision API in batches of 16 (base64)
    batch_size = 16
    for i in range(0, len(downloaded), batch_size):
        batch = downloaded[i:i + batch_size]
        requests_list = [
            {
                "image": {"content": base64.b64encode(img_bytes).decode()},
                "features": [{"type": "IMAGE_PROPERTIES", "maxResults": 10}],
            }
            for _code, img_bytes in batch
        ]

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{VISION_API_URL}?key={api_key}",
                    json={"requests": requests_list},
                )
            data = resp.json()

            for idx, (code, _img) in enumerate(batch):
                response = data.get("responses", [])[idx] if idx < len(data.get("responses", [])) else {}
                if "error" in response:
                    logger.warning(f"Vision API error for {code}: {response['error'].get('message', '')[:100]}")
                    continue
                colors = (
                    response.get("imagePropertiesAnnotation", {})
                    .get("dominantColors", {})
                    .get("colors", [])
                )
                results[code] = [
                    {
                        "r": int(c.get("color", {}).get("red", 0)),
                        "g": int(c.get("color", {}).get("green", 0)),
                        "b": int(c.get("color", {}).get("blue", 0)),
                        "score": c.get("score", 0),
                        "pixelFraction": c.get("pixelFraction", 0),
                    }
                    for c in colors
                ]
        except Exception as e:
            logger.error(f"Vision API batch failed: {e}")

    return results


# ── Color Comparison ───────────────────────────────────────────

def _color_distance(c1: dict, c2: dict) -> float:
    """Weighted Euclidean distance (human-perception adjusted)."""
    dr = c1["r"] - c2["r"]
    dg = c1["g"] - c2["g"]
    db = c1["b"] - c2["b"]
    return math.sqrt(2 * dr * dr + 4 * dg * dg + 3 * db * db)


def _is_background_color(c: dict) -> bool:
    """Filter out near-black, near-white, and gray background colors."""
    r, g, b = c["r"], c["g"], c["b"]
    # Near-black
    if r < 45 and g < 45 and b < 45:
        return True
    # Near-white
    if r > 215 and g > 215 and b > 215:
        return True
    # Gray (all channels similar, low saturation)
    avg = (r + g + b) / 3
    if avg > 30 and max(abs(r - avg), abs(g - avg), abs(b - avg)) < 20:
        return True
    return False


def compare_colors(profile1: list[dict], profile2: list[dict]) -> float:
    """Compare two color profiles from Cloud Vision API. Returns 0.0–1.0.

    Focuses on BACKGROUND/FABRIC color (high pixelFraction = many pixels).
    De-weights small accent colors (flowers, patterns) that are shared
    across many products.
    """
    if not profile1 or not profile2:
        return 0.0

    # Filter out near-black and near-white (photo edges, shadows)
    filtered1 = [c for c in profile1 if not _is_background_color(c)]
    filtered2 = [c for c in profile2 if not _is_background_color(c)]

    if not filtered1:
        filtered1 = profile1
    if not filtered2:
        filtered2 = profile2

    max_dist = 765.0
    score = 0.0
    total_weight = 0.0

    for c1 in filtered1:
        px_frac = c1.get("pixelFraction", 0.01)
        # Weight = pixelFraction SQUARED — heavily favors background fabric color
        # Small accent colors (flowers: ~2-5% pixels) get very low weight
        # Background fabric (30-60% pixels) gets dominant weight
        weight = px_frac * px_frac
        if weight < 0.0001:
            continue

        min_dist = max_dist
        for c2 in filtered2:
            d = _color_distance(c1, c2)
            if d < min_dist:
                min_dist = d
        similarity = 1.0 - (min_dist / max_dist)
        score += similarity * weight
        total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0


# ── Product Color Loading (lazy) ──────────────────────────────

async def _ensure_product_colors() -> None:
    """Load product colors from Cloud Vision API (lazy, cached in memory).

    Uses Cloudinary URLs from seed_inventory.json (always correct),
    falling back to DB image_url if not found.
    """
    global _product_colors, _colors_loaded

    if _colors_loaded:
        return

    # Load Cloudinary URLs from seed_inventory.json
    import json
    from pathlib import Path
    seed_file = Path(__file__).parent.parent / "seed_inventory.json"
    cloudinary_urls: dict[str, str] = {}
    if seed_file.exists():
        for item in json.loads(seed_file.read_text()):
            url = item.get("image_url", "")
            if url.startswith("http"):
                cloudinary_urls[item.get("code", "")] = url

    # Get product codes from DB (only in-stock items)
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, image_url FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        db_products = {dict(r)["code"]: dict(r)["image_url"] for r in await cursor.fetchall()}
    finally:
        await db.close()

    # Build URL list: prefer Cloudinary, fall back to DB
    products: list[tuple[str, str]] = []
    for code in db_products:
        url = cloudinary_urls.get(code) or db_products[code]
        if url.startswith("http"):
            products.append((code, url))

    if not products:
        _colors_loaded = True
        return

    logger.info(f"Extracting colors for {len(products)} products via Cloud Vision API...")
    _product_colors = await _extract_colors_batch(products)
    _colors_loaded = True
    logger.info(f"Color profiles loaded: {len(_product_colors)} products")


# ── Receipt Detection (Gemini) ─────────────────────────────────

async def is_expecting_receipt(conversation_id: str) -> bool:
    """Check if we recently asked for a payment screenshot."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT content FROM messages WHERE conversation_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 3",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        receipt_keywords = ("სქრინ", "ჩარიცხ", "გადარიცხ", "გადასახდელი")
        for row in rows:
            text = row["content"] or ""
            if any(kw in text for kw in receipt_keywords):
                return True
        return False
    finally:
        await db.close()


async def _is_payment_receipt(image_bytes: bytes, conversation_id: str) -> bool:
    """Use Gemini to detect if image is a payment receipt."""
    client = _get_gemini_client()
    expecting = await is_expecting_receipt(conversation_id)
    ctx = "კლიენტმა გადახდის სქრინი უნდა გამოეგზავნა." if expecting else "კლიენტი ჩანთას ეძებს."

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part(text=(
                    f'კონტექსტი: {ctx}\n'
                    'ეს ფოტო payment_receipt (ბანკის ტრანზაქცია/გადარიცხვა) თუ product (ჩანთა/ქეისი)?\n'
                    'უპასუხე მხოლოდ: payment_receipt ან product'
                )),
            ])],
        )
        return "payment_receipt" in (resp.text or "").lower()
    except Exception as e:
        logger.error(f"Receipt detection failed: {e}")
        return False


# ── Main Analysis Function ─────────────────────────────────────

async def analyze_image(image_bytes: bytes, conversation_id: str) -> ImageAnalysisResult:
    """Analyze customer image — receipt or product color match."""

    # Step 1: Receipt check (Gemini)
    if await _is_payment_receipt(image_bytes, conversation_id):
        return ImageAnalysisResult("payment_receipt", "გადახდის ქვითარი", [], "")

    # Step 2: Extract customer photo colors (Cloud Vision API)
    customer_colors = await _extract_colors_from_bytes(image_bytes)
    if not customer_colors:
        logger.warning("Cloud Vision API failed for customer photo")
        return ImageAnalysisResult("product", "", [], "")

    # Step 3: Ensure product colors are loaded
    await _ensure_product_colors()
    if not _product_colors:
        logger.warning("No product color data available")
        return ImageAnalysisResult("product", "", [], "")

    # Step 4: Compare colors
    scores: list[tuple[str, float]] = []
    for code, product_profile in _product_colors.items():
        similarity = compare_colors(customer_colors, product_profile)
        scores.append((code, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)

    # Always return top 3 — customer's phone photo will look very different
    # from product photo (lighting, angle, background). Trust the ranking.
    matches = [code for code, _score in scores[:3]]

    top_info = ", ".join(f"{c}={s:.2f}" for c, s in scores[:5])
    return ImageAnalysisResult("product", f"ტოპ: {top_info}", matches, str(scores[:5]))


# ── Download helper (used by facebook.py) ──────────────────────

async def download_image(url: str) -> bytes | None:
    """Download image from URL."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.content
    except Exception as e:
        logger.error(f"Image download failed: {e}")
    return None
