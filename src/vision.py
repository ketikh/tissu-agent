"""Tissu Shop — Image Analysis.

Two functions:
1. Receipt detection (Gemini Vision)
2. Product color matching (Pillow color histogram — no AI needed)
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass

import httpx
from PIL import Image
from google import genai
from google.genai import types

from src.db import get_db

logger = logging.getLogger(__name__)

# Cached product color profiles: code -> list of (r, g, b, percentage)
_product_colors: dict[str, list[tuple[int, int, int, float]]] = {}
_colors_loaded = False


def _get_vision_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


@dataclass
class ImageAnalysisResult:
    image_type: str  # "payment_receipt" | "product"
    description: str
    similar_codes: list[str]
    raw_text: str


# ── Color Extraction (Pillow) ──────────────────────────────────

def extract_color_profile(image_bytes: bytes) -> list[tuple[int, int, int, float]]:
    """Extract dominant colors from image using Pillow quantization.

    Returns list of (R, G, B, percentage) sorted by percentage descending.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to speed up processing
    img = img.resize((80, 80), Image.Resampling.LANCZOS)
    # Quantize to 8 dominant colors
    quantized = img.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    if not palette:
        return []

    pixel_counts = Counter(quantized.getdata())
    total_pixels = sum(pixel_counts.values())

    colors = []
    for color_idx, count in pixel_counts.most_common(8):
        r = palette[color_idx * 3]
        g = palette[color_idx * 3 + 1]
        b = palette[color_idx * 3 + 2]
        pct = count / total_pixels
        colors.append((r, g, b, pct))

    return colors


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Weighted Euclidean distance in RGB space (human-perception weighted)."""
    # Human eyes are more sensitive to green, less to blue
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return math.sqrt(2 * dr * dr + 4 * dg * dg + 3 * db * db)


def compare_color_profiles(
    profile1: list[tuple[int, int, int, float]],
    profile2: list[tuple[int, int, int, float]],
) -> float:
    """Compare two color profiles, return similarity 0.0–1.0.

    For each dominant color in profile1, find the closest color in profile2
    and weight by the color's percentage.
    """
    if not profile1 or not profile2:
        return 0.0

    max_distance = 765.0  # max weighted RGB distance
    score = 0.0

    for r1, g1, b1, pct1 in profile1:
        min_dist = max_distance
        for r2, g2, b2, _pct2 in profile2:
            d = _color_distance((r1, g1, b1), (r2, g2, b2))
            if d < min_dist:
                min_dist = d
        similarity = 1.0 - (min_dist / max_distance)
        score += similarity * pct1

    return score


# ── Image Download ─────────────────────────────────────────────

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


# ── Startup: Pre-compute product colors ────────────────────────

async def preload_product_images() -> None:
    """Download all product images and extract color profiles at startup."""
    global _product_colors, _colors_loaded

    if _colors_loaded:
        return

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, image_url FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        products = [dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()

    count = 0
    for p in products:
        img = await download_image(p["image_url"])
        if img:
            try:
                profile = extract_color_profile(img)
                if profile:
                    _product_colors[p["code"]] = profile
                    count += 1
            except Exception as e:
                logger.error(f"Color extraction failed for {p['code']}: {e}")

    _colors_loaded = True
    logger.info(f"Color profiles: {count}/{len(products)} products loaded")


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
    client = _get_vision_client()
    expecting = await is_expecting_receipt(conversation_id)

    ctx = "კლიენტმა გადახდის სქრინი უნდა გამოეგზავნა." if expecting else "კლიენტი ჩანთას ეძებს."

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part(text=(
                    f'კონტექსტი: {ctx}\n'
                    'ეს ფოტო payment_receipt (ბანკის ტრანზაქცია/გადარიცხვა/check) თუ product (ჩანთა/ქეისი)?\n'
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

    # Step 1: Check if it's a payment receipt (Gemini)
    if await _is_payment_receipt(image_bytes, conversation_id):
        return ImageAnalysisResult("payment_receipt", "გადახდის ქვითარი", [], "")

    # Step 2: Color comparison with inventory (Pillow — no AI)
    if not _colors_loaded:
        await preload_product_images()

    customer_profile = extract_color_profile(image_bytes)
    if not customer_profile or not _product_colors:
        return ImageAnalysisResult("product", "ფერების ამოცნობა ვერ მოხერხდა", [], "")

    # Compare with every product
    scores: list[tuple[str, float]] = []
    for code, product_profile in _product_colors.items():
        similarity = compare_color_profiles(customer_profile, product_profile)
        scores.append((code, similarity))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    # Top matches with minimum threshold 0.75
    matches = [code for code, score in scores if score >= 0.75][:3]

    # If no strong matches, take top 2 if above 0.65
    if not matches:
        matches = [code for code, score in scores if score >= 0.65][:2]

    desc = f"ფერის შედარება: ტოპ={scores[0][1]:.2f}" if scores else ""
    return ImageAnalysisResult("product", desc, matches, str(scores[:5]))
