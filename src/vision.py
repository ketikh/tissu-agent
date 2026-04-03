"""Gemini Vision AI — image analysis for Tissu Shop.

Analyzes customer photos:
- Payment receipt detection
- Product comparison: downloads all product images at startup,
  then sends customer photo + product photos to Gemini for direct visual comparison.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import types

from src.db import get_db

logger = logging.getLogger(__name__)

# Cached product images: code -> (image_bytes, model, size)
_product_images: dict[str, tuple[bytes, str, str]] = {}
_images_loaded = False


def _get_vision_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


@dataclass
class ImageAnalysisResult:
    image_type: str  # "payment_receipt" | "product" | "unknown"
    description: str
    similar_codes: list[str]
    raw_text: str


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


async def preload_product_images() -> None:
    """Download all product images at startup and cache in memory."""
    global _product_images, _images_loaded

    if _images_loaded:
        return

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, model, size, image_url FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        products = [dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()

    count = 0
    for p in products:
        img = await download_image(p["image_url"])
        if img:
            _product_images[p["code"]] = (img, p["model"], p["size"])
            count += 1

    _images_loaded = True
    logger.info(f"Preloaded {count}/{len(products)} product images")


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


async def analyze_image(image_bytes: bytes, conversation_id: str) -> ImageAnalysisResult:
    """Analyze customer image — receipt or product."""
    client = _get_vision_client()
    expecting_receipt = await is_expecting_receipt(conversation_id)

    ctx_hint = (
        "კლიენტმა გადახდის სქრინი/ქვითარი უნდა გამოეგზავნა."
        if expecting_receipt
        else "კლიენტი ჩანთას ეძებს."
    )

    analysis = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text=(
                f'კონტექსტი: {ctx_hint}\n'
                'ეს ფოტო რა არის? თუ ბანკის ტრანზაქცია/გადარიცხვა/check — payment_receipt. '
                'თუ ჩანთა/ქეისი — product. '
                'JSON: {"type": "payment_receipt" ან "product", "description": "მოკლე"}'
            )),
        ])],
    )

    analysis_text = analysis.text.strip() if analysis.text else ""

    if "payment_receipt" in analysis_text:
        return ImageAnalysisResult("payment_receipt", analysis_text, [], analysis_text)

    similar_codes = await _find_similar_products(client, image_bytes)
    return ImageAnalysisResult("product", analysis_text, similar_codes, analysis_text)


async def _find_similar_products(client: genai.Client, image_bytes: bytes) -> list[str]:
    """Send customer photo + all product photos to Gemini for direct visual comparison."""

    # Ensure images are loaded
    if not _images_loaded:
        await preload_product_images()

    if not _product_images:
        return []

    # Split into batches of 10 products per API call
    codes = list(_product_images.keys())
    batch_size = 10
    all_matches: list[str] = []

    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i:i + batch_size]

        parts: list[types.Part] = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text="⬆️ კლიენტის ფოტო. ქვემოთ ჩვენი პროდუქტები:\n\n"),
        ]

        for code in batch_codes:
            img_data, model, size = _product_images[code]
            parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
            parts.append(types.Part(text=f"⬆️ {code}\n"))

        parts.append(types.Part(text=(
            '\nრომელი პროდუქტი ჰგავს კლიენტის ფოტოს ფერით და ნახატით?\n'
            'მნიშვნელოვანია: ფერი უნდა ემთხვეოდეს! ნარინჯისფერი ≠ წითელი, ლურჯი ≠ იისფერი.\n'
            'თუ არცერთი ფერით არ ჰგავს — ცარიელი სია.\n'
            'JSON: {"matches": ["CODE1", "CODE2"]} ან {"matches": []}'
        )))

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=parts)],
            )
            resp_text = resp.text.strip() if resp.text else ""

            if "{" in resp_text:
                parsed = json.loads(resp_text[resp_text.index("{"):resp_text.rindex("}") + 1])
                all_matches.extend(parsed.get("matches", []))
        except Exception as e:
            logger.error(f"Vision comparison batch failed: {e}")
            continue

    # Deduplicate and limit to 3
    seen = set()
    unique: list[str] = []
    for code in all_matches:
        if code not in seen and code in _product_images:
            seen.add(code)
            unique.append(code)
    return unique[:3]
