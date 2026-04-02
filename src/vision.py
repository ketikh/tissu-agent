"""Gemini Vision AI — image analysis for Tissu Shop.

Analyzes customer photos:
- Payment receipt detection (bank transfers, screenshots)
- Product comparison with inventory (find similar items)
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


def _get_vision_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


@dataclass
class ImageAnalysisResult:
    image_type: str  # "payment_receipt" | "product" | "unknown"
    description: str
    similar_codes: list[str]  # product codes if type == "product"
    raw_text: str  # original analysis text


async def download_image(url: str) -> bytes | None:
    """Download image from URL, return bytes or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.content
    except Exception as e:
        logger.error(f"Image download failed: {e}")
    return None


async def is_expecting_receipt(conversation_id: str) -> bool:
    """Check recent bot messages to see if we asked for a payment screenshot."""
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
    """Analyze customer image — detect receipt vs product photo."""
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
                'ეს ფოტო რა არის? თუ ბანკის ტრანზაქცია, გადარიცხვა, check icon, თანხა — payment_receipt. '
                'თუ ჩანთა/ქეისი — product. '
                'JSON: {"type": "payment_receipt" ან "product", "description": "მოკლე აღწერა"}'
            )),
        ])],
    )

    analysis_text = analysis.text.strip() if analysis.text else ""
    is_receipt = "payment_receipt" in analysis_text

    if is_receipt:
        return ImageAnalysisResult(
            image_type="payment_receipt",
            description=analysis_text,
            similar_codes=[],
            raw_text=analysis_text,
        )

    # Product photo — compare with inventory
    similar_codes = await _find_similar_products(client, image_bytes)

    return ImageAnalysisResult(
        image_type="product",
        description=analysis_text,
        similar_codes=similar_codes,
        raw_text=analysis_text,
    )


async def _find_similar_products(client: genai.Client, image_bytes: bytes) -> list[str]:
    """Compare customer photo with inventory, return similar product codes."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, model, size, price FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        products = [dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()

    if not products:
        return []

    product_list = "\n".join(
        f"- {p['code']}: {p['model']}, {p['size']}, {p['price']}₾"
        for p in products
    )

    compare = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text=(
                f"კლიენტის ფოტოზე ჩანთა/ქეისია. ჩვენ მარაგში ეს პროდუქტები გვაქვს:\n{product_list}\n\n"
                'რომელი კოდ(ებ)ი ჰგავს ყველაზე მეტად? უპასუხე JSON: '
                '{"similar_codes": ["FD1","FP3"], "found": true} ან {"similar_codes": [], "found": false}'
            )),
        ])],
    )

    compare_text = compare.text.strip() if compare.text else ""

    try:
        if "{" in compare_text:
            parsed = json.loads(compare_text[compare_text.index("{"):compare_text.rindex("}") + 1])
            return parsed.get("similar_codes", [])[:5]
    except Exception:
        pass

    return []
