"""Gemini Vision AI — image analysis for Tissu Shop.

Analyzes customer photos:
- Payment receipt detection (bank transfers, screenshots)
- Product comparison with inventory (visual similarity matching)
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

# In-memory cache for downloaded product images
_image_cache: dict[str, bytes] = {}


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
    # Check cache first
    if url in _image_cache:
        return _image_cache[url]

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                _image_cache[url] = resp.content
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

    # Product photo — compare visually with inventory
    similar_codes = await _find_similar_products(client, image_bytes)

    return ImageAnalysisResult(
        image_type="product",
        description=analysis_text,
        similar_codes=similar_codes,
        raw_text=analysis_text,
    )


async def _find_similar_products(client: genai.Client, image_bytes: bytes) -> list[str]:
    """Compare customer photo VISUALLY with inventory images."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, model, size, price, image_url FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        products = [dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()

    if not products:
        return []

    # Step 1: Analyze customer photo — extract visual features
    feature_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text=(
                'ეს ლეპტოპის ქეისი/ჩანთაა. დეტალურად აღწერე:\n'
                '1. ძირითადი ფერ(ებ)ი (მაგ: ლურჯი, წითელი, მწვანე, ყავისფერი, შავი, ვარდისფერი, ნარინჯისფერი...)\n'
                '2. ნახატი/პატერნი (ყვავილები, ზოლები, გეომეტრიული, ერთფეროვანი, აბსტრაქტული...)\n'
                '3. ტიპი: ფხრიწიანი (zipper ჩანს) თუ თასმიანი (ღილი/თასმა ჩანს) თუ გაურკვეველი?\n'
                'JSON: {"colors": ["ფერი1", "ფერი2"], "pattern": "აღწერა", "type": "ფხრიწიანი" ან "თასმიანი" ან "გაურკვეველი"}'
            )),
        ])],
    )

    feature_text = feature_response.text.strip() if feature_response.text else ""
    detected_type = ""
    try:
        if "{" in feature_text:
            features = json.loads(feature_text[feature_text.index("{"):feature_text.rindex("}") + 1])
            detected_type = features.get("type", "")
    except Exception:
        pass

    # Step 2: Filter by type if detected
    candidates = products
    if detected_type in ("ფხრიწიანი", "თასმიანი"):
        type_filtered = [p for p in products if detected_type in p.get("model", "")]
        if type_filtered:
            candidates = type_filtered

    # Step 3: Download candidate images and compare visually
    # Process in batches of 6 to stay within limits
    batch_size = 6
    all_matches: list[dict] = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_parts: list[types.Part] = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text="ეს კლიენტის ფოტოა. ახლა ჩვენი პროდუქტების ფოტოები მოდის:\n"),
        ]

        loaded_codes: list[str] = []
        for product in batch:
            img_url = product.get("image_url", "")
            if not img_url:
                continue
            product_img = await download_image(img_url)
            if not product_img:
                continue
            code = product["code"]
            loaded_codes.append(code)
            batch_parts.append(types.Part.from_bytes(data=product_img, mime_type="image/jpeg"))
            batch_parts.append(types.Part(text=f"ეს არის {code} ({product['model']}, {product['size']}).\n"))

        if not loaded_codes:
            continue

        codes_str = ", ".join(loaded_codes)
        batch_parts.append(types.Part(text=(
            f'\nშეადარე კლიენტის ფოტო (პირველი) ჩვენს პროდუქტებს ({codes_str}).\n'
            'ყურადღება მიაქციე: ფერს, ნახატს/პატერნს, მატერიას, სტილს.\n'
            'მხოლოდ ნამდვილად მსგავსი პროდუქტები დააბრუნე (ფერი და სტილი უნდა ემთხვეოდეს).\n'
            'თუ არცერთი არ ჰგავს — ცარიელი სია დააბრუნე.\n'
            'JSON: {"matches": [{"code": "XX1", "similarity": 0.0-1.0, "reason": "მოკლედ რატომ"}]}'
        )))

        try:
            compare = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=batch_parts)],
            )
            compare_text = compare.text.strip() if compare.text else ""

            if "{" in compare_text:
                parsed = json.loads(compare_text[compare_text.index("{"):compare_text.rindex("}") + 1])
                for match in parsed.get("matches", []):
                    if match.get("similarity", 0) >= 0.5:
                        all_matches.append(match)
        except Exception as e:
            logger.error(f"Vision comparison failed for batch: {e}")
            continue

    # Sort by similarity, return top 3
    all_matches.sort(key=lambda m: m.get("similarity", 0), reverse=True)
    return [m["code"] for m in all_matches[:3]]
