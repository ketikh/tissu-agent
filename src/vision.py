"""Gemini Vision AI — image analysis for Tissu Shop.

Analyzes customer photos:
- Payment receipt detection (bank transfers, screenshots)
- Product comparison with inventory (color + pattern matching)

Product images are analyzed ONCE at startup and descriptions cached in DB.
Customer photo comparison uses cached descriptions for fast text matching,
then visual confirmation only for top candidates.
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

# Cached product descriptions (populated at startup)
_product_descriptions: dict[str, dict] = {}  # code -> {colors, pattern, description}


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


async def preload_product_descriptions() -> None:
    """Analyze all product images with Gemini and cache color/pattern descriptions.

    Runs at startup. Skips products that already have tags in DB.
    Updates DB with color/tags for future use.
    """
    global _product_descriptions

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, code, model, size, color, tags, image_url FROM inventory WHERE stock > 0 AND image_url IS NOT NULL AND image_url != ''"
        )
        products = [dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()

    if not products:
        return

    client = _get_vision_client()
    updated_count = 0

    for product in products:
        code = product["code"]
        existing_tags = (product.get("tags") or "").strip()

        # If tags already populated, use cached data
        if existing_tags:
            try:
                desc = json.loads(existing_tags)
                _product_descriptions[code] = desc
                continue
            except (json.JSONDecodeError, TypeError):
                pass

        # Download and analyze image
        img_bytes = await download_image(product["image_url"])
        if not img_bytes:
            continue

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part(text=(
                        'ეს ლეპტოპის ქეისია. აღწერე მოკლედ:\n'
                        '1. ძირითადი ფერ(ებ)ი ქართულად (მაგ: ნარინჯისფერი, ლურჯი, ვარდისფერი, შავი, თეთრი, ყავისფერი, მწვანე, წითელი, იისფერი, ყვითელი, ტურქუაზი, ბეჟი)\n'
                        '2. ნახატის ტიპი (ყვავილები, ზოლები, კლეტკა, გეომეტრიული, აბსტრაქტული, ერთფეროვანი, ფოთლები, ცხოველები)\n'
                        '3. ფონის ფერი (მუქი/ღია/საშუალო)\n'
                        'JSON: {"colors": ["ფერი1", "ფერი2"], "pattern": "ნახატის ტიპი", "background": "მუქი/ღია", "description": "1 წინადადება"}'
                    )),
                ])],
            )
            resp_text = response.text.strip() if response.text else ""

            if "{" in resp_text:
                desc = json.loads(resp_text[resp_text.index("{"):resp_text.rindex("}") + 1])
                _product_descriptions[code] = desc

                # Save to DB for future startups
                db = await get_db()
                try:
                    color_str = ", ".join(desc.get("colors", []))
                    tags_json = json.dumps(desc, ensure_ascii=False)
                    await db.execute(
                        "UPDATE inventory SET color = ?, tags = ? WHERE id = ?",
                        (color_str, tags_json, product["id"]),
                    )
                    await db.commit()
                    updated_count += 1
                finally:
                    await db.close()

        except Exception as e:
            logger.error(f"Failed to analyze {code}: {e}")
            continue

    logger.info(f"Product descriptions: {len(_product_descriptions)} cached, {updated_count} newly analyzed")


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

    # Product photo — find similar by color/pattern matching
    similar_codes = await _find_similar_products(client, image_bytes)

    return ImageAnalysisResult(
        image_type="product",
        description=analysis_text,
        similar_codes=similar_codes,
        raw_text=analysis_text,
    )


async def _find_similar_products(client: genai.Client, image_bytes: bytes) -> list[str]:
    """Find similar products using cached color descriptions + visual confirmation."""

    # Step 1: Extract features from customer photo
    feature_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(text=(
                'ეს ლეპტოპის ქეისია. აღწერე:\n'
                '1. ძირითადი ფერ(ებ)ი ქართულად\n'
                '2. ნახატის ტიპი\n'
                '3. ფონი მუქი/ღია\n'
                '4. ტიპი: ფხრიწიანი (zipper ჩანს) / თასმიანი (ღილი/თასმა) / გაურკვეველი\n'
                'JSON: {"colors": ["ფერი1"], "pattern": "ტიპი", "background": "მუქი/ღია", "type": "ფხრიწიანი/თასმიანი/გაურკვეველი"}'
            )),
        ])],
    )

    customer_features = {}
    feature_text = feature_response.text.strip() if feature_response.text else ""
    try:
        if "{" in feature_text:
            customer_features = json.loads(feature_text[feature_text.index("{"):feature_text.rindex("}") + 1])
    except Exception:
        pass

    customer_colors = set(c.lower().strip() for c in customer_features.get("colors", []))
    customer_pattern = customer_features.get("pattern", "").lower()
    customer_bg = customer_features.get("background", "").lower()
    customer_type = customer_features.get("type", "")

    # Step 2: Score each product by text-matching cached descriptions
    if not _product_descriptions:
        # Fallback: if descriptions not loaded, try DB
        await _load_descriptions_from_db()

    scored: list[tuple[str, float]] = []

    for code, desc in _product_descriptions.items():
        score = 0.0
        prod_colors = set(c.lower().strip() for c in desc.get("colors", []))
        prod_pattern = desc.get("pattern", "").lower()
        prod_bg = desc.get("background", "").lower()

        # Color match (most important — 60% weight)
        if customer_colors and prod_colors:
            overlap = customer_colors & prod_colors
            if overlap:
                score += 0.6 * (len(overlap) / max(len(customer_colors), len(prod_colors)))

        # Pattern match (25% weight)
        if customer_pattern and prod_pattern:
            if customer_pattern in prod_pattern or prod_pattern in customer_pattern:
                score += 0.25
            elif any(w in prod_pattern for w in customer_pattern.split()):
                score += 0.12

        # Background match (15% weight)
        if customer_bg and prod_bg and customer_bg == prod_bg:
            score += 0.15

        # Type filter: penalize wrong type
        if customer_type in ("ფხრიწიანი", "თასმიანი"):
            # Check if code matches type (FP/FD = ფხრიწიანი, TP/TD = თასმიანი)
            is_zipper = code.startswith("F")
            if (customer_type == "ფხრიწიანი" and not is_zipper) or \
               (customer_type == "თასმიანი" and is_zipper):
                score *= 0.3  # Heavy penalty for wrong type

        if score > 0.2:
            scored.append((code, score))

    # Sort by score, take top 5 candidates
    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [code for code, _ in scored[:5]]

    if not top_candidates:
        return []

    # Step 3: Visual confirmation — send customer photo + top candidate images to Gemini
    confirmation_parts: list[types.Part] = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        types.Part(text="ეს კლიენტის ფოტოა. ქვემოთ ჩვენი კანდიდატებია:\n"),
    ]

    loaded_codes: list[str] = []
    db = await get_db()
    try:
        for code in top_candidates:
            cursor = await db.execute("SELECT image_url FROM inventory WHERE code = ? AND stock > 0", (code,))
            row = await cursor.fetchone()
            if not row or not row["image_url"]:
                continue
            img = await download_image(row["image_url"])
            if not img:
                continue
            loaded_codes.append(code)
            confirmation_parts.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
            confirmation_parts.append(types.Part(text=f"ეს არის {code}.\n"))
    finally:
        await db.close()

    if not loaded_codes:
        return top_candidates[:3]

    codes_str = ", ".join(loaded_codes)
    confirmation_parts.append(types.Part(text=(
        f'\nშეადარე კლიენტის ფოტო (პირველი) კანდიდატებს ({codes_str}).\n'
        'მხოლოდ ფერით და ნახატით ნამდვილად მსგავსი დატოვე.\n'
        'თუ არცერთი ფერით არ ჰგავს — ცარიელი.\n'
        'JSON: {"confirmed": ["კოდი1", "კოდი2"]}'
    )))

    try:
        confirm_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=confirmation_parts)],
        )
        confirm_text = confirm_resp.text.strip() if confirm_resp.text else ""

        if "{" in confirm_text:
            parsed = json.loads(confirm_text[confirm_text.index("{"):confirm_text.rindex("}") + 1])
            confirmed = parsed.get("confirmed", [])
            if confirmed:
                return confirmed[:3]
    except Exception as e:
        logger.error(f"Visual confirmation failed: {e}")

    # Fallback to text-matched results
    return top_candidates[:3]


async def _load_descriptions_from_db() -> None:
    """Load cached product descriptions from DB tags field."""
    global _product_descriptions

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT code, tags FROM inventory WHERE stock > 0 AND tags IS NOT NULL AND tags != ''"
        )
        for row in await cursor.fetchall():
            try:
                desc = json.loads(row["tags"])
                if isinstance(desc, dict) and "colors" in desc:
                    _product_descriptions[row["code"]] = desc
            except (json.JSONDecodeError, TypeError):
                pass
    finally:
        await db.close()
