"""Tissu Shop — Receipt Detection Only.

Uses Gemini to detect if customer photo is a payment receipt.
Product photo matching is handled manually by owner via WhatsApp.
"""
from __future__ import annotations

import logging
import os

import httpx
from google import genai
from google.genai import types

from src.db import get_db

logger = logging.getLogger(__name__)


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))


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


async def is_payment_receipt(image_bytes: bytes, conversation_id: str) -> bool:
    """Check if image is a payment receipt (vs product photo).

    Uses conversation context: if we recently asked for payment screenshot,
    it's more likely a receipt.
    """
    client = _get_gemini_client()

    # Check conversation context
    expecting = False
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
                expecting = True
                break
    finally:
        await db.close()

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
        return expecting  # If we were expecting receipt, assume it is one
