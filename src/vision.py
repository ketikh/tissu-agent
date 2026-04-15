"""Tissu Shop — Receipt Detection via Cloud Vision OCR.

Uses Google Cloud Vision TEXT_DETECTION to extract text from photos,
then checks for Georgian banking keywords to identify payment receipts.
Much faster, cheaper and more accurate than LLM-based detection.
"""
from __future__ import annotations

import logging
import os

import httpx

from src.db import get_db

# Ensure credentials path is set for Cloud Vision
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _default_cred = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "secrets", "gcloud-key.json"
    )
    if os.path.exists(_default_cred):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _default_cred
    elif os.getenv("GOOGLE_CREDENTIALS_JSON"):
        import tempfile
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _tmp.write(os.getenv("GOOGLE_CREDENTIALS_JSON"))
        _tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

from google.cloud import vision  # noqa: E402

logger = logging.getLogger(__name__)

_vision_client: vision.ImageAnnotatorClient | None = None


def _get_vision_client() -> vision.ImageAnnotatorClient:
    global _vision_client
    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


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


# Keywords that appear on Georgian bank payment receipts
RECEIPT_KEYWORDS = (
    # Georgian — payment/transfer actions
    "ჩარიცხულია", "ჩარიცხვა", "გადარიცხვა", "გადაირიცხა",
    "თანხა", "წარმატებით", "შესრულებულია", "გადახდა",
    # Georgian — bank names
    "თიბისი", "TBC", "tbc",
    "საქართველოს ბანკი", "BOG", "bog",
    "ლიბერთი", "Liberty", "liberty",
    # Georgian — common receipt fields
    "ანგარიში", "ბენეფიციარი", "მიმღები", "გამგზავნი",
    "თარიღი", "ოპერაცია", "ტრანზაქცია",
    # Symbols / currency
    "GEL", "₾", "USD", "$", "EUR",
    # English banking terms
    "Transfer", "Payment", "Amount", "Receipt", "Transaction",
    "Beneficiary", "Sender",
    # Status indicators
    "SUCCESS", "Completed", "Successful", "✓", "✔",
)


async def is_payment_receipt(image_bytes: bytes, conversation_id: str) -> bool:
    """Check if image is a payment receipt using Cloud Vision OCR.

    Fast, cheap, deterministic — no LLM needed.
    Uses conversation context as fallback if OCR fails.
    """
    # Check conversation context first — faster path if we're expecting a receipt
    expecting = False
    try:
        pool = await get_db()
        rows = await pool.fetch(
            "SELECT content FROM messages WHERE conversation_id = $1 AND role = 'assistant' ORDER BY created_at DESC LIMIT 3",
            conversation_id,
        )
        ctx_keywords = ("სქრინ", "ჩარიცხ", "გადარიცხ", "გადასახდელი")
        for row in rows:
            text = row["content"] or ""
            if any(kw in text for kw in ctx_keywords):
                expecting = True
                break
    except Exception as e:
        logger.warning(f"Context check failed: {e}")

    # Run Cloud Vision OCR on the image
    try:
        client = _get_vision_client()
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)

        if response.error.message:
            logger.error(f"Vision OCR error: {response.error.message}")
            return expecting  # fall back to context

        # Get the full text (first element is the full block)
        annotations = response.text_annotations
        if not annotations:
            # No text detected — definitely not a receipt
            print(f"[RECEIPT] No text detected — NOT a receipt", flush=True)
            return False

        full_text = annotations[0].description or ""
        text_lower = full_text.lower()

        # Count matching keywords
        matches = [kw for kw in RECEIPT_KEYWORDS if kw.lower() in text_lower]

        print(f"[RECEIPT] OCR detected {len(annotations)} text blocks, "
              f"{len(matches)} banking keywords: {matches[:5]}", flush=True)

        # Need at least 2 banking keywords to be confident it's a receipt
        if len(matches) >= 2:
            return True
        # Single strong match — bank name + nothing else — still counts
        bank_names = ("თიბისი", "TBC", "საქართველოს ბანკი", "BOG", "ლიბერთი")
        if any(bn.lower() in text_lower for bn in bank_names) and expecting:
            return True
        return False

    except Exception as e:
        logger.error(f"Receipt detection failed: {e}")
        return expecting  # fall back to context
