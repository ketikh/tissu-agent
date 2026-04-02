"""WhatsApp webhook handler — owner responses.

Receives messages from the shop owner via WhatsApp and:
- Forwards payment confirmations/denials to customer via Messenger
- Forwards owner instructions to the agent for customer reply
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.agents.support_sales import get_support_sales_agent
from src.db import get_db
from src.engine import run_agent

logger = logging.getLogger(__name__)

router = APIRouter()

FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")

# Anti-duplicate tracking
_wa_processed_mids: dict[str, float] = {}


def _cleanup_old_mids() -> None:
    now = time.time()
    for key in list(_wa_processed_mids):
        if now - _wa_processed_mids[key] > 300:
            del _wa_processed_mids[key]


async def _get_latest_conversation_id() -> str:
    """Find the most recent customer conversation from tickets."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT conversation_id FROM tickets ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row["conversation_id"] if row else ""
    finally:
        await db.close()


def _extract_sender_id(conv_id: str) -> str:
    """Extract Facebook/Instagram sender ID from conversation_id."""
    return conv_id.replace("facebook_messenger_", "").replace("instagram_dm_", "")


async def _send_to_customer(sender_id: str, text: str) -> None:
    """Send a message to the customer via Facebook Messenger."""
    if not FB_PAGE_TOKEN or not sender_id:
        return
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            "https://graph.facebook.com/v21.0/me/messages",
            params={"access_token": FB_PAGE_TOKEN},
            json={"recipient": {"id": sender_id}, "message": {"text": text}},
        )


async def _handle_confirmation(conv_id: str) -> str:
    """Owner confirmed payment — ask customer for address."""
    # Clean up uploaded receipt photos
    uploads_dir = Path(__file__).parent.parent.parent / "static" / "uploads"
    if uploads_dir.exists():
        for f in uploads_dir.iterdir():
            if f.suffix in ('.jpg', '.jpeg', '.png'):
                f.unlink(missing_ok=True)

    agent = get_support_sales_agent()
    result = await run_agent(agent, "[მფლობელმა დაადასტურა გადახდა. მოითხოვე მისამართი და ტელეფონი.]", conv_id)
    return result["reply"].strip() or "გადახდა დადასტურებულია! ✨ მისამართი და ტელეფონის ნომერი მოგვწერეთ."


async def _handle_denial(conv_id: str) -> str:
    """Owner denied — tell customer product is unavailable."""
    agent = get_support_sales_agent()
    result = await run_agent(agent, "[მფლობელმა უარყო. თავაზიანად უთხარი რომ ეს პროდუქტი ამჟამად არ არის ხელმისაწვდომი და შესთავაზე სხვა ვარიანტი.]", conv_id)
    return result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად არ არის ხელმისაწვდომი. გსურთ სხვა ვარიანტი ნახოთ?"


@router.get("/wa-webhook")
async def wa_webhook_verify(request: Request):
    """WhatsApp webhook verification."""
    params = request.query_params
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == "tissu_wa_verify":
        return int(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/wa-webhook")
async def wa_webhook_receive(request: Request):
    """Receive WhatsApp messages from owner — forward to customer or confirm/deny."""
    body = await request.json()

    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                sender = msg.get("from", "")
                owner_number = os.getenv("OWNER_WHATSAPP", "")
                if sender != owner_number:
                    continue

                # Anti-duplicate
                wa_mid = msg.get("id", "")
                if wa_mid and wa_mid in _wa_processed_mids:
                    continue
                if wa_mid:
                    _wa_processed_mids[wa_mid] = time.time()
                    _cleanup_old_mids()

                text = msg.get("text", {}).get("body", "").strip()
                if not text:
                    continue

                conv_id = await _get_latest_conversation_id()
                if not conv_id:
                    continue

                sender_id = _extract_sender_id(conv_id)
                if not FB_PAGE_TOKEN or not sender_id:
                    continue

                # Determine action based on owner's message
                text_lower = text.lower()
                if "ვადასტურებ" in text_lower and "არ" not in text_lower:
                    reply = await _handle_confirmation(conv_id)
                elif "არ ვადასტურებ" in text_lower or ("არ" in text_lower and "ვადასტურებ" in text_lower):
                    reply = await _handle_denial(conv_id)
                elif text.startswith("უპასუხე:") or text.startswith("უპასუხე "):
                    # Owner dictates reply — send directly
                    reply = text.replace("უპასუხე:", "").replace("უპასუხე ", "").strip()
                else:
                    # Other text — forward as instruction to bot
                    agent = get_support_sales_agent()
                    result = await run_agent(agent, f"[მფლობელის ინსტრუქცია: {text}]", conv_id)
                    reply = result["reply"].strip()
                    if not reply:
                        continue

                await _send_to_customer(sender_id, reply)

    return {"status": "ok"}


@router.get("/api/owner-confirm/{conversation_id}")
async def owner_confirm(conversation_id: str):
    """Owner confirms order via link in WhatsApp."""
    sender_id = _extract_sender_id(conversation_id)
    reply = await _handle_confirmation(conversation_id)
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>✅ დადასტურებულია!</h1><p>კლიენტს ეცნობა.</p>")


@router.get("/api/owner-deny/{conversation_id}")
async def owner_deny(conversation_id: str):
    """Owner denies order via link in WhatsApp."""
    sender_id = _extract_sender_id(conversation_id)
    reply = await _handle_denial(conversation_id)
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>❌ უარყოფილია</h1><p>კლიენტს ეცნობა.</p>")
