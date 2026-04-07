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

from src.agents.support_sales import run_orchestrator
from src.db import get_db

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
    """Find the most recent customer conversation — check tickets first, then conversations."""
    pool = await get_db()
    # Try tickets first (order/payment related)
    row = await pool.fetchrow(
        "SELECT conversation_id FROM tickets WHERE conversation_id IS NOT NULL ORDER BY created_at DESC LIMIT 1"
    )
    if row and row["conversation_id"]:
        return row["conversation_id"]
    # Fallback: most recent conversation (covers link/photo forwards)
    row = await pool.fetchrow(
        "SELECT id FROM conversations WHERE agent_type = 'support_sales' ORDER BY updated_at DESC LIMIT 1"
    )
    return row["id"] if row else ""


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

    result = await run_orchestrator("[მფლობელმა დაადასტურა გადახდა. მოითხოვე მისამართი და ტელეფონი.]", conv_id)
    return result["reply"].strip() or "გადახდა დადასტურებულია! ✨ მისამართი და ტელეფონის ნომერი მოგვწერეთ."


async def _handle_denial(conv_id: str) -> str:
    """Owner denied payment — tell customer payment wasn't confirmed."""
    result = await run_orchestrator("[მფლობელმა გადახდა ვერ დაადასტურა. თავაზიანად უთხარი რომ გადახდა ვერ დადასტურდა და გთხოვთ გადაამოწმოთ ან ხელახლა გამოაგზავნოთ ქვითარი.]", conv_id)
    return result["reply"].strip() or "გადახდა ვერ დადასტურდა 😔 გთხოვთ გადაამოწმოთ და ქვითარი ხელახლა გამოგვიგზავნეთ ✨"


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
                text_upper = text.strip().upper()

                if "ვადასტურებ" in text_lower and "არ" not in text_lower:
                    reply = await _handle_confirmation(conv_id)
                    await _send_to_customer(sender_id, reply)

                elif "არ ვადასტურებ" in text_lower or ("არ" in text_lower and "ვადასტურებ" in text_lower):
                    reply = await _handle_denial(conv_id)
                    await _send_to_customer(sender_id, reply)

                elif "არ გვაქვს" in text_lower or "არა" == text_lower.strip():
                    # Owner says product not available
                    result = await run_orchestrator("[მფლობელის ინსტრუქცია: ეს მოდელი არ გვაქვს, შესთავაზე სხვა]", conv_id)
                    reply = result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად არ გვაქვს. სხვა ლამაზი მოდელები გაჩვენოთ? ✨"
                    await _send_to_customer(sender_id, reply)

                elif len(text_upper) <= 5 and any(text_upper.startswith(p) for p in ("FP", "TP", "FD", "TD")):
                    # Owner sent a product code (e.g., "FP3") — send that product to customer
                    code = text_upper
                    pool = await get_db()
                    row = await pool.fetchrow(
                        "SELECT code, model, size, price, image_url, image_url_back FROM inventory WHERE UPPER(code) = $1 AND stock > 0",
                        code,
                    )

                    if row:
                        product = dict(row)
                        # Tell agent the owner found the product
                        result = await run_orchestrator(
                            f"[მფლობელის ინსტრუქცია: კლიენტის ფოტოს {code} ემთხვევა. აჩვენე ეს პროდუქტი და ეკითხე მოეწონა თუ არა]",
                            conv_id,
                        )
                        reply = result["reply"].strip() or f"თქვენი ფოტოს მიხედვით ეს ვიპოვე ✨ მოგეწონებათ?"
                        await _send_to_customer(sender_id, reply)

                        # Send product photos
                        public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
                        async with httpx.AsyncClient(timeout=30) as client:
                            fb_api = "https://graph.facebook.com/v21.0/me/messages"
                            fb_params = {"access_token": FB_PAGE_TOKEN}

                            await client.post(fb_api, params=fb_params, json={
                                "recipient": {"id": sender_id},
                                "message": {"text": f"📌 {code}"},
                            })

                            img_url = product["image_url"]
                            if not img_url.startswith("http"):
                                img_url = public_url + img_url
                            await client.post(fb_api, params=fb_params, json={
                                "recipient": {"id": sender_id},
                                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                            })

                            if product.get("image_url_back"):
                                back_url = product["image_url_back"]
                                if not back_url.startswith("http"):
                                    back_url = public_url + back_url
                                await client.post(fb_api, params=fb_params, json={
                                    "recipient": {"id": sender_id},
                                    "message": {"attachment": {"type": "image", "payload": {"url": back_url, "is_reusable": True}}},
                                })
                    else:
                        await _send_to_customer(sender_id, "სამწუხაროდ ეს მოდელი ამჟამად არ არის მარაგში ✨")

                elif text_lower in ("მე ვპასუხობ", "ჩემია", "მე", "stop", "სტოპ"):
                    # Owner takes over — tell bot to shut up, notify owner
                    await run_orchestrator("[SYSTEM: owner_is_chatting]", conv_id)
                    from src.notifications import send_whatsapp_text
                    await send_whatsapp_text("✅ ბოტი გაჩერდა, შენ აგრძელებ. 'უპასუხე:' ტექსტით მიწერე კლიენტს.")

                elif text.startswith("უპასუხე:") or text.startswith("უპასუხე "):
                    reply = text.replace("უპასუხე:", "").replace("უპასუხე ", "").strip()
                    await _send_to_customer(sender_id, reply)

                elif text_lower in ("ბოტი", "bot", "გააგრძელე"):
                    # Resume bot — clear owner_is_chatting state
                    await run_orchestrator("[SYSTEM: owner_stopped_chatting — ბოტი ისევ აგრძელებს]", conv_id)
                    from src.notifications import send_whatsapp_text
                    await send_whatsapp_text("🤖 ბოტი ისევ ჩაირთო.")

                else:
                    # Other text — forward as instruction to bot
                    result = await run_orchestrator(f"[მფლობელის ინსტრუქცია: {text}]", conv_id)
                    reply = result["reply"].strip()
                    if reply:
                        await _send_to_customer(sender_id, reply)

    return {"status": "ok"}


@router.get("/api/owner-confirm/{conversation_id}")
async def owner_confirm(conversation_id: str):
    """Owner confirms payment via link in WhatsApp."""
    sender_id = _extract_sender_id(conversation_id)
    reply = await _handle_confirmation(conversation_id)
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>✅ დადასტურებულია!</h1><p>კლიენტს ეცნობა.</p>")


@router.get("/api/owner-deny/{conversation_id}")
async def owner_deny(conversation_id: str):
    """Owner denies payment via link in WhatsApp."""
    sender_id = _extract_sender_id(conversation_id)
    reply = await _handle_denial(conversation_id)
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>❌ უარყოფილია</h1><p>კლიენტს ეცნობა.</p>")


@router.get("/api/photo-confirm/{conversation_id}")
async def photo_confirm(conversation_id: str):
    """Owner confirms product is in stock (photo match)."""
    sender_id = _extract_sender_id(conversation_id)
    result = await run_orchestrator(
        "[მფლობელმა დაადასტურა — მარაგშია. უთხარი 'გვაქვს მარაგში ✨ გავაფორმოთ შეკვეთა?' — როცა დაეთანხმება, ეკითხე 'თიბისი თუ საქართველოს ბანკი?' სტილს ᲐᲠ ეკითხო, ფოტოებს ᲐᲠ გაუგზავნო, check_inventory ᲐᲠ გამოიძახო.]",
        conversation_id,
    )
    reply = result["reply"].strip() or "გვაქვს მარაგში ✨ გავაფორმოთ შეკვეთა?"
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>✅ მარაგშია!</h1><p>კლიენტს ეცნობა.</p>")


@router.get("/api/photo-deny/{conversation_id}")
async def photo_deny(conversation_id: str):
    """Owner says product is not in stock (photo match)."""
    sender_id = _extract_sender_id(conversation_id)
    result = await run_orchestrator(
        "[მფლობელმა უარყო — კლიენტის ფოტოზე მოდელი არ არის მარაგში. უთხარი 'სამწუხაროდ ეს მოდელი ამჟამად აღარ გვაქვს ✨ სხვა ლამაზი მოდელები გაჩვენოთ?']",
        conversation_id,
    )
    reply = result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად აღარ გვაქვს ✨ სხვა ლამაზი მოდელები გაჩვენოთ?"
    await _send_to_customer(sender_id, reply)
    return HTMLResponse("<h1>❌ არ გვაქვს</h1><p>კლიენტს ეცნობა.</p>")
