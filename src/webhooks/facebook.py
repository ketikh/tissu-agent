"""Facebook Messenger / Instagram DM webhook handler.

Receives messages from Facebook, processes them through the agent,
and sends replies back via the Facebook Send API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException, Request

from src.agents.support_sales import get_support_sales_agent
from src.engine import run_agent
from src.notifications import send_whatsapp_image, send_whatsapp_text
from src.vision import ImageAnalysisResult, analyze_image, download_image

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"

# Anti-duplicate: track recently processed message IDs
_processed_mids: dict[str, float] = {}


def _cleanup_old_mids() -> None:
    """Remove message IDs older than 5 minutes."""
    now = time.time()
    for key in list(_processed_mids):
        if now - _processed_mids[key] > 300:
            del _processed_mids[key]


def _build_image_context(image_url: str, analysis: ImageAnalysisResult) -> str:
    """Build context string for the agent based on image analysis result."""
    original_tag = f"[კლიენტმა გამოგზავნა ფოტო: {image_url}]"

    if analysis.image_type == "payment_receipt":
        return "[კლიენტმა გადახდის ქვითარი/სქრინი გამოგზავნა. უთხარი 'მადლობა, გადავამოწმებ ✨' და ᲒᲐᲩᲔᲠᲓᲘ! მისამართს ᲐᲠ ეკითხო! notify_owner ᲐᲠ გამოიძახო!]"

    if analysis.similar_codes:
        codes_str = ", ".join(analysis.similar_codes[:5])
        return f"[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელები ვიპოვეთ: {codes_str}. check_inventory გამოიძახე და ეს კოდები აჩვენე. უთხარი 'თქვენი ფოტოს მიხედვით ეს ვიპოვე ✨'. კოდებს და URL-ებს ტექსტში ᲐᲠ ჩადო! notify_owner ᲐᲠ გამოიძახო!]"

    return "[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელი ვერ ვიპოვეთ. მფლობელს უკვე ეცნობა. უთხარი 'სამწუხაროდ ზუსტად ასეთი ამჟამად არ გვაქვს, სხვა ლამაზი მოდელები გაჩვენოთ? ✨'. notify_owner ᲐᲠ გამოიძახო!]"


async def _handle_image(
    text: str, image_url: str, conversation_id: str, customer_name: str,
) -> str:
    """Analyze customer image and update message text with context."""
    image_bytes = await download_image(image_url)
    if not image_bytes:
        return text.replace(
            f"[კლიენტმა გამოგზავნა ფოტო: {image_url}]",
            "[კლიენტმა ფოტო გამოგზავნა. ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨' და გამოიძახე notify_owner]",
        )

    try:
        analysis = await analyze_image(image_bytes, conversation_id)
        cname = customer_name or "კლიენტი"

        if analysis.image_type == "payment_receipt":
            # Forward receipt to owner via WhatsApp
            public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
            confirm_url = f"{public_url}/api/owner-confirm/{conversation_id}"
            deny_url = f"{public_url}/api/owner-deny/{conversation_id}"
            await send_whatsapp_image(
                image_bytes,
                caption=f"📷 {cname} — გადახდის ქვითარი\n\n✅ ვადასტურებ:\n{confirm_url}\n\n❌ არ ვადასტურებ:\n{deny_url}",
                filename="receipt.jpg",
            )
        elif not analysis.similar_codes:
            # Product not found — notify owner with photo
            await send_whatsapp_image(
                image_bytes,
                caption=f"📷 {cname} ეძებს ამ მოდელს. მარაგში ვერ ვიპოვეთ.",
            )

        return _build_image_context(image_url, analysis)

    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        return text.replace(
            f"[კლიენტმა გამოგზავნა ფოტო: {image_url}]",
            "[კლიენტმა ფოტო გამოგზავნა. ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨' და გამოიძახე notify_owner]",
        )


async def _process_message(
    sender_id: str, text: str, conversation_id: str,
    channel: str, customer_name: str = "", image_url: str = "",
) -> None:
    """Process a message in the background — agent + reply + images."""
    # Handle image analysis
    if image_url:
        text = await _handle_image(text, image_url, conversation_id, customer_name)

    # Add customer name as hidden context
    if customer_name:
        text = f"[SYSTEM: customer_name={customer_name}]\n{text}"

    # Run agent
    agent = get_support_sales_agent()
    try:
        result = await run_agent(agent, text, conversation_id)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        await send_whatsapp_text(f"🚨 აგენტის შეცდომა!\nკლიენტის მესიჯი: {text[:200]}\nშეცდომა: {str(e)[:300]}")
        result = {"reply": "გადავამოწმებ და მოგწერთ ✨", "tool_calls_made": [], "tool_results_data": {}}

    # Send reply back via Facebook/Instagram API
    if not FB_PAGE_TOKEN:
        return

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            fb_api = "https://graph.facebook.com/v21.0/me/messages"
            fb_params = {"access_token": FB_PAGE_TOKEN}

            reply_text = result["reply"].strip()
            # Clean internal instructions from reply
            reply_text = re.sub(r'\(აქ ავტომატურად[^)]*\)', '', reply_text).strip()
            reply_text = re.sub(r'\[SYSTEM:[^\]]*\]', '', reply_text).strip()
            # Clean URLs, image references, file paths from reply
            reply_text = re.sub(r'https?://\S+', '', reply_text).strip()
            reply_text = re.sub(r'\[Image[^\]]*\]', '', reply_text).strip()
            reply_text = re.sub(r'\[Photo[^\]]*\]', '', reply_text).strip()
            reply_text = re.sub(r'/static/\S+', '', reply_text).strip()
            # Clean leftover empty lines and brackets
            reply_text = re.sub(r'\n{3,}', '\n\n', reply_text).strip()
            if not reply_text:
                if image_url:
                    reply_text = "მადლობა, გადავამოწმებ ✨"
                else:
                    reply_text = "გადავამოწმებ და მოგწერთ ✨"
                    await send_whatsapp_text(f"⚠️ ბოტი ვერ უპასუხა!\nკლიენტის მესიჯი: {text[:200]}")

            # Send text reply
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": reply_text[:2000]},
            })

            # Send product images if inventory was checked
            await _send_product_images(client, fb_api, fb_params, sender_id, result)

    except Exception as e:
        logger.error(f"Failed to send FB reply: {e}", exc_info=True)
        await send_whatsapp_text(f"⚠️ პასუხის გაგზავნა ვერ მოხერხდა!\nკლიენტი: {sender_id}\nშეცდომა: {str(e)[:300]}")


async def _send_product_images(
    client: httpx.AsyncClient, fb_api: str, fb_params: dict,
    sender_id: str, result: dict,
) -> None:
    """Send product images (code + front + back) after inventory check. No duplicates."""
    tool_data = result.get("tool_results_data", {})
    inventory_data = tool_data.get("check_inventory")
    if not inventory_data or not inventory_data.get("found"):
        return

    sent_codes: set[str] = set()
    for item in inventory_data.get("items", []):
        code = item.get("code", "")
        front = item.get("image_url", "")
        back = item.get("image_url_back", "")
        if not front or not code:
            continue

        # Skip duplicates
        if code in sent_codes:
            continue
        sent_codes.add(code)

        # Send product code
        if code:
            try:
                await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"text": f"📌 {code}"},
                })
            except Exception:
                pass

        # Send front photo
        try:
            img_url = PUBLIC_URL + front if front.startswith("/") else front
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
            })
        except Exception:
            pass

        # Send back photo
        if back:
            try:
                img_url = PUBLIC_URL + back if back.startswith("/") else back
                await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                })
            except Exception:
                pass


@router.get("/webhook")
async def webhook_verify(request: Request):
    """Facebook webhook verification (GET request)."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def webhook_receive(request: Request):
    """Receive messages from Facebook Messenger / Instagram DM."""
    body = await request.json()

    if body.get("object") not in ("page", "instagram"):
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            if event.get("delivery") or event.get("read"):
                continue

            sender_id = event.get("sender", {}).get("id", "")
            message = event.get("message", {})
            text = message.get("text", "")
            mid = message.get("mid", "")

            if message.get("is_echo") or sender_id == PAGE_ID or not sender_id:
                continue

            # Check for image attachments
            image_url = ""
            for att in message.get("attachments", []):
                if att.get("type") == "image":
                    image_url = att.get("payload", {}).get("url", "")
                    break

            if not text and not image_url:
                continue

            # Add image context tag
            if image_url:
                text = (text or "") + f"\n[კლიენტმა გამოგზავნა ფოტო: {image_url}]"

            # Link detection (not a photo)
            if not image_url and text and re.search(r'https?://', text):
                text += "\n[კლიენტმა ბმული/ლინკი გამოგზავნა. უთხარი: 'ბმულებს სამწუხაროდ ვერ ვხსნი 😊 თუ შეგიძლიათ, ფოტო გამომიგზავნეთ და გადავამოწმებ ✨'. notify_owner ᲐᲠ გამოიძახო!]"

            # Anti-duplicate check
            if mid and mid in _processed_mids:
                continue
            if mid:
                _processed_mids[mid] = time.time()
                _cleanup_old_mids()

            channel = "instagram_dm" if body["object"] == "instagram" else "facebook_messenger"
            conversation_id = f"{channel}_{sender_id}"

            # Get customer name from Facebook profile
            customer_name = ""
            if FB_PAGE_TOKEN:
                try:
                    async with httpx.AsyncClient(timeout=5) as c:
                        resp = await c.get(
                            f"https://graph.facebook.com/v21.0/{sender_id}",
                            params={"fields": "first_name,last_name,name", "access_token": FB_PAGE_TOKEN},
                        )
                        profile = resp.json()
                        customer_name = profile.get("name", "") or f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
                except Exception:
                    pass

            # Process in background — return 200 to Facebook immediately
            asyncio.create_task(_process_message(sender_id, text, conversation_id, channel, customer_name, image_url))

    return {"status": "ok"}
