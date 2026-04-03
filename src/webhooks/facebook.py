"""Facebook Messenger / Instagram DM webhook handler.

Receives messages from Facebook, processes them through the agent,
and sends replies back via the Facebook Send API.

Key feature: Message buffering — when customer sends "ეს გაქვთ?" (text),
Facebook often delivers the photo as a separate message 1-3 seconds later.
We buffer text messages that look like "photo incoming" for 3 seconds
before processing, so text + photo get combined.
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
from src.tools.support import check_inventory
from src.vision import ImageAnalysisResult, analyze_image, download_image

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"

# Anti-duplicate: track recently processed message IDs
_processed_mids: dict[str, float] = {}

# Message buffer: holds text messages waiting for a potential photo
# sender_id -> {text, mid, conversation_id, channel, customer_name, timestamp}
_pending_text: dict[str, dict] = {}

# Patterns that suggest a photo is coming next
_PHOTO_INCOMING_PATTERNS = re.compile(
    r'(ეს\s*(გაქვთ|მოდელი|ჩანთა)|გაქვთ\s*ეს|ნახეთ|მაქვს\s*ეს|ასეთი|მსგავსი|'
    r'ეს\s*არის|ამას|ამისთანა|ესეთი|have\s*this|do\s*you\s*have|like\s*this)',
    re.IGNORECASE,
)

BUFFER_WAIT_SECONDS = 3


def _cleanup_old_mids() -> None:
    """Remove message IDs older than 5 minutes."""
    now = time.time()
    for key in list(_processed_mids):
        if now - _processed_mids[key] > 300:
            del _processed_mids[key]


async def _send_typing_on(sender_id: str) -> None:
    """Show 'typing...' indicator to customer."""
    if not FB_PAGE_TOKEN:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                "https://graph.facebook.com/v21.0/me/messages",
                params={"access_token": FB_PAGE_TOKEN},
                json={"recipient": {"id": sender_id}, "sender_action": "typing_on"},
            )
    except Exception:
        pass


def _build_image_context(image_url: str, analysis: ImageAnalysisResult) -> str:
    """Build context string for the agent based on image analysis result."""
    if analysis.image_type == "payment_receipt":
        return "[კლიენტმა გადახდის ქვითარი/სქრინი გამოგზავნა. უთხარი 'მადლობა, გადავამოწმებ ✨' და ᲒᲐᲩᲔᲠᲓᲘ! მისამართს ᲐᲠ ეკითხო! notify_owner ᲐᲠ გამოიძახო!]"

    if analysis.similar_codes:
        codes_str = ", ".join(analysis.similar_codes[:3])
        return f"[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელები ვიპოვეთ: {codes_str}. check_inventory გამოიძახე და ეს კოდები აჩვენე. უთხარი 'თქვენი ფოტოს მიხედვით ეს ვიპოვე ✨'. კოდებს და URL-ებს ტექსტში ᲐᲠ ჩადო! notify_owner ᲐᲠ გამოიძახო!]"

    return "[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელი ვერ ვიპოვეთ. მფლობელს უკვე ეცნობა. უთხარი 'სამწუხაროდ ზუსტად ასეთი ამჟამად არ გვაქვს, სხვა ლამაზი მოდელები გაჩვენოთ? ✨'. notify_owner ᲐᲠ გამოიძახო!]"


async def _handle_image(
    text: str, image_url: str, conversation_id: str, customer_name: str,
) -> tuple[str, dict | None]:
    """Analyze customer image. Returns (agent_context, inventory_data_or_None).

    When products are found, inventory_data is returned so _process_message
    can send photos DIRECTLY without the agent needing to call check_inventory.
    """
    image_bytes = await download_image(image_url)
    if not image_bytes:
        return "[კლიენტმა ფოტო გამოგზავნა. ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨' და გამოიძახე notify_owner]", None

    try:
        analysis = await analyze_image(image_bytes, conversation_id)
        cname = customer_name or "კლიენტი"

        if analysis.image_type == "payment_receipt":
            confirm_url = f"{PUBLIC_URL}/api/owner-confirm/{conversation_id}"
            deny_url = f"{PUBLIC_URL}/api/owner-deny/{conversation_id}"
            await send_whatsapp_image(
                image_bytes,
                caption=f"📷 {cname} — გადახდის ქვითარი\n\n✅ ვადასტურებ:\n{confirm_url}\n\n❌ არ ვადასტურებ:\n{deny_url}",
                filename="receipt.jpg",
            )
            return "[კლიენტმა გადახდის ქვითარი/სქრინი გამოგზავნა. უთხარი 'მადლობა, გადავამოწმებ ✨' და ᲒᲐᲩᲔᲠᲓᲘ! მისამართს ᲐᲠ ეკითხო! notify_owner ᲐᲠ გამოიძახო!]", None

        if analysis.similar_codes:
            # Fetch matched products directly — bypass agent
            codes_search = " ".join(analysis.similar_codes)
            inventory_data = await check_inventory(search=codes_search)
            # Filter to only matched codes
            if inventory_data.get("found"):
                matched_items = [
                    item for item in inventory_data["items"]
                    if item.get("code") in analysis.similar_codes
                ]
                inventory_data = {"found": bool(matched_items), "items": matched_items, "count": len(matched_items)}

            return "[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელები ვიპოვეთ და ფოტოებს ავტომატურად ვუგზავნით. უთხარი 'თქვენი ფოტოს მიხედვით ეს ვიპოვე ✨ მოგეწონებათ რომელიმე?' კოდებს ტექსტში ᲐᲠ ჩადო! ზომას ᲐᲠ ეკითხო! notify_owner ᲐᲠ გამოიძახო!]", inventory_data

        # Not found
        await send_whatsapp_image(
            image_bytes,
            caption=f"📷 {cname} ეძებს ამ მოდელს. მარაგში ვერ ვიპოვეთ.",
        )
        return "[კლიენტმა ფოტო გამოგზავნა. მსგავსი მოდელი ვერ ვიპოვეთ. მფლობელს უკვე ეცნობა. უთხარი 'სამწუხაროდ ზუსტად ასეთი ამჟამად არ გვაქვს, სხვა ლამაზი მოდელები გაჩვენოთ? ✨'. notify_owner ᲐᲠ გამოიძახო!]", None

    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        return "[კლიენტმა ფოტო გამოგზავნა. ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨' და გამოიძახე notify_owner]", None


async def _process_message(
    sender_id: str, text: str, conversation_id: str,
    channel: str, customer_name: str = "", image_url: str = "",
) -> None:
    """Process a message in the background — agent + reply + images."""
    # Show typing indicator while processing
    await _send_typing_on(sender_id)

    # Handle image analysis — may return inventory data for direct photo sending
    photo_inventory_data = None
    if image_url:
        text, photo_inventory_data = await _handle_image(text, image_url, conversation_id, customer_name)
        await _send_typing_on(sender_id)

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

    # Inject photo-matched inventory data (bypasses agent's check_inventory call)
    if photo_inventory_data:
        if "tool_results_data" not in result:
            result["tool_results_data"] = {}
        result["tool_results_data"]["check_inventory"] = photo_inventory_data

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


async def _process_buffered_text(sender_id: str, buffered: dict) -> None:
    """Process a buffered text message after waiting for a photo that never came."""
    asyncio.create_task(_process_message(
        sender_id,
        buffered["text"],
        buffered["conversation_id"],
        buffered["channel"],
        buffered["customer_name"],
    ))


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
    """Receive messages from Facebook Messenger / Instagram DM.

    Implements message buffering: if customer sends "ეს გაქვთ?" (text only),
    we wait 3 seconds for a potential photo before processing.
    """
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

            # === MESSAGE BUFFERING LOGIC ===

            # Case 1: Photo arrived — check if there's buffered text from same sender
            if image_url:
                combined_text = text or ""
                if sender_id in _pending_text:
                    buffered = _pending_text.pop(sender_id)
                    combined_text = buffered["text"] + ("\n" + text if text else "")

                combined_text = combined_text + f"\n[კლიენტმა გამოგზავნა ფოტო: {image_url}]"
                asyncio.create_task(_process_message(
                    sender_id, combined_text, conversation_id, channel, customer_name, image_url,
                ))
                continue

            # Case 2: Text only — check if it looks like "photo is coming"
            if text and _PHOTO_INCOMING_PATTERNS.search(text):
                # Buffer this text and wait for photo
                _pending_text[sender_id] = {
                    "text": text,
                    "mid": mid,
                    "conversation_id": conversation_id,
                    "channel": channel,
                    "customer_name": customer_name,
                }

                async def _wait_and_process(sid: str, stored_mid: str):
                    await asyncio.sleep(BUFFER_WAIT_SECONDS)
                    # If still pending (no photo arrived), process text alone
                    if sid in _pending_text and _pending_text[sid]["mid"] == stored_mid:
                        buffered = _pending_text.pop(sid)
                        await _process_buffered_text(sid, buffered)

                asyncio.create_task(_wait_and_process(sender_id, mid))
                continue

            # Case 3: Regular text — check for links, then process immediately
            if re.search(r'https?://', text):
                text += "\n[კლიენტმა ბმული/ლინკი გამოგზავნა. უთხარი: 'ბმულებს სამწუხაროდ ვერ ვხსნი 😊 თუ შეგიძლიათ, ფოტო გამომიგზავნეთ და გადავამოწმებ ✨'. notify_owner ᲐᲠ გამოიძახო!]"

            asyncio.create_task(_process_message(
                sender_id, text, conversation_id, channel, customer_name, image_url,
            ))

    return {"status": "ok"}
