"""Facebook Messenger / Instagram DM webhook handler."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import traceback as _tb

import httpx
from fastapi import APIRouter, HTTPException, Request

from src.agents.support_sales import get_support_sales_agent
from src.db import get_db
from src.engine import run_agent
from src.notifications import send_whatsapp_image, send_whatsapp_text
from src.tools.support import _pending_photos, _ai_hints
from src.vision import download_image, is_payment_receipt

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"

_processed_mids: dict[str, float] = {}

# Track which inventory categories were already sent to each conversation
# Key: conversation_id, Value: set of "model|size" combos already sent
_sent_inventory: dict[str, set[str]] = {}

# Buffers for text↔photo pairing
_pending_text: dict[str, dict] = {}   # sender_id -> {text, mid, ...} — text waiting for photo
_pending_photo: dict[str, dict] = {}  # sender_id -> {image_url, mid, ...} — photo waiting for text

_PHOTO_HINT = re.compile(r'(გაქვთ|მოდელი|ჩანთა|ასეთი|მსგავსი|ეს\s*არის|have|this)', re.IGNORECASE)

BUFFER_SECONDS = 3


def _cleanup_old_mids() -> None:
    now = time.time()
    for key in list(_processed_mids):
        if now - _processed_mids[key] > 300:
            del _processed_mids[key]


async def _send_typing_on(sender_id: str) -> None:
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


async def _process_message(
    sender_id: str, text: str, conversation_id: str,
    channel: str, customer_name: str = "", image_url: str = "",
) -> None:
    print(f"[MSG] START: sender={sender_id}, text={text[:50]}..., image={'YES' if image_url else 'NO'}", flush=True)
    try:
        await _send_typing_on(sender_id)

        # ── Photo handling ──
        if image_url:
            image_bytes = await download_image(image_url)
            if not image_bytes:
                text = "[კლიენტმა ფოტო გამოგზავნა მაგრამ ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨']"
            else:
                is_receipt = await is_payment_receipt(image_bytes, conversation_id)
                if is_receipt:
                    cname = customer_name or "კლიენტი"
                    confirm_url = f"{PUBLIC_URL}/api/owner-confirm/{conversation_id}"
                    deny_url = f"{PUBLIC_URL}/api/owner-deny/{conversation_id}"
                    await send_whatsapp_image(
                        image_bytes,
                        caption=f"📷 {cname} — გადახდის ქვითარი\n\n✅ ვადასტურებ:\n{confirm_url}\n\n❌ არ ვადასტურებ:\n{deny_url}",
                        filename="receipt.jpg",
                    )
                    text = "[კლიენტმა გადახდის სქრინი გამოგზავნა. უთხარი 'მადლობა, გადავამოწმებ ✨' და ᲒᲐᲩᲔᲠᲓᲘ. მისამართს ᲐᲠ ეკითხო.]"
                else:
                    _pending_photos[conversation_id] = image_bytes
                    print(f"[PHOTO] Saved: {conversation_id}, {len(image_bytes)} bytes", flush=True)

                    # Run AI match — analyzes photo and compares to indexed products
                    try:
                        from src.vision_match import analyze_and_match
                        match_result = await asyncio.wait_for(
                            analyze_and_match(image_bytes),
                            timeout=30,
                        )
                        if match_result and match_result.get("matched"):
                            code = match_result["code"]
                            score = match_result["score"]
                            product = match_result.get("product", {})
                            alts = match_result.get("alternatives", [])
                            alt_str = ""
                            if alts:
                                alt_str = " | " + ", ".join(f"{a['code']}={int(a['score']*100)}%" for a in alts[:2])
                            # Store richer hint with model + size + price
                            _ai_hints[conversation_id] = {
                                "code": code,
                                "score": score,
                                "model": product.get("model", ""),
                                "size": product.get("size", ""),
                                "price": product.get("price", 0),
                                "image_url": product.get("image_url", ""),
                                "text": f"\n🤖 AI რეკომენდაცია: {code} ({int(score*100)}%){alt_str}",
                            }
                            print(f"[PHOTO] AI hint stored: {code} (score={score})", flush=True)
                        else:
                            print(f"[PHOTO] AI no match", flush=True)
                    except Exception as e:
                        print(f"[PHOTO] AI match skipped: {e}", flush=True)

                    text = "[კლიენტმა პროდუქტის ფოტო გამოგზავნა. ჯერ ეკითხე: 'პატარა თუ დიდი ზომაში გაინტერესებთ? ✨' სტილს ᲐᲠ ეკითხო! ზომა რომ გეცოდინება, გამოიძახე forward_photo_to_owner იმ ზომით და უპასუხე 'ერთი წუთით, გადავამოწმებ ✨'.]"

        # ── Link handling — forward to owner like photo ──
        elif text and re.search(r'https?://', text):
            link_match = re.search(r'https?://\S+', text)
            link_url = link_match.group(0) if link_match else ""
            # Extract user's text without the link
            user_text = re.sub(r'https?://\S+', '', text).strip()
            # Forward link to owner via WhatsApp
            wa_msg = f"🔗 კლიენტმა ბმული გამოგზავნა:\n{link_url}"
            if user_text:
                wa_msg += f"\n💬 {user_text}"
            wa_msg += f"\n\n👤 {customer_name or 'კლიენტი'}"
            await send_whatsapp_text(wa_msg)
            if user_text:
                # User also wrote something ("ეს გაქვთ?") — let bot respond to the question
                text = f"[კლიენტმა ბმული გამოგზავნა პროდუქტის. მფლობელს გადაეგზავნა. კლიენტი ეკითხება: '{user_text}'. ეკითხე ზომა: პატარა თუ დიდი?]"
            else:
                # Only a link, no text
                text = "[კლიენტმა ბმული გამოგზავნა. მფლობელს გადაეგზავნა. უთხარი 'გადავამოწმებ ✨']"

        # ── Customer name ──
        if customer_name:
            text = f"[SYSTEM: customer_name={customer_name}]\n{text}"

        # ── Run agent ──
        print(f"[MSG] Calling agent...", flush=True)
        agent = get_support_sales_agent()
        try:
            result = await run_agent(agent, text, conversation_id)
        except Exception as e:
            print(f"[MSG] Agent error: {e}", flush=True)
            await send_whatsapp_text(f"🚨 აგენტის შეცდომა!\n{text[:200]}\n{str(e)[:300]}")
            result = {"reply": "გადავამოწმებ და მოგწერთ ✨", "tool_calls_made": [], "tool_results_data": {}}

        print(f"[MSG] Agent replied: {result['reply'][:80]}...", flush=True)

        # ── Send reply ──
        if not FB_PAGE_TOKEN:
            return

        async with httpx.AsyncClient(timeout=30) as client:
            fb_api = "https://graph.facebook.com/v21.0/me/messages"
            fb_params = {"access_token": FB_PAGE_TOKEN}

            reply_text = result["reply"].strip()
            reply_text = re.sub(r'\[[^\]]{10,}\]', '', reply_text).strip()
            reply_text = re.sub(r'https?://\S+', '', reply_text).strip()
            reply_text = re.sub(r'/static/\S+', '', reply_text).strip()
            reply_text = re.sub(r'\n{3,}', '\n\n', reply_text).strip()

            if not reply_text:
                # Empty reply = bot chose silence (e.g. waiting for address+phone)
                print(f"[MSG] Empty reply — not sending anything", flush=True)
            else:
                fb_resp = await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"text": reply_text[:2000]},
                })
                print(f"[MSG] FB reply: {fb_resp.status_code} {fb_resp.text[:200]}", flush=True)

            await _send_product_images(client, fb_api, fb_params, sender_id, result, conversation_id)

        print(f"[MSG] DONE", flush=True)

    except Exception as e:
        print(f"[MSG] CRASH: {e}", flush=True)
        _tb.print_exc()
        # Always try to respond to customer even on crash
        try:
            if FB_PAGE_TOKEN and sender_id:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        "https://graph.facebook.com/v21.0/me/messages",
                        params={"access_token": FB_PAGE_TOKEN},
                        json={"recipient": {"id": sender_id}, "message": {"text": "ერთი წუთით, გადავამოწმებ ✨"}},
                    )
        except Exception:
            pass


async def _send_product_images(
    client: httpx.AsyncClient, fb_api: str, fb_params: dict,
    sender_id: str, result: dict, conversation_id: str = "",
) -> None:
    tool_data = result.get("tool_results_data", {})
    inventory_data = tool_data.get("check_inventory")
    if not inventory_data or not inventory_data.get("found"):
        return

    items = inventory_data.get("items", [])
    if not items:
        return

    # Check if we already sent this exact category to this conversation
    if conversation_id and len(items) > 1:
        first = items[0]
        category_key = f"{first.get('model', '')}|{first.get('size', '')}"
        already_sent = _sent_inventory.get(conversation_id, set())
        if category_key in already_sent:
            print(f"[PHOTO] Skipping — already sent {category_key} to {conversation_id}", flush=True)
            return
        already_sent.add(category_key)
        _sent_inventory[conversation_id] = already_sent

    sent_codes: set[str] = set()
    photos_sent = 0
    for item in items:
        code = item.get("code", "")
        front = item.get("image_url", "")
        if not front or not code or code in sent_codes:
            continue
        sent_codes.add(code)

        try:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": f"📌 {code}"},
            })
        except Exception:
            pass

        try:
            img_url = PUBLIC_URL + front if front.startswith("/") else front
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
            })
            photos_sent += 1
        except Exception:
            pass

        back = item.get("image_url_back", "")
        if back:
            try:
                img_url = PUBLIC_URL + back if back.startswith("/") else back
                await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                })
            except Exception:
                pass

    # After photos — tell customer to pick a code
    if photos_sent > 0 and len(items) > 1:
        try:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": "შეარჩიეთ მოდელი და შესაბამისი კოდი მოგვწერეთ ✨"},
            })
        except Exception:
            pass


@router.get("/webhook")
async def webhook_verify(request: Request):
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

            image_url = ""
            for att in message.get("attachments", []):
                if att.get("type") == "image":
                    image_url = att.get("payload", {}).get("url", "")
                    break

            if not text and not image_url:
                continue

            if mid and mid in _processed_mids:
                continue
            if mid:
                _processed_mids[mid] = time.time()
                _cleanup_old_mids()

            channel = "instagram_dm" if body["object"] == "instagram" else "facebook_messenger"
            conversation_id = f"{channel}_{sender_id}"

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

            # === BUFFERING: text↔photo pairing ===

            # Case 1: Photo arrived
            if image_url:
                if sender_id in _pending_text:
                    # Text was waiting → combine
                    buffered = _pending_text.pop(sender_id)
                    asyncio.create_task(_process_message(
                        sender_id, buffered["text"], conversation_id, channel, customer_name, image_url,
                    ))
                else:
                    # Photo alone → buffer, wait for text
                    _pending_photo[sender_id] = {
                        "image_url": image_url, "mid": mid, "conv_id": conversation_id,
                        "channel": channel, "customer_name": customer_name,
                    }

                    async def _photo_wait(sid: str, stored_mid: str):
                        await asyncio.sleep(BUFFER_SECONDS)
                        if sid in _pending_photo and _pending_photo[sid]["mid"] == stored_mid:
                            buf = _pending_photo.pop(sid)
                            asyncio.create_task(_process_message(
                                sid, "", buf["conv_id"], buf["channel"], buf["customer_name"], buf["image_url"],
                            ))

                    asyncio.create_task(_photo_wait(sender_id, mid))
                continue

            # Case 2: Text arrived
            if text:
                if sender_id in _pending_photo:
                    # Photo was waiting → combine
                    buffered = _pending_photo.pop(sender_id)
                    asyncio.create_task(_process_message(
                        sender_id, text, conversation_id, channel, customer_name, buffered["image_url"],
                    ))
                elif _PHOTO_HINT.search(text):
                    # Text hints at photo → buffer, wait for photo
                    _pending_text[sender_id] = {
                        "text": text, "mid": mid, "conv_id": conversation_id,
                        "channel": channel, "customer_name": customer_name,
                    }

                    async def _text_wait(sid: str, stored_mid: str):
                        await asyncio.sleep(BUFFER_SECONDS)
                        if sid in _pending_text and _pending_text[sid]["mid"] == stored_mid:
                            buf = _pending_text.pop(sid)
                            asyncio.create_task(_process_message(
                                sid, buf["text"], buf["conv_id"], buf["channel"], buf["customer_name"], "",
                            ))

                    asyncio.create_task(_text_wait(sender_id, mid))
                else:
                    # Normal text → process immediately
                    asyncio.create_task(_process_message(
                        sender_id, text, conversation_id, channel, customer_name, "",
                    ))
                continue

    return {"status": "ok"}
