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
from src.tools.support import _pending_photos
from src.vision import download_image, is_payment_receipt

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"

_processed_mids: dict[str, float] = {}


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
                    text = "[კლიენტმა პროდუქტის ფოტო გამოგზავნა. ეკითხე მხოლოდ ზომა: პატარა თუ დიდი? (თუ ზომა უკვე იცი — პირდაპირ forward_photo_to_owner გამოიძახე). სტილს ᲐᲠ ეკითხო! ზომა რომ გეცოდინება, forward_photo_to_owner გამოიძახე და უთხარი 'გადავამოწმებ ✨'.]"

        # ── Link handling ──
        elif text and re.search(r'https?://', text):
            text += "\n[კლიენტმა ბმული გამოგზავნა. უთხარი 'ბმულებს სამწუხაროდ ვერ ვხსნი 😊 თუ შეგიძლიათ ფოტო გამომიგზავნეთ ✨']"

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
                reply_text = "გადავამოწმებ და მოგწერთ ✨"

            fb_resp = await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": reply_text[:2000]},
            })
            print(f"[MSG] FB reply: {fb_resp.status_code} {fb_resp.text[:200]}", flush=True)

            await _send_product_images(client, fb_api, fb_params, sender_id, result)

        print(f"[MSG] DONE", flush=True)

    except Exception as e:
        print(f"[MSG] CRASH: {e}", flush=True)
        _tb.print_exc()


async def _send_product_images(
    client: httpx.AsyncClient, fb_api: str, fb_params: dict,
    sender_id: str, result: dict,
) -> None:
    tool_data = result.get("tool_results_data", {})
    inventory_data = tool_data.get("check_inventory")
    if not inventory_data or not inventory_data.get("found"):
        return

    sent_codes: set[str] = set()
    for item in inventory_data.get("items", []):
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

            asyncio.create_task(_process_message(
                sender_id, text, conversation_id, channel, customer_name, image_url,
            ))

    return {"status": "ok"}
