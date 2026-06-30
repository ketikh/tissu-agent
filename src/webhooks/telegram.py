"""Telegram webhook handler — owner responses.

Mirror of webhooks/whatsapp.py for Telegram. The shop owner messages the
Telegram bot, and these handlers:
- Confirm/deny photo matches and payment receipts
- Forward owner instructions to the agent for customer reply
- Let the owner take over the conversation, send specific product codes,
  or send free-form text directly to the customer

No 24-hour window restriction — works whenever the owner messages the bot.
"""
from __future__ import annotations

import logging
import os
import time

import httpx
from fastapi import APIRouter, Request

from src.agents.support_sales import get_support_sales_agent
from src.db import get_db
from src.engine import run_agent
from src.webhooks.whatsapp import (
    _get_latest_conversation_id,
    _extract_sender_id,
    _send_to_customer,
    _get_conv_tenant_id,
    _handle_confirmation,
    _handle_denial,
)

logger = logging.getLogger(__name__)

router = APIRouter()

FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")

# Anti-duplicate tracking — Telegram retries updates if our webhook returns
# non-200, so we dedupe by update_id.
_tg_processed_ids: dict[int, float] = {}


def _cleanup_old_ids() -> None:
    now = time.time()
    for key in list(_tg_processed_ids):
        if now - _tg_processed_ids[key] > 300:
            del _tg_processed_ids[key]


def _owner_chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


@router.post("/tg-webhook")
async def tg_webhook_receive(request: Request):
    """Receive messages from owner via Telegram — same command vocabulary
    as the WhatsApp webhook so the operator's muscle memory transfers."""
    body = await request.json()

    update_id = body.get("update_id")
    if update_id is not None and update_id in _tg_processed_ids:
        return {"ok": True}
    if update_id is not None:
        _tg_processed_ids[update_id] = time.time()
        _cleanup_old_ids()

    message = body.get("message") or body.get("edited_message") or {}
    chat = message.get("chat", {})
    sender_chat_id = str(chat.get("id", ""))
    owner_chat_id = _owner_chat_id()

    # Only accept messages from the configured owner — anyone else who
    # discovers the bot's username and messages it is ignored.
    if owner_chat_id and sender_chat_id != owner_chat_id:
        # Helpful one-time hint so the owner can find their chat_id on first
        # /start before they've populated TELEGRAM_CHAT_ID.
        text_in = (message.get("text") or "").strip()
        if text_in == "/start":
            await _tg_reply(sender_chat_id, f"შენი chat_id: <code>{sender_chat_id}</code>")
        return {"ok": True}

    text = (message.get("text") or "").strip()
    if not text:
        return {"ok": True}

    # /start while already configured — show identity confirmation
    if text == "/start":
        await _tg_reply(owner_chat_id, "✅ ბოტი ჩართულია. შემოვა ფოტოები/ლინკები/გადახდის ქვითრები კლიენტებიდან.")
        return {"ok": True}

    conv_id = await _get_latest_conversation_id()
    if not conv_id:
        await _tg_reply(owner_chat_id, "ℹ️ ჯერ აქტიური საუბარი არ არის.")
        return {"ok": True}

    tenant_id = await _get_conv_tenant_id(conv_id)
    sender_id = _extract_sender_id(conv_id)
    if not FB_PAGE_TOKEN or not sender_id:
        return {"ok": True}

    text_lower = text.lower()
    text_upper = text.strip().upper()

    if "ვადასტურებ" in text_lower and "არ" not in text_lower:
        reply = await _handle_confirmation(conv_id, tenant_id)
        await _send_to_customer(sender_id, reply)

    elif "არ ვადასტურებ" in text_lower or ("არ" in text_lower and "ვადასტურებ" in text_lower):
        reply = await _handle_denial(conv_id, tenant_id)
        await _send_to_customer(sender_id, reply)

    elif "არ გვაქვს" in text_lower or text_lower.strip() == "არა":
        agent = await get_support_sales_agent(tenant_id)
        result = await run_agent(
            agent,
            "[მფლობელის ინსტრუქცია: ეს მოდელი არ გვაქვს, შესთავაზე სხვა]",
            conv_id,
        )
        reply = result["reply"].strip() or "სამწუხაროდ ეს მოდელი ამჟამად არ გვაქვს. სხვა ლამაზი მოდელები გაჩვენოთ? ✨"
        await _send_to_customer(sender_id, reply)

    elif len(text_upper) <= 5 and any(text_upper.startswith(p) for p in ("FP", "TP", "FD", "TD")):
        await _handle_product_code(text_upper, conv_id, tenant_id, sender_id)

    elif text_lower in ("მე ვპასუხობ", "ჩემია", "მე", "stop", "სტოპ"):
        agent = await get_support_sales_agent(tenant_id)
        await run_agent(agent, "[SYSTEM: owner_is_chatting]", conv_id)
        await _tg_reply(owner_chat_id, "✅ ბოტი გაჩერდა, შენ აგრძელებ. 'უპასუხე:' ტექსტით მიწერე კლიენტს.")

    elif text.startswith("უპასუხე:") or text.startswith("უპასუხე "):
        reply = text.replace("უპასუხე:", "", 1).replace("უპასუხე ", "", 1).strip()
        if reply:
            await _send_to_customer(sender_id, reply)
            await _tg_reply(owner_chat_id, "✅ გავაგზავნე.")

    elif text_lower in ("ბოტი", "bot", "გააგრძელე"):
        agent = await get_support_sales_agent(tenant_id)
        await run_agent(agent, "[SYSTEM: owner_stopped_chatting — ბოტი ისევ აგრძელებს]", conv_id)
        await _tg_reply(owner_chat_id, "🤖 ბოტი ისევ ჩაირთო.")

    else:
        # Free-form: forward as instruction to the agent
        agent = await get_support_sales_agent(tenant_id)
        result = await run_agent(agent, f"[მფლობელის ინსტრუქცია: {text}]", conv_id)
        reply = result["reply"].strip()
        if reply:
            await _send_to_customer(sender_id, reply)

    return {"ok": True}


async def _tg_reply(chat_id: str, text: str) -> None:
    """Send a message back to the owner inside the Telegram chat."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not (token and chat_id):
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
    except Exception as e:
        print(f"[TG] reply error: {e}", flush=True)


async def _handle_product_code(
    code: str, conv_id: str, tenant_id: str, sender_id: str,
) -> None:
    """Owner typed a product code (e.g. 'FP3') — show that product to the customer."""
    pool = await get_db()
    row = await pool.fetchrow(
        "SELECT code, model, size, price, image_url, image_url_back "
        "FROM inventory WHERE UPPER(code) = $1 AND stock > 0",
        code,
    )

    if not row:
        await _send_to_customer(sender_id, "სამწუხაროდ ეს მოდელი ამჟამად არ არის მარაგში ✨")
        return

    product = dict(row)

    agent = await get_support_sales_agent(tenant_id)
    result = await run_agent(
        agent,
        f"[მფლობელის ინსტრუქცია: კლიენტის ფოტოს {code} ემთხვევა. "
        f"აჩვენე ეს პროდუქტი და ეკითხე მოეწონა თუ არა]",
        conv_id,
    )
    reply = result["reply"].strip() or "თქვენი ფოტოს მიხედვით ეს ვიპოვე ✨ მოგეწონებათ?"
    await _send_to_customer(sender_id, reply)

    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
    async with httpx.AsyncClient(timeout=30) as client:
        fb_api = "https://graph.facebook.com/v21.0/me/messages"
        fb_params = {"access_token": FB_PAGE_TOKEN}

        await client.post(fb_api, params=fb_params, json={
            "recipient": {"id": sender_id},
            "message": {"text": f"📌 {code}"},
        })

        img_url = product["image_url"] or ""
        if img_url and not img_url.startswith("http"):
            img_url = public_url + img_url
        if img_url:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
            })

        back_url = product.get("image_url_back") or ""
        if back_url:
            if not back_url.startswith("http"):
                back_url = public_url + back_url
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"attachment": {"type": "image", "payload": {"url": back_url, "is_reusable": True}}},
            })
