"""Owner notification helpers for Tissu Shop.

Sends notifications to the shop owner via Telegram (primary, no 24h window)
and WhatsApp (legacy fallback). Used by webhooks and agent tools.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

WA_API_BASE = "https://graph.facebook.com/v21.0"
TG_API_BASE = "https://api.telegram.org"


def _get_wa_config() -> tuple[str, str, str]:
    """Return (phone_id, token, owner_number). Empty strings if not configured."""
    return (
        os.getenv("WA_PHONE_ID", ""),
        os.getenv("WA_TOKEN", ""),
        os.getenv("OWNER_WHATSAPP", ""),
    )


async def send_whatsapp_text(message: str) -> bool:
    """Send a text message to the owner via WhatsApp."""
    phone_id, token, owner = _get_wa_config()
    if not (phone_id and token and owner):
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{WA_API_BASE}/{phone_id}/messages",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "messaging_product": "whatsapp",
                    "to": owner,
                    "type": "text",
                    "text": {"body": message},
                },
            )
        return True
    except Exception as e:
        logger.error(f"WhatsApp text failed: {e}")
        return False


async def send_whatsapp_image(image_bytes: bytes, caption: str, filename: str = "photo.jpg") -> bool:
    """Upload image to WhatsApp and send to owner with caption."""
    phone_id, token, owner = _get_wa_config()
    if not (phone_id and token and owner):
        return False

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Upload media
            upload_resp = await client.post(
                f"{WA_API_BASE}/{phone_id}/media",
                headers={"Authorization": f"Bearer {token}"},
                data={"messaging_product": "whatsapp", "type": "image/jpeg"},
                files={"file": (filename, image_bytes, "image/jpeg")},
            )
            upload_data = upload_resp.json()
            media_id = upload_data.get("id", "")
            print(f"[WA] Upload response: {upload_resp.status_code} media_id={media_id}")
            if not media_id:
                print(f"[WA] Upload FAILED: {upload_data}")
                return False

            # Send image message
            send_resp = await client.post(
                f"{WA_API_BASE}/{phone_id}/messages",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "messaging_product": "whatsapp",
                    "to": owner,
                    "type": "image",
                    "image": {"id": media_id, "caption": caption},
                },
            )
            send_data = send_resp.json()
            print(f"[WA] Send response: {send_resp.status_code} data={send_data}")
            if "error" in send_data:
                print(f"[WA] Send ERROR: {send_data['error']}")
                return False
        return True
    except Exception as e:
        print(f"[WA] Exception: {e}")
        return False


# ── Telegram ──────────────────────────────────────────────────────────────────
# No 24-hour window restriction, free, supports inline buttons + photo+caption.

def _get_tg_config() -> tuple[str, str]:
    """Return (bot_token, owner_chat_id). Empty strings if not configured."""
    return (
        os.getenv("TELEGRAM_BOT_TOKEN", ""),
        os.getenv("TELEGRAM_CHAT_ID", ""),
    )


async def send_telegram_text(message: str) -> bool:
    """Send a text message to the owner via Telegram."""
    token, chat_id = _get_tg_config()
    if not (token and chat_id):
        print("[TG] Not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID", flush=True)
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{TG_API_BASE}/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "disable_web_page_preview": True,
                },
            )
            data = resp.json()
            if not data.get("ok"):
                print(f"[TG] Text send failed: {data}", flush=True)
                return False
        return True
    except Exception as e:
        print(f"[TG] Text exception: {e}", flush=True)
        return False


async def send_telegram_image(image_bytes: bytes, caption: str, filename: str = "photo.jpg") -> bool:
    """Send a photo with caption to the owner via Telegram.

    Telegram captions are limited to 1024 chars — longer captions are sent
    as a follow-up text message so confirm/deny URLs aren't truncated.
    """
    token, chat_id = _get_tg_config()
    if not (token and chat_id):
        print("[TG] Not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID", flush=True)
        return False

    short_caption = caption[:1024] if caption else ""
    overflow = caption[1024:] if caption and len(caption) > 1024 else ""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            files = {"photo": (filename, image_bytes, "image/jpeg")}
            data = {"chat_id": chat_id, "caption": short_caption}
            resp = await client.post(
                f"{TG_API_BASE}/bot{token}/sendPhoto",
                data=data,
                files=files,
            )
            resp_data = resp.json()
            if not resp_data.get("ok"):
                print(f"[TG] Photo send failed: {resp_data}", flush=True)
                return False
            if overflow:
                await client.post(
                    f"{TG_API_BASE}/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": overflow, "disable_web_page_preview": True},
                )
        return True
    except Exception as e:
        print(f"[TG] Photo exception: {e}", flush=True)
        return False


# ── Unified dispatch ─────────────────────────────────────────────────────────
# Notify the owner via Telegram first (preferred). Falls back to WhatsApp
# only if TELEGRAM_BOT_TOKEN isn't set, so the migration is just an env var
# flip.

async def notify_owner_text(message: str) -> bool:
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        return await send_telegram_text(message)
    return await send_whatsapp_text(message)


async def notify_owner_image(image_bytes: bytes, caption: str, filename: str = "photo.jpg") -> bool:
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        return await send_telegram_image(image_bytes, caption=caption, filename=filename)
    return await send_whatsapp_image(image_bytes, caption=caption, filename=filename)
