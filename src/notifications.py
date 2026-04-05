"""WhatsApp notification helpers for Tissu Shop.

Sends notifications to the shop owner via WhatsApp Business API.
Used by webhooks and agent tools.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

WA_API_BASE = "https://graph.facebook.com/v21.0"


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
