"""Channel adapters for multi-platform message ingestion.

Architecture:
  External Platform → Channel Adapter → Normalized ChatRequest → Agent Engine → Response → Channel Adapter → Platform Reply

Each adapter handles two things:
1. parse_incoming(): Convert platform-specific webhook payload → ChatRequest
2. format_outgoing(): Convert ChatResponse → platform-specific reply format

Phase 1: Web (direct API calls) — works now
Phase 2: Facebook Messenger, Instagram DM — via n8n webhooks + these adapters
"""
from __future__ import annotations

from src.models import ChatRequest, ChatResponse, CustomerContext


# ── Web Channel (default, already works) ──────────────────────

class WebAdapter:
    """Direct API calls. No transformation needed."""

    @staticmethod
    def parse_incoming(payload: dict) -> ChatRequest:
        return ChatRequest(**payload)

    @staticmethod
    def format_outgoing(response: ChatResponse) -> dict:
        return response.model_dump()


# ── Facebook Messenger Adapter ────────────────────────────────

class FacebookMessengerAdapter:
    """
    Converts Facebook Messenger webhook format to our ChatRequest.

    Facebook sends webhooks like:
    {
        "object": "page",
        "entry": [{
            "messaging": [{
                "sender": {"id": "USER_ID"},
                "message": {"text": "Hello"},
                "timestamp": 1234567890
            }]
        }]
    }

    To connect (Phase 2):
    1. Create Facebook App + Page
    2. Set webhook URL to n8n webhook endpoint
    3. n8n workflow: receive → transform with this adapter → call /api/support → send reply via FB API
    """

    @staticmethod
    def parse_incoming(payload: dict) -> ChatRequest:
        entry = payload.get("entry", [{}])[0]
        messaging = entry.get("messaging", [{}])[0]
        sender_id = messaging.get("sender", {}).get("id", "")
        message_text = messaging.get("message", {}).get("text", "")

        return ChatRequest(
            message=message_text,
            conversation_id=f"fb_{sender_id}",
            channel="facebook_messenger",
            customer_context=CustomerContext(name=None),
        )

    @staticmethod
    def format_outgoing(response: ChatResponse, recipient_id: str) -> dict:
        """Format for Facebook Send API."""
        return {
            "recipient": {"id": recipient_id},
            "message": {"text": response.reply},
        }


# ── Instagram DM Adapter ─────────────────────────────────────

class InstagramDMAdapter:
    """
    Converts Instagram DM webhook format to our ChatRequest.

    Instagram uses the same Messenger Platform API.
    Webhook format is identical to Facebook Messenger.

    To connect (Phase 2):
    1. Link Instagram to Facebook Page
    2. Enable Instagram messaging in Facebook App settings
    3. Same n8n workflow as Facebook, just different page token
    """

    @staticmethod
    def parse_incoming(payload: dict) -> ChatRequest:
        entry = payload.get("entry", [{}])[0]
        messaging = entry.get("messaging", [{}])[0]
        sender_id = messaging.get("sender", {}).get("id", "")
        message_text = messaging.get("message", {}).get("text", "")

        return ChatRequest(
            message=message_text,
            conversation_id=f"ig_{sender_id}",
            channel="instagram_dm",
            customer_context=CustomerContext(name=None),
        )

    @staticmethod
    def format_outgoing(response: ChatResponse, recipient_id: str) -> dict:
        return {
            "recipient": {"id": recipient_id},
            "message": {"text": response.reply},
        }


# ── WhatsApp Adapter (future) ─────────────────────────────────

class WhatsAppAdapter:
    """
    WhatsApp Business API webhook format.

    Webhook payload:
    {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": "PHONE_NUMBER",
                        "text": {"body": "Hello"},
                        "timestamp": "1234567890",
                        "type": "text"
                    }],
                    "contacts": [{"profile": {"name": "Customer Name"}}]
                }
            }]
        }]
    }

    To connect (Phase 2):
    1. Set up WhatsApp Business API (via Meta Business Suite)
    2. Configure webhook to n8n
    3. n8n: receive → parse → call /api/support → reply via WhatsApp API
    """

    @staticmethod
    def parse_incoming(payload: dict) -> ChatRequest:
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [{}])
        contacts = value.get("contacts", [{}])

        if messages:
            msg = messages[0]
            phone = msg.get("from", "")
            text = msg.get("text", {}).get("body", "")
            name = contacts[0].get("profile", {}).get("name") if contacts else None
        else:
            phone, text, name = "", "", None

        return ChatRequest(
            message=text,
            conversation_id=f"wa_{phone}",
            channel="whatsapp",
            customer_context=CustomerContext(name=name),
        )

    @staticmethod
    def format_outgoing(response: ChatResponse, phone_number: str) -> dict:
        return {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {"body": response.reply},
        }


# ── Adapter Registry ─────────────────────────────────────────

ADAPTERS = {
    "web": WebAdapter,
    "facebook_messenger": FacebookMessengerAdapter,
    "instagram_dm": InstagramDMAdapter,
    "whatsapp": WhatsAppAdapter,
}


def get_adapter(channel: str):
    return ADAPTERS.get(channel, WebAdapter)
