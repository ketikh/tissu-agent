"""Tap-to-copy values in Telegram owner messages (src/notifications.py).

Telegram renders ``<code>`` spans in monospace and copies them on tap, so
bank account numbers and phone numbers the owner forwards to customers
must be wrapped that way — and the rest of the message must be escaped so
customer-typed text can't break the HTML."""
from __future__ import annotations

import pytest

import src.notifications as notifications
from src.notifications import telegram_html_with_copyable, telegram_text_payload


def test_wraps_each_copyable_value_in_code_tags():
    text = telegram_html_with_copyable(
        "📱 599123456\nთიბისი: GE58TB7085345064300066",
        ["599123456", "GE58TB7085345064300066"],
    )

    assert "📱 <code>599123456</code>" in text
    assert "თიბისი: <code>GE58TB7085345064300066</code>" in text


def test_escapes_html_in_the_message_body():
    text = telegram_html_with_copyable("👤 <script> & co\n📱 599", ["599"])

    assert "&lt;script&gt; &amp; co" in text
    assert "<code>599</code>" in text
    assert "<script>" not in text


def test_skips_blank_and_duplicate_values():
    text = telegram_html_with_copyable("GE1 GE1", ["", "  ", "GE1", "GE1"])

    assert text == "<code>GE1</code> <code>GE1</code>"


def test_value_that_needs_escaping_is_still_matched():
    text = telegram_html_with_copyable("კოდი: A&B", ["A&B"])

    assert text == "კოდი: <code>A&amp;B</code>"


def test_payload_stays_plain_text_without_copyable_values():
    payload = telegram_text_payload("42", "hello <b>", copyable=[])

    assert payload == {"chat_id": "42", "text": "hello <b>", "disable_web_page_preview": True}


def test_payload_switches_to_html_when_values_present():
    payload = telegram_text_payload("42", "📱 599", copyable=["599"])

    assert payload["parse_mode"] == "HTML"
    assert payload["text"] == "📱 <code>599</code>"


async def test_notify_owner_text_passes_copyable_to_telegram(monkeypatch):
    captured: dict = {}

    async def fake_tg(message: str, copyable=()) -> bool:
        captured["message"] = message
        captured["copyable"] = list(copyable)
        return True

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setattr(notifications, "send_telegram_text", fake_tg)

    assert await notifications.notify_owner_text("📱 599", copyable=["599"]) is True
    assert captured == {"message": "📱 599", "copyable": ["599"]}


async def test_notify_owner_text_whatsapp_fallback_gets_plain_message(monkeypatch):
    captured: list[str] = []

    async def fake_wa(message: str) -> bool:
        captured.append(message)
        return True

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(notifications, "send_whatsapp_text", fake_wa)

    await notifications.notify_owner_text("📱 599", copyable=["599"])
    assert captured == ["📱 599"]


def test_overlapping_values_never_nest_code_tags():
    # The phone number happens to appear inside the account number.
    text = telegram_html_with_copyable(
        "📱 345064300\nGE58TB7085345064300066",
        ["345064300", "GE58TB7085345064300066"],
    )

    assert text == "📱 <code>345064300</code>\n<code>GE58TB7085345064300066</code>"
    assert "<code><code>" not in text
