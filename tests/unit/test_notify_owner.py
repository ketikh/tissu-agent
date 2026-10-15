"""Unit tests for owner notifications in src/tools/support.py.

The owner must be reached through the unified dispatcher (Telegram first,
WhatsApp fallback) — never through WhatsApp directly — and a ticket must be
written before any message is attempted.
"""
from __future__ import annotations

import pytest

import src.notifications as notifications
import src.tools.support as support


class _FakePool:
    def __init__(self) -> None:
        self.executed: list[tuple] = []

    async def execute(self, sql: str, *params) -> None:
        self.executed.append((sql, params))

    async def fetchrow(self, sql: str, *params):
        return None


@pytest.fixture
def owner_inbox(monkeypatch) -> dict:
    """Capture what would be sent to the owner; stub the DB and token creation."""
    inbox: dict = {"texts": [], "images": [], "pool": _FakePool()}

    async def fake_get_db():
        return inbox["pool"]

    async def fake_text(message: str) -> bool:
        inbox["texts"].append(message)
        return True

    async def fake_image(image_bytes: bytes, caption: str, filename: str = "photo.jpg") -> bool:
        inbox["images"].append(caption)
        return True

    async def fake_token(conversation_id: str, action: str) -> str:
        return f"tok-{action}"

    monkeypatch.setattr(support, "get_db", fake_get_db)
    monkeypatch.setattr(notifications, "notify_owner_text", fake_text)
    monkeypatch.setattr(notifications, "notify_owner_image", fake_image)
    monkeypatch.setattr("src.webhooks.whatsapp.create_confirm_token", fake_token)
    monkeypatch.setenv("PUBLIC_URL", "https://example.test")
    return inbox


def _ticket_inserts(pool: _FakePool) -> list[str]:
    return [sql for sql, _ in pool.executed if "INSERT INTO tickets" in sql]


async def test_operator_request_reaches_owner_and_logs_ticket(owner_inbox):
    result = await support.notify_owner(
        reason="კლიენტს ოპერატორთან დაკავშირება სურს",
        conversation_id="instagram_dm_1",
    )

    assert result["notified"] is True
    assert len(owner_inbox["texts"]) == 1
    text = owner_inbox["texts"][0]
    assert "ოპერატორთან" in text
    assert "https://example.test/admin" in text
    assert "ვადასტურებ" not in text          # no confirm links for non-payment reasons
    assert len(_ticket_inserts(owner_inbox["pool"])) == 1


async def test_payment_reason_includes_confirm_and_deny_links(owner_inbox):
    await support.notify_owner(reason="გადახდა შესამოწმებელია", conversation_id="instagram_dm_1")

    text = owner_inbox["texts"][0]
    # build_confirm_url reads PUBLIC_URL at import time, so only assert the path.
    assert "/api/owner-confirm/tok-owner-confirm" in text
    assert "/api/owner-deny/tok-owner-deny" in text


async def test_delivery_failure_is_reported_but_ticket_still_logged(owner_inbox, monkeypatch):
    async def failing(message: str) -> bool:
        return False
    monkeypatch.setattr(notifications, "notify_owner_text", failing)

    result = await support.notify_owner(reason="გაუგებარი: ???", conversation_id="c1")

    assert result["notified"] is False
    assert len(_ticket_inserts(owner_inbox["pool"])) == 1


async def test_sender_exception_does_not_propagate(owner_inbox, monkeypatch):
    async def exploding(message: str) -> bool:
        raise RuntimeError("telegram down")
    monkeypatch.setattr(notifications, "notify_owner_text", exploding)

    result = await support.notify_owner(reason="რჩევა სჭირდება", conversation_id="c1")

    assert result["notified"] is False


@pytest.mark.parametrize("text,expected", [
    ("კლიენტს ტდ3 მოსწონს", "TD3"),
    ("ფოტო fp12 ემთხვევა", "FP12"),
    ("კლიენტს ოპერატორთან დაკავშირება სურს", ""),
    ("", ""),
])
def test_extract_product_code(text, expected):
    assert support._extract_product_code(text) == expected
