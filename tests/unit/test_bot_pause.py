"""Unit tests for the owner's bot on/off switch (bot_paused).

The Meta webhook must ignore every incoming Messenger / Instagram event
while the owner has switched the bot off in the admin panel, and must
keep working (fail open) if the pause lookup itself errors out.
"""
from __future__ import annotations

import asyncio

import pytest

import src.webhooks.facebook as fb


class _FakeRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


def _instagram_text_event(text: str = "გამარჯობა") -> dict:
    return {
        "object": "instagram",
        "entry": [{
            "id": fb.IG_PAGE_ID,
            "messaging": [{
                "sender": {"id": "customer-123"},
                "message": {"mid": "mid-1", "text": text},
            }],
        }],
    }


@pytest.fixture
def processed(monkeypatch) -> list[dict]:
    """Stub the DB / Meta calls around webhook_receive and record every
    message that reaches the agent pipeline."""
    seen: list[dict] = []

    async def fake_resolve(page_id: str):
        return (fb.DEFAULT_TENANT_ID, "token")

    async def fake_process(sender_id, text, conversation_id, channel,
                           customer_name="", image_url=""):
        seen.append({"sender_id": sender_id, "text": text, "channel": channel})

    monkeypatch.setattr(fb, "_resolve_tenant_context", fake_resolve)
    monkeypatch.setattr(fb, "_process_message", fake_process)
    monkeypatch.setattr(fb, "FB_PAGE_TOKEN", "")   # skip the profile lookup
    monkeypatch.setattr(fb, "_processed_mids", {})
    return seen


async def _receive(body: dict) -> dict:
    result = await fb.webhook_receive(_FakeRequest(body))
    await asyncio.sleep(0)  # let fire-and-forget tasks run
    return result


async def test_paused_bot_ignores_customer_messages(processed, monkeypatch):
    async def paused(tenant_id: str) -> bool:
        return True
    monkeypatch.setattr(fb, "is_bot_paused", paused)

    result = await _receive(_instagram_text_event())

    assert result == {"status": "ok"}   # Meta must still get a 200
    assert processed == []


async def test_running_bot_processes_customer_messages(processed, monkeypatch):
    async def not_paused(tenant_id: str) -> bool:
        return False
    monkeypatch.setattr(fb, "is_bot_paused", not_paused)

    await _receive(_instagram_text_event("ფასი?"))

    assert len(processed) == 1
    assert processed[0]["text"] == "ფასი?"
    assert processed[0]["channel"] == "instagram_dm"


async def test_pause_lookup_failure_fails_open(processed, monkeypatch):
    async def broken(tenant_id: str) -> bool:
        raise RuntimeError("db down")
    monkeypatch.setattr(fb, "is_bot_paused", broken)

    await _receive(_instagram_text_event())

    assert len(processed) == 1
