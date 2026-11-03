"""The new-order alert the owner gets (create_order in src/tools/support.py)
must carry the shop's bank accounts and mark them, plus the customer's
phone, as tap-to-copy — the owner forwards the account number to the
customer herself."""
from __future__ import annotations

import pytest

import src.notifications as notifications
import src.tools.support as support


class _FakePool:
    async def fetchrow(self, sql: str, *params):
        assert "INSERT INTO orders" in sql
        return {"id": 77}


@pytest.fixture
def order_env(monkeypatch) -> dict:
    sent: dict = {}

    async def fake_get_db():
        return _FakePool()

    async def fake_text(message: str, copyable=()) -> bool:
        sent["text"] = message
        sent["copyable"] = list(copyable)
        return True

    monkeypatch.setattr(support, "get_db", fake_get_db)
    monkeypatch.setattr(notifications, "notify_owner_text", fake_text)
    monkeypatch.setenv("PUBLIC_URL", "https://example.test")
    return sent


def _with_config(monkeypatch, cfg):
    async def fake_cfg(tenant_id: str):
        return cfg
    monkeypatch.setattr(support, "get_bot_config", fake_cfg)


async def test_alert_lists_bank_accounts_and_marks_them_copyable(order_env, monkeypatch):
    _with_config(monkeypatch, {"payment_accounts": [
        {"bank_name": "თიბისი", "account_number": "GE58TB7085345064300066"},
        {"bank_name": "საქართველოს ბანკი", "account_number": "GE65BG0000000358364200"},
    ]})

    result = await support.create_order(
        "ნინო", "599123456", "თბილისი, რუსთაველის 1", "TD1 x1", 80.0,
        payment_method="თიბისი",
    )

    assert result["success"] is True
    text = order_env["text"]
    assert "🛒 ახალი შეკვეთა #77!" in text
    assert "💳 თიბისი" in text
    assert "🏦 ანგარიშები" in text
    assert "თიბისი: GE58TB7085345064300066" in text
    assert "საქართველოს ბანკი: GE65BG0000000358364200" in text
    assert text.endswith("📋 ადმინ პანელი:\nhttps://example.test/admin")
    assert order_env["copyable"] == [
        "599123456", "GE58TB7085345064300066", "GE65BG0000000358364200",
    ]


async def test_alert_without_configured_accounts_still_goes_out(order_env, monkeypatch):
    _with_config(monkeypatch, None)

    await support.create_order("ნინო", "599123456", "თბილისი", "FP2 x1", 75.0)

    assert "🏦" not in order_env["text"]
    assert "💳" not in order_env["text"]
    assert order_env["copyable"] == ["599123456"]


async def test_config_lookup_failure_does_not_block_the_alert(order_env, monkeypatch):
    async def exploding(tenant_id: str):
        raise RuntimeError("db down")
    monkeypatch.setattr(support, "get_bot_config", exploding)

    await support.create_order("ნინო", "599123456", "თბილისი", "FP2 x1", 75.0)

    assert "ახალი შეკვეთა #77" in order_env["text"]
    assert order_env["copyable"] == ["599123456"]


async def test_blank_or_malformed_accounts_are_ignored(order_env, monkeypatch):
    _with_config(monkeypatch, {"payment_accounts": [
        {"bank_name": "ბანკი", "account_number": "   "},
        "garbage",
        {"account_number": "GE00XX0000000000000000"},
    ]})

    await support.create_order("ნინო", "599123456", "თბილისი", "FP2 x1", 75.0)

    text = order_env["text"]
    assert "ბანკი:" not in text
    assert "\nGE00XX0000000000000000" in text
    assert order_env["copyable"] == ["599123456", "GE00XX0000000000000000"]
