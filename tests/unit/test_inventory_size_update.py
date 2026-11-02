"""Regression tests for PUT /api/inventory/{id}?size=…

The admin panel's size dropdown used to re-price *every* product to the
laptop-case prices (69₾ / 74₾) whenever a size was saved. Cosmetic bags
and other categories must keep the price the owner entered; laptop cases
keep their fixed per-size price.
"""
from __future__ import annotations

import pytest

import server


class _FakePool:
    """Records SQL issued by update_inventory; answers the category lookup."""

    def __init__(self, category: str | None) -> None:
        self._category = category
        self.executed: list[tuple[str, tuple]] = []

    async def fetchval(self, query: str, *params):
        assert "SELECT category FROM inventory" in query
        return self._category

    async def execute(self, query: str, *params) -> None:
        self.executed.append((query, params))


@pytest.fixture
def fake_pool(monkeypatch):
    holder: dict[str, _FakePool] = {}

    def install(category: str | None) -> _FakePool:
        pool = _FakePool(category)
        holder["pool"] = pool

        async def get_db():
            return pool

        monkeypatch.setattr(server, "get_db", get_db)
        return pool

    return install


async def test_cosmetic_bag_size_change_leaves_price_alone(fake_pool):
    pool = fake_pool("cosmetic-bag")

    await server.update_inventory(item_id=172, size="პატარა", tenant_id="default")

    assert len(pool.executed) == 1
    query, params = pool.executed[0]
    assert "SET size = $1, updated_at = $2" in query
    assert "price" not in query
    assert params[0] == "პატარა"
    assert params[2:] == (172, "default")


async def test_laptop_case_size_change_snaps_price(fake_pool):
    pool = fake_pool("laptop-cases")

    await server.update_inventory(item_id=7, size="დიდი (37x27)", tenant_id="default")

    assert len(pool.executed) == 1
    query, params = pool.executed[0]
    assert "SET size = $1, price = $2" in query
    assert params[0] == "დიდი (37x27)"
    assert params[1] == 74.0


async def test_legacy_row_without_category_is_treated_as_laptop(fake_pool):
    pool = fake_pool(None)

    await server.update_inventory(item_id=3, size="პატარა (33x25)", tenant_id="default")

    query, params = pool.executed[0]
    assert "price = $2" in query
    assert params[1] == 69.0
