"""Tests for src.db.list_angle_photos_for_ids — the batched lookup of the
extra "angle" photos the owner adds from the admin card's "სხვა რაკურსი"
tile. The storefront relies on it to show those photos on the website."""
from __future__ import annotations

import pytest

from src import db


class _FakePool:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.queries: list[tuple[str, tuple]] = []

    async def fetch(self, query: str, *params):
        self.queries.append((query, params))
        return self._rows


@pytest.fixture
def fake_pool(monkeypatch):
    def install(rows: list[dict]) -> _FakePool:
        pool = _FakePool(rows)

        async def get_db():
            return pool

        monkeypatch.setattr(db, "get_db", get_db)
        return pool

    return install


async def test_empty_id_list_skips_the_database(fake_pool):
    pool = fake_pool([])

    assert await db.list_angle_photos_for_ids("tissu", []) == {}
    assert pool.queries == []


async def test_groups_urls_by_product_in_upload_order(fake_pool):
    pool = fake_pool([
        {"inventory_id": 171, "image_url": "https://cdn/a1.jpg"},
        {"inventory_id": 171, "image_url": "https://cdn/a2.jpg"},
        {"inventory_id": 172, "image_url": "https://cdn/b1.jpg"},
        {"inventory_id": 173, "image_url": "   "},
    ])

    out = await db.list_angle_photos_for_ids("tissu", [171, 172, 173])

    assert out == {
        171: ["https://cdn/a1.jpg", "https://cdn/a2.jpg"],
        172: ["https://cdn/b1.jpg"],
    }
    query, params = pool.queries[0]
    assert "product_extra_photos" in query
    assert "photo_type = 'angle'" in query
    assert params == ("tissu", [171, 172, 173])
