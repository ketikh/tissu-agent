"""Unit tests for the public storefront response shape.

We exercise the pure serialization helpers (no DB) so this test runs in
CI without network access. Integration coverage for the HTTP endpoints
lives in tests/integration/test_multitenant.py (extended ad-hoc with
TEST_INTEGRATION=1).
"""
from __future__ import annotations

import pytest

from src.api.storefront import (
    CATEGORY_ALIASES,
    PUBLIC_CATEGORIES,
    _map_category,
    _parse_tags,
    _effective_price,
    _serialize,
)


def test_category_alias_bag_to_pouch():
    assert _map_category("bag") == "pouch"


def test_category_alias_preserves_necklace():
    assert _map_category("necklace") == "necklace"


def test_category_alias_passes_unknown_through():
    # A brand-new category the owner just added — storefront still gets
    # a value instead of blowing up.
    assert _map_category("jewelry-box") == "jewelry-box"


def test_category_alias_none_defaults_to_pouch():
    assert _map_category(None) == "pouch"
    assert _map_category("") == "pouch"


def test_public_categories_cover_task_contract():
    expected = {"pouch", "laptop", "tote", "kidsbackpack", "apron", "necklace"}
    assert expected.issubset(PUBLIC_CATEGORIES)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", []),
        (None, []),
        ("new", ["new"]),
        ("new, sale, featured", ["new", "sale", "featured"]),
        ('["new", "sale"]', ["new", "sale"]),
        (["new", "sale"], ["new", "sale"]),
        ("  new ,  ,   sale  ", ["new", "sale"]),
    ],
)
def test_tag_parsing_handles_every_format(raw, expected):
    assert _parse_tags(raw) == expected


def test_effective_price_uses_sale_price_when_on_sale():
    row = {"price": 74.0, "sale_price": 59.0, "on_sale": True}
    assert _effective_price(row) == 59.0


def test_effective_price_ignores_sale_when_flag_off():
    row = {"price": 74.0, "sale_price": 59.0, "on_sale": False}
    assert _effective_price(row) == 74.0


def test_effective_price_ignores_zero_or_negative_sale_price():
    row = {"price": 74.0, "sale_price": 0, "on_sale": True}
    assert _effective_price(row) == 74.0


def test_serialize_matches_task_contract():
    """The Tissu website depends on every one of these keys verbatim."""
    row = {
        "id": 31,
        "code": "FP1",
        "product_name": "Tissu without strap #1",
        "model": "ფხრიწიანი",
        "size": "პატარა (33x25)",
        "color": "",
        "description": "Handmade canvas pouch",
        "price": 69.0,
        "stock": 1,
        "image_url": "https://cdn/front.jpg",
        "image_url_back": "https://cdn/back.jpg",
        "category": "bag",
        "tags": "new",
        "on_sale": False,
        "sale_price": None,
        "updated_at": "2026-04-23T12:34:56Z",
        "created_at": "2026-04-01T00:00:00Z",
    }

    out = _serialize(row)

    assert out == {
        "id": "31",
        "code": "FP1",
        "name": "Tissu without strap #1",
        "model": "ფხრიწიანი",
        "size": "პატარა (33x25)",
        "color": "",
        "description": "Handmade canvas pouch",
        "price": 69.0,
        "original_price": None,
        "currency": "GEL",
        "stock": 1,
        "in_stock": True,
        "image_front": "https://cdn/front.jpg",
        "image_back": "https://cdn/back.jpg",
        "category": "pouch",
        "tags": ["new"],
        "updated_at": "2026-04-23T12:34:56Z",
    }


def test_serialize_exposes_original_price_on_sale():
    row = {
        "id": 42, "code": "FD2", "product_name": "Big",
        "model": "", "size": "", "color": "",
        "description": None,
        "price": 74.0, "stock": 2,
        "image_url": "", "image_url_back": "",
        "category": "bag", "tags": "",
        "on_sale": True, "sale_price": 59.0,
        "updated_at": "2026-04-23T12:34:56Z", "created_at": "",
    }
    out = _serialize(row)
    assert out["price"] == 59.0
    assert out["original_price"] == 74.0


def test_serialize_normalizes_empty_description_to_none():
    row = {
        "id": 1, "code": "", "product_name": "",
        "model": "", "size": "", "color": "",
        "description": "   ",
        "price": 0, "stock": 0, "image_url": "", "image_url_back": "",
        "category": "bag", "tags": "", "on_sale": False, "sale_price": None,
        "updated_at": "", "created_at": "",
    }
    assert _serialize(row)["description"] is None


def test_serialize_marks_zero_stock_as_out():
    row = {
        "id": 99,
        "code": "XX1",
        "product_name": "sold out",
        "model": "",
        "size": "",
        "color": "",
        "price": 10.0,
        "stock": 0,
        "image_url": "",
        "image_url_back": "",
        "category": "bag",
        "tags": "",
        "on_sale": False,
        "sale_price": None,
        "updated_at": "",
        "created_at": "",
    }
    out = _serialize(row)
    assert out["stock"] == 0
    assert out["in_stock"] is False


def test_serialize_falls_back_to_created_at_when_updated_at_missing():
    row = {
        "id": 1, "code": "", "product_name": "", "model": "", "size": "",
        "color": "", "price": 0, "stock": 0, "image_url": "", "image_url_back": "",
        "category": "bag", "tags": "", "on_sale": False, "sale_price": None,
        "updated_at": None, "created_at": "2026-04-01T00:00:00Z",
    }
    assert _serialize(row)["updated_at"] == "2026-04-01T00:00:00Z"
