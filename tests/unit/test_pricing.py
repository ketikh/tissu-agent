"""Unit tests for category-dependent pricing rules.

Regression for the admin panel's size dropdown: changing the size of a
laptop case snaps the price to 69₾ / 74₾, but cosmetic bags (and every
other category) must keep the price the owner typed.
"""
from __future__ import annotations

import pytest

from src.pricing import (
    LAPTOP_PRICE_LARGE,
    LAPTOP_PRICE_SMALL,
    fixed_price_for_size,
    is_laptop_category,
)


@pytest.mark.parametrize("slug", ["bag", "laptop-cases", None, ""])
def test_laptop_slugs_including_legacy_blank(slug):
    assert is_laptop_category(slug) is True


@pytest.mark.parametrize("slug", ["cosmetic-bag", "necklace", "apron", "tote-bags"])
def test_other_categories_are_not_laptop(slug):
    assert is_laptop_category(slug) is False


@pytest.mark.parametrize(
    "size, expected",
    [
        ("პატარა (33x25)", LAPTOP_PRICE_SMALL),
        ("დიდი (37x27)", LAPTOP_PRICE_LARGE),
        ("დიდი", LAPTOP_PRICE_LARGE),
        ("", LAPTOP_PRICE_SMALL),
    ],
)
def test_laptop_size_forces_fixed_price(size, expected):
    assert fixed_price_for_size("laptop-cases", size) == expected
    assert fixed_price_for_size("bag", size) == expected


@pytest.mark.parametrize("size", ["პატარა", "დიდი", "დიდი ზომა", ""])
def test_cosmetic_bag_size_keeps_owner_price(size):
    # 25₾ / 35₾ cosmetic bags must never be re-priced to 69 / 74.
    assert fixed_price_for_size("cosmetic-bag", size) is None
