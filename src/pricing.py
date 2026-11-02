"""Category-dependent pricing rules.

Laptop cases are the only products with a fixed price per size: the owner
picks პატარა or დიდი in the admin panel and the price follows automatically
(69₾ / 74₾). Every other category (cosmetic bags, aprons, …) has a size
that is purely descriptive, so changing it must leave the owner-entered
price untouched.
"""
from __future__ import annotations

from typing import Optional

# 'bag' is the legacy slug, 'laptop-cases' the renamed one. Rows with no
# category at all are legacy laptop cases too (the admin panel applies the
# same `i.category || 'laptop-cases'` fallback).
LAPTOP_CATEGORY_SLUGS: frozenset[str] = frozenset({"bag", "laptop-cases"})
LAPTOP_PRICE_SMALL: float = 69.0
LAPTOP_PRICE_LARGE: float = 74.0


def is_laptop_category(slug: Optional[str]) -> bool:
    """True for laptop-case rows, including legacy rows with no category."""
    return (slug or "bag") in LAPTOP_CATEGORY_SLUGS


def fixed_price_for_size(category: Optional[str], size: str) -> Optional[float]:
    """Return the price a size change forces, or None to keep the current one.

    Only laptop cases have a fixed price per size. A size mentioning დიდი is
    the large case; anything else is the small one — matching what the admin
    dropdown stores ("პატარა (33x25)" / "დიდი (37x27)").
    """
    if not is_laptop_category(category):
        return None
    return LAPTOP_PRICE_LARGE if "დიდი" in size else LAPTOP_PRICE_SMALL
