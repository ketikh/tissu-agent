"""Local smoke test for the link → photo pipeline.

Usage:
    python scripts/test_link_match.py <url>

Runs the same pipeline the Facebook webhook does for customer-pasted links:
    _extract_og_image(url) → crop_to_main_bag → analyze_and_match

Prints every stage so you can see exactly where a failure happens.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.image_match import crop_to_main_bag  # noqa: E402
from src.vision_match import analyze_and_match  # noqa: E402
from src.webhooks.facebook import _extract_og_image  # noqa: E402


async def main(url: str) -> None:
    print(f"── INPUT ──")
    print(f"url: {url}")

    print(f"\n── STAGE 1: _extract_og_image ──")
    img_bytes = await _extract_og_image(url)
    if not img_bytes:
        print("❌ No image could be extracted from the URL.")
        sys.exit(1)
    raw_path = Path("test_photos/_link_raw.jpg")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(img_bytes)
    print(f"downloaded: {len(img_bytes)} bytes")
    print(f"saved to: {raw_path}")

    print(f"\n── STAGE 2: crop_to_main_bag ──")
    cropped = await crop_to_main_bag(img_bytes)
    cropped_path = Path("test_photos/_link_cropped.jpg")
    cropped_path.write_bytes(cropped)
    shrink = (1 - len(cropped) / len(img_bytes)) * 100
    print(f"cropped: {len(cropped)} bytes ({shrink:+.1f}% vs raw)")
    print(f"saved to: {cropped_path}")

    print(f"\n── STAGE 3: analyze_and_match ──")
    result = await analyze_and_match(img_bytes)

    print(f"\n── RESULT ──")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/test_link_match.py <url>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
