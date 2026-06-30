"""Local smoke test for the photo-matching pipeline.

Usage:
    python scripts/test_photo_match.py <path-to-image>

Runs the same pipeline the Facebook webhook does:
    crop_to_main_bag → analyze_and_match

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


async def main(image_path: str) -> None:
    p = Path(image_path)
    if not p.exists():
        print(f"❌ File not found: {p}")
        sys.exit(1)

    original = p.read_bytes()
    print(f"── INPUT ──")
    print(f"path: {p}")
    print(f"size: {len(original)} bytes")

    print(f"\n── STAGE 1: crop_to_main_bag ──")
    cropped = await crop_to_main_bag(original)
    cropped_path = p.with_name(p.stem + ".cropped.jpg")
    cropped_path.write_bytes(cropped)
    shrink = (1 - len(cropped) / len(original)) * 100
    print(f"cropped bytes: {len(cropped)} ({shrink:+.1f}% vs original)")
    print(f"saved to: {cropped_path}")

    print(f"\n── STAGE 2: analyze_and_match ──")
    result = await analyze_and_match(original)

    print(f"\n── RESULT ──")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/test_photo_match.py <path-to-image>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
