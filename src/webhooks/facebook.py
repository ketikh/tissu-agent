"""Facebook Messenger / Instagram DM webhook handler."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import traceback as _tb
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

from src.agents.support_sales import get_support_sales_agent
from src.db import (
    get_db, DEFAULT_TENANT_ID,
    get_tenant, get_tenant_by_fb_page_id,
)
from src.engine import run_agent
from src.notifications import send_whatsapp_image, send_whatsapp_text
from src.secrets_vault import decrypt_secret
from src.tools.support import _pending_photos, _ai_hints
from src.vision import download_image, is_payment_receipt
from src.webhooks.whatsapp import build_confirm_url, create_confirm_token

logger = logging.getLogger(__name__)

router = APIRouter()

# Legacy single-tenant fallbacks — used for the current Tissu
# deployment until (and unless) the super-admin fills in
# fb_page_id / fb_page_token for a tenant. Multi-tenant onboarding
# stores per-tenant credentials in the tenants table; the webhook
# dispatcher below prefers those when available.
VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"


async def _resolve_tenant_context(page_id: str) -> tuple[str, str]:
    """Given a Meta webhook entry's page_id, return the tenant_id and
    page access token to use for replies. Falls back to the legacy
    env var token + DEFAULT_TENANT_ID when the page is the original
    Tissu page (so nothing breaks for the existing single-shop
    deployment). Returns ('', '') when the page is unknown — the
    caller should ignore unknown-page events."""
    if not page_id:
        return (DEFAULT_TENANT_ID, FB_PAGE_TOKEN)
    try:
        tenant = await get_tenant_by_fb_page_id(page_id)
    except Exception:
        tenant = None
    if tenant and tenant.get("fb_page_token_encrypted"):
        token = decrypt_secret(tenant["fb_page_token_encrypted"])
        if token:
            return (tenant["tenant_id"], token)
    # Legacy Tissu page: use env var token.
    if page_id == PAGE_ID:
        return (DEFAULT_TENANT_ID, FB_PAGE_TOKEN)
    # Unknown page — log but don't crash. The webhook handler can
    # return 200 so Meta doesn't retry, and the event is dropped.
    logger.warning(
        "[webhook] unknown fb page_id=%s — event dropped", page_id,
    )
    return ("", "")
# Instagram Business Account ID — used to filter self-echoes on IG DMs.
# Set this in Railway env after connecting IG to the Facebook Page.
IG_USER_ID = os.getenv("IG_USER_ID", "")

_processed_mids: dict[str, float] = {}

# Track which inventory categories were already sent to each conversation
# Key: conversation_id, Value: set of "model|size" combos already sent
_sent_inventory: dict[str, set[str]] = {}

# Track individual product codes whose photos have been sent to each conversation.
# Avoids re-sending a photo the customer already saw — when they ask
# "დიდიც გაქვთ?" after seeing the small size, bot confirms in text only.
_sent_codes: dict[str, set[str]] = {}

# Buffers for text↔photo pairing
_pending_text: dict[str, dict] = {}   # sender_id -> {text, mid, ...} — text waiting for photo
_pending_photo: dict[str, dict] = {}  # sender_id -> {image_url, mid, ...} — photo waiting for text

# When AI fails to match a photo we ask the customer for size first; we keep the
# bytes here and only forward to WhatsApp once we know which size to send.
_pending_failed_photos: dict[str, bytes] = {}

# When AI matches but the product is sold out, we ask the customer for size first
# and only then announce it's out of stock for that size.
_pending_soldout: dict[str, str] = {}  # conversation_id -> matched code

_PHOTO_HINT = re.compile(r'(გაქვთ|მოდელი|ჩანთა|ასეთი|მსგავსი|ეს\s*არის|have|this)', re.IGNORECASE)

# Longer buffer so the customer has time to attach a photo after typing
# "ესეც" / "ეს გაქვთ?" etc. 3s was often too fast for humans.
BUFFER_SECONDS = 6


def _bg_wa_image(image_bytes: bytes, caption: str, filename: str = "photo.jpg") -> None:
    """Fire-and-forget WhatsApp image send. Owner notifications must NEVER
    block the customer's reply — a slow or failing WA call (e.g. 24h window
    closed) previously stalled FB/IG responses entirely."""
    async def _run():
        try:
            await asyncio.wait_for(
                send_whatsapp_image(image_bytes, caption=caption, filename=filename),
                timeout=20,
            )
        except asyncio.TimeoutError:
            print("[WA] image send timed out (20s) — continuing", flush=True)
        except Exception as e:
            print(f"[WA] image send error: {e}", flush=True)
    asyncio.create_task(_run())


def _bg_wa_text(message: str) -> None:
    """Fire-and-forget WhatsApp text send."""
    async def _run():
        try:
            await asyncio.wait_for(send_whatsapp_text(message), timeout=10)
        except asyncio.TimeoutError:
            print("[WA] text send timed out (10s) — continuing", flush=True)
        except Exception as e:
            print(f"[WA] text send error: {e}", flush=True)
    asyncio.create_task(_run())


async def _fetch_fb_photo_via_graph(url: str) -> bytes:
    """If the URL is a Facebook photo page (has fbid=), use Graph API to fetch
    the image directly — normal HTTP fetch returns 400 without a session."""
    if not FB_PAGE_TOKEN:
        return b""
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    if "facebook.com" not in parsed.netloc:
        return b""
    qs = parse_qs(parsed.query)
    fbid = (qs.get("fbid") or qs.get("story_fbid") or [""])[0]
    if not fbid:
        # Try path like /photo/123456
        parts = [p for p in parsed.path.split("/") if p]
        for p in parts:
            if p.isdigit() and len(p) > 10:
                fbid = p
                break
    if not fbid:
        return b""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            meta = await client.get(
                f"https://graph.facebook.com/v21.0/{fbid}",
                params={"fields": "images", "access_token": FB_PAGE_TOKEN},
            )
            data = meta.json()
            images = data.get("images", [])
            if not images:
                print(f"[LINK] Graph fbid={fbid} no images: {data}", flush=True)
                return b""
            # Pick the largest (first entry is usually the largest)
            img_url = images[0].get("source", "")
            if not img_url:
                return b""
            img = await client.get(img_url)
            if img.status_code == 200 and len(img.content) > 500:
                print(f"[LINK] Graph API fetched fbid={fbid} ({len(img.content)} bytes)", flush=True)
                return img.content
    except Exception as e:
        print(f"[LINK] Graph API fetch error: {e}", flush=True)
    return b""


async def _extract_og_image(url: str) -> bytes:
    """Fetch the URL and pull the product photo from its OpenGraph / Twitter
    card meta tags. Returns the image bytes, or b"" if nothing usable found.
    Facebook photo pages are routed through the Graph API because the public
    HTML returns 400 to anonymous clients."""
    # Facebook photo page → Graph API shortcut
    if "facebook.com/photo" in url or ("facebook.com" in url and "fbid=" in url):
        fb_bytes = await _fetch_fb_photo_via_graph(url)
        if fb_bytes:
            return fb_bytes
        # fall through to generic og:image attempt below

    headers = {
        # Mimic a real browser — some sites (Instagram, Facebook) refuse bot UAs.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ka;q=0.8",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                print(f"[LINK] URL fetch {resp.status_code}: {url[:80]}", flush=True)
                return b""
            html = resp.text
            # Look for og:image, og:image:secure_url, or twitter:image
            patterns = (
                r'<meta[^>]+property=["\']og:image:secure_url["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            )
            img_url = ""
            for pat in patterns:
                m = re.search(pat, html, re.IGNORECASE)
                if m:
                    img_url = m.group(1)
                    break
            if not img_url:
                print(f"[LINK] No og:image meta tag in HTML from {url[:80]}", flush=True)
                return b""
            # Resolve relative URLs
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            elif img_url.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(url)
                img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
            # Decode HTML entities that can appear in meta content
            img_url = img_url.replace("&amp;", "&")

            img_resp = await client.get(img_url)
            if img_resp.status_code == 200 and len(img_resp.content) > 500:
                print(f"[LINK] og:image fetched ({len(img_resp.content)} bytes) from {url[:80]}", flush=True)
                # Instagram frequently returns a 273×273 thumbnail when the
                # post is fetched anonymously. Upscale tiny images so CLIP /
                # Gemini have enough canvas to match the pattern against the
                # catalog (which is full-size).
                from src.image_match import upscale_if_small
                return upscale_if_small(img_resp.content)
            print(f"[LINK] og:image HTTP {img_resp.status_code}, {len(img_resp.content)}B from {img_url[:80]}", flush=True)
            return b""
    except Exception as e:
        print(f"[LINK] og:image error: {e}", flush=True)
        return b""


def _get_delivery_day() -> str:
    """Tbilisi-local delivery day text. Saturday→ორშაბათს, Sunday→ხვალ (=Monday), else→ხვალ."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Tbilisi"))
    except Exception:
        now = datetime.now(timezone.utc)
    weekday = now.weekday()  # Mon=0 ... Sat=5, Sun=6
    if weekday == 5:
        return "ორშაბათს"
    return "ხვალ"


def _cleanup_old_mids() -> None:
    now = time.time()
    for key in list(_processed_mids):
        if now - _processed_mids[key] > 300:
            del _processed_mids[key]


async def _send_typing_on(sender_id: str) -> None:
    if not FB_PAGE_TOKEN:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                "https://graph.facebook.com/v21.0/me/messages",
                params={"access_token": FB_PAGE_TOKEN},
                json={"recipient": {"id": sender_id}, "sender_action": "typing_on"},
            )
    except Exception:
        pass


async def _process_message(
    sender_id: str, text: str, conversation_id: str,
    channel: str, customer_name: str = "", image_url: str = "",
) -> None:
    print(f"[MSG] START: sender={sender_id}, text={text[:50]}..., image={'YES' if image_url else 'NO'}", flush=True)
    try:
        await _send_typing_on(sender_id)

        # ── Photo handling ──
        if image_url:
            # A new photo replaces whatever old photo was pending size confirmation
            _pending_failed_photos.pop(conversation_id, None)
            _pending_soldout.pop(conversation_id, None)
            image_bytes = await download_image(image_url)
            if not image_bytes:
                text = "[კლიენტმა ფოტო გამოგზავნა მაგრამ ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨']"
            else:
                is_receipt = await is_payment_receipt(image_bytes, conversation_id)
                if is_receipt:
                    cname = customer_name or "კლიენტი"
                    confirm_url = build_confirm_url("owner-confirm", await create_confirm_token(conversation_id, "owner-confirm"))
                    deny_url = build_confirm_url("owner-deny", await create_confirm_token(conversation_id, "owner-deny"))
                    _bg_wa_image(
                        image_bytes,
                        caption=f"📷 {cname} — გადახდის ქვითარი\n\n✅ ვადასტურებ:\n{confirm_url}\n\n❌ არ ვადასტურებ:\n{deny_url}",
                        filename="receipt.jpg",
                    )
                    text = "[კლიენტმა გადახდის სქრინი გამოგზავნა. უთხარი 'მადლობა, გადავამოწმებ ✨' და ᲒᲐᲩᲔᲠᲓᲘ. მისამართს ᲐᲠ ეკითხო.]"
                else:
                    _pending_photos[conversation_id] = image_bytes
                    await _send_typing_on(sender_id)

                    # Log every step to database for debugging
                    _debug_pool = await get_db()
                    _log_counter = [0]
                    async def _log(step: str):
                        _log_counter[0] += 1
                        try:
                            await _debug_pool.execute(
                                "INSERT INTO ai_photo_hints (conversation_id, code, model, size, price, image_url, score, created_at) VALUES ($1, $2, '', '', 0, '', 0, $3)",
                                f"debug_{_log_counter[0]}_{conversation_id[:30]}", step, datetime.now(timezone.utc).isoformat(),
                            )
                        except Exception:
                            pass

                    await _log(f"step1_photo_received_{len(image_bytes)}_bytes")

                    # Run AI match INLINE
                    ai_code = ""
                    ai_product = {}
                    ai_sold_out = False
                    try:
                        await _log("step2_importing_vision_match")
                        from src.vision_match import analyze_and_match
                        await _log("step3_calling_analyze_and_match")
                        match_result = await asyncio.wait_for(analyze_and_match(image_bytes), timeout=120)
                        await _log(f"step4_result_matched={match_result.get('matched')}_code={match_result.get('code')}_score={match_result.get('score')}_sold_out={match_result.get('sold_out')}")
                        if match_result and match_result.get("matched"):
                            ai_code = match_result["code"]
                            ai_product = match_result.get("product", {})
                            ai_sold_out = bool(match_result.get("sold_out"))
                            await _log(f"step5_SUCCESS_{ai_code}_sold_out={ai_sold_out}")
                        else:
                            await _log(f"step5_NO_MATCH_score={match_result.get('score','?')}_closest={match_result.get('closest_code','?')}_msg={match_result.get('message','?')}")
                    except asyncio.TimeoutError:
                        await _log("step_ERROR_TIMEOUT_60s")
                    except Exception as e:
                        await _log(f"step_ERROR_{type(e).__name__}_{str(e)[:100]}")

                    # Reuse the customer's most-recent size choice (within 60 min)
                    # — used for BOTH AI-match and AI-fail branches so we don't
                    # ask the customer the same question twice in one session.
                    prev_size = ""
                    try:
                        prev_msgs = await _debug_pool.fetch(
                            "SELECT content, created_at FROM messages "
                            "WHERE conversation_id = $1 AND role = 'user' "
                            "ORDER BY created_at DESC LIMIT 30",
                            conversation_id,
                        )
                        now_utc = datetime.now(timezone.utc)
                        for pm in prev_msgs:
                            created = pm["created_at"] if pm and "created_at" in pm else None
                            age_min = 0.0
                            try:
                                if isinstance(created, str):
                                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                                elif hasattr(created, "tzinfo"):
                                    created_dt = created
                                else:
                                    created_dt = None
                                if created_dt is not None:
                                    if created_dt.tzinfo is None:
                                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                                    age_min = (now_utc - created_dt).total_seconds() / 60
                            except Exception:
                                age_min = 0.0
                            if age_min > 60:
                                continue
                            c = (pm["content"] or "").lower()
                            if any(w in c for w in ("პატარა", "პატარ", "patara", "small")):
                                prev_size = "პატარა"
                                break
                            elif any(w in c for w in ("დიდი", "დიდ", "didi", "large")):
                                prev_size = "დიდი"
                                break
                        await _log(f"step5a_prev_size={prev_size or 'NONE'}")
                    except Exception as e:
                        await _log(f"step5a_error={str(e)[:80]}")

                    if ai_code:
                        # Get ALL linked models from product_pairs (transitive: A↔B, B↔C → A,B,C)
                        all_codes = set([ai_code])
                        try:
                            # BFS to find all connected codes
                            to_check = [ai_code]
                            while to_check:
                                current = to_check.pop(0)
                                pair_rows = await _debug_pool.fetch(
                                    "SELECT code_b FROM product_pairs WHERE code_a = $1", current
                                )
                                for pr in pair_rows:
                                    if pr["code_b"] not in all_codes:
                                        all_codes.add(pr["code_b"])
                                        to_check.append(pr["code_b"])
                        except Exception:
                            pass
                        all_codes = sorted(all_codes)
                        codes_str = ",".join(all_codes)
                        await _log(f"step5b_linked_codes={codes_str}")
                        try:
                            now_iso = datetime.now(timezone.utc).isoformat()
                            await _debug_pool.execute(
                                """INSERT INTO ai_photo_hints (conversation_id, code, model, size, price, image_url, score, created_at)
                                   VALUES ($1, $2, '', '', 0, '', $3, $4)
                                   ON CONFLICT (conversation_id) DO UPDATE SET code=$2, score=$3, created_at=$4""",
                                conversation_id, codes_str, float(match_result.get("score", 0)), now_iso,
                            )
                            await _log(f"step6_saved_codes={codes_str}")
                        except Exception as e:
                            await _log(f"step6_save_error={e}")

                        if ai_sold_out:
                            # Matched, but sold out. Size must be confirmed BEFORE
                            # we announce "sold out" — user wants size clarified on
                            # every outcome, incl. sold-out and owner-forward cases.
                            if prev_size:
                                _pending_photos.pop(conversation_id, None)
                                await _log(f"step6_soldout_with_prev_size={prev_size}_{ai_code}")
                                text = (
                                    f"[AI-მ იპოვა მოდელი მაგრამ {prev_size} ზომაში მარაგი ამოწურულია. ზუსტად ეს უპასუხე: "
                                    f"'სამწუხაროდ {prev_size} ზომაში მარაგი ამოწურულია ✨']"
                                )
                            else:
                                # Queue sold-out announcement until size is known.
                                _pending_soldout[conversation_id] = ai_code
                                _pending_photos.pop(conversation_id, None)
                                await _log(f"step6_soldout_awaiting_size_{ai_code}")
                                text = (
                                    "[AI-მ იპოვა მაგრამ მარაგი ამოწურულია. ჯერ ზომა ვკითხოთ. ზუსტად ეს უპასუხე: "
                                    "'რა ზომა გაინტერესებთ? ✨\\n\\n"
                                    "📦 პატარა (33x25სმ) — 69₾\\n"
                                    "📦 დიდი (37x27სმ) — 74₾']"
                                )
                        elif prev_size:
                            # We know size from before — short question
                            text = (
                                f"[კლიენტმა ახალი ფოტო გამოგზავნა. ადრე {prev_size} ზომა აირჩია. "
                                f"ზუსტად ეს უპასუხე: 'ესეც {prev_size} ზომაში გადაგიმოწმოთ? ✨']"
                            )
                        else:
                            # First time — ask size with prices
                            text = (
                                f"[კლიენტმა ფოტო გამოგზავნა. ზუსტად ეს უპასუხე: "
                                f"'რა ზომაში გადაგიმოწმოთ? ✨\\n\\n"
                                f"📦 პატარა (33x25სმ) — 69₾\\n"
                                f"📦 დიდი (37x27სმ) — 74₾']"
                            )
                    else:
                        # AI failed — clear any stale hints from earlier photos so next
                        # size answer doesn't match against the previous photo's codes.
                        try:
                            await _debug_pool.execute(
                                "DELETE FROM ai_photo_hints WHERE conversation_id = $1",
                                conversation_id,
                            )
                        except Exception:
                            pass
                        photo_bytes = _pending_photos.get(conversation_id)
                        if prev_size and photo_bytes:
                            # Size already known from this session — forward the photo
                            # to the owner right away with size context, don't ask again.
                            try:
                                public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
                                admin_url = f"{public_url}/admin"
                                confirm_url = build_confirm_url("photo-confirm", await create_confirm_token(conversation_id, "photo-confirm"))
                                deny_url = build_confirm_url("photo-deny", await create_confirm_token(conversation_id, "photo-deny"))
                                _bg_wa_image(
                                    photo_bytes,
                                    caption=(
                                        f"📷 კლიენტი ეძებს ამ მოდელს.\n"
                                        f"📐 ზომა: {prev_size}\n"
                                        f"AI ვერ იპოვა ზუსტი შესაბამისი.\n\n"
                                        f"✅ გვაქვს:\n{confirm_url}\n\n"
                                        f"❌ არ გვაქვს:\n{deny_url}\n\n"
                                        f"📋 ადმინ პანელი:\n{admin_url}"
                                    ),
                                )
                                await _log(f"step6_forwarded_with_prev_size={prev_size}")
                            except Exception as e:
                                await _log(f"step6_direct_forward_error={str(e)[:80]}")
                            _pending_photos.pop(conversation_id, None)
                            text = (
                                "[AI ვერ იპოვა. ზომა უკვე ცნობილია. ზუსტად ეს უპასუხე: "
                                "'ვამოწმებ და მალე მოგწერთ ✨']"
                            )
                        elif photo_bytes:
                            # First photo in the session and AI failed — we don't know the
                            # customer's size yet. Hold the photo and ask first.
                            _pending_failed_photos[conversation_id] = photo_bytes
                            _pending_photos.pop(conversation_id, None)
                            await _log("step6_awaiting_size_before_owner")
                            text = (
                                "[AI ვერ იპოვა ზუსტი შესაბამისი. ზუსტად ეს უპასუხე: "
                                "'რა ზომა გაინტერესებთ? ✨\\n\\n"
                                "📦 პატარა (33x25სმ) — 69₾\\n"
                                "📦 დიდი (37x27სმ) — 74₾']"
                            )
                        else:
                            await _log("step6_NO_PHOTO_BYTES")
                            text = "[ფოტო ვერ დამუშავდა. ზუსტად ეს უპასუხე: 'გადავამოწმებ და მალე მოგწერთ ✨']"

        # ── Link handling — try to extract og:image and treat like a photo ──
        elif text and re.search(r'https?://', text):
            link_match = re.search(r'https?://\S+', text)
            link_url = link_match.group(0) if link_match else ""
            user_text = re.sub(r'https?://\S+', '', text).strip()

            # Try to extract the product image from the link (og:image / twitter:image)
            og_image_bytes = b""
            try:
                og_image_bytes = await _extract_og_image(link_url)
            except Exception as e:
                print(f"[LINK] og:image extraction failed: {e}", flush=True)

            if og_image_bytes:
                # Got the image — convert link into an image message and
                # reuse the normal photo pipeline by setting image_url.
                print(f"[LINK] Extracted og:image ({len(og_image_bytes)} bytes) from {link_url[:60]}", flush=True)
                image_bytes = og_image_bytes
                _pending_photos[conversation_id] = image_bytes
                # Run AI match against the extracted image
                try:
                    from src.vision_match import analyze_and_match
                    match_result = await asyncio.wait_for(analyze_and_match(image_bytes), timeout=120)
                except Exception as e:
                    print(f"[LINK] AI match error: {e}", flush=True)
                    match_result = {}

                if match_result.get("matched"):
                    ai_code = match_result["code"]
                    # Compute prev_size for the link branch too — mirrors the
                    # photo branch's lookup so a customer who already said
                    # "პატარა" earlier in the session gets a short re-ask.
                    prev_size = ""
                    try:
                        _db2 = await get_db()
                        _msgs = await _db2.fetch(
                            "SELECT content, created_at FROM messages WHERE conversation_id = $1 AND role = 'user' ORDER BY created_at DESC LIMIT 30",
                            conversation_id,
                        )
                        _now = datetime.now(timezone.utc)
                        for pm in _msgs:
                            _c = (pm["content"] or "").lower()
                            _created = pm["created_at"] if pm and "created_at" in pm else None
                            _age = 0.0
                            try:
                                if isinstance(_created, str):
                                    _dt = datetime.fromisoformat(_created.replace("Z", "+00:00"))
                                elif hasattr(_created, "tzinfo"):
                                    _dt = _created
                                else:
                                    _dt = None
                                if _dt is not None:
                                    if _dt.tzinfo is None:
                                        _dt = _dt.replace(tzinfo=timezone.utc)
                                    _age = (_now - _dt).total_seconds() / 60
                            except Exception:
                                _age = 0.0
                            if _age > 60:
                                continue
                            if any(w in _c for w in ("პატარა", "პატარ", "patara", "small")):
                                prev_size = "პატარა"
                                break
                            elif any(w in _c for w in ("დიდი", "დიდ", "didi", "large")):
                                prev_size = "დიდი"
                                break
                    except Exception:
                        pass

                    # Persist linked codes in ai_photo_hints so the downstream
                    # size filter can resolve "პატარა" / "დიდი" to the real
                    # inventory rows — without this, the bot would fall back
                    # to the agent which tries to ask for style next.
                    try:
                        _db = await get_db()
                        linked: set[str] = {ai_code}
                        to_check = [ai_code]
                        while to_check:
                            current = to_check.pop(0)
                            pair_rows = await _db.fetch(
                                "SELECT code_b FROM product_pairs WHERE code_a = $1", current,
                            )
                            for pr in pair_rows:
                                if pr["code_b"] not in linked:
                                    linked.add(pr["code_b"])
                                    to_check.append(pr["code_b"])
                        codes_str = ",".join(sorted(linked))
                        now_iso = datetime.now(timezone.utc).isoformat()
                        await _db.execute(
                            """INSERT INTO ai_photo_hints (conversation_id, code, model, size, price, image_url, score, created_at)
                               VALUES ($1, $2, '', '', 0, '', $3, $4)
                               ON CONFLICT (conversation_id) DO UPDATE SET code=$2, score=$3, created_at=$4""",
                            conversation_id, codes_str, float(match_result.get("score", 0)), now_iso,
                        )
                        print(f"[LINK] Saved hints for {conversation_id}: {codes_str}", flush=True)
                    except Exception as e:
                        print(f"[LINK] Hint-save error: {e}", flush=True)

                    if match_result.get("sold_out"):
                        if prev_size:
                            _pending_photos.pop(conversation_id, None)
                            text = (
                                f"[ლინკიდან იპოვა მოდელი ({ai_code}) მაგრამ {prev_size} ზომაში მარაგი ამოწურულია. ზუსტად ეს უპასუხე: "
                                f"'სამწუხაროდ {prev_size} ზომაში მარაგი ამოწურულია ✨']"
                            )
                        else:
                            _pending_soldout[conversation_id] = ai_code
                            _pending_photos.pop(conversation_id, None)
                            text = (
                                "[ლინკიდან იპოვა მოდელი მაგრამ მარაგი ამოწურულია. ჯერ ზომა ვკითხოთ. ზუსტად ეს უპასუხე: "
                                "'რა ზომა გაინტერესებთ? ✨\\n\\n"
                                "📦 პატარა (33x25სმ) — 69₾\\n"
                                "📦 დიდი (37x27სმ) — 74₾']"
                            )
                    else:
                        if prev_size:
                            text = (
                                f"[ლინკიდან იპოვა მოდელი ({ai_code}). ადრე {prev_size} ზომა აირჩია. ზუსტად ეს უპასუხე: "
                                f"'ესეც {prev_size} ზომაში გადაგიმოწმოთ? ✨']"
                            )
                        else:
                            text = (
                                f"[ლინკიდან იპოვა მოდელი ({ai_code}). ზუსტად ეს უპასუხე: "
                                f"'რა ზომაში გადაგიმოწმოთ? ✨\\n\\n📦 პატარა (33x25სმ) — 69₾\\n📦 დიდი (37x27სმ) — 74₾']"
                            )
                else:
                    # Extracted the image but AI couldn't match — forward
                    # both link AND image to owner in background.
                    _bg_wa_image(
                        image_bytes,
                        caption=(
                            f"🔗 კლიენტმა ბმული გამოგზავნა:\n{link_url}\n\n"
                            f"📷 ფოტო ლინკიდან ამოვწერე, AI ვერ ჩანცა.\n"
                            f"{('💬 ' + user_text) if user_text else ''}\n"
                            f"👤 {customer_name or 'კლიენტი'}"
                        ),
                    )
                    text = "[ლინკიდან ფოტო ამოვიღე მაგრამ ვერ დავადგინე. ზუსტად ეს უპასუხე: 'გადავამოწმებ და მალე მოგწერთ ✨']"
            else:
                # Couldn't extract image from link — old behavior: forward URL + ask size
                wa_msg = f"🔗 კლიენტმა ბმული გამოგზავნა:\n{link_url}"
                if user_text:
                    wa_msg += f"\n💬 {user_text}"
                wa_msg += f"\n\n👤 {customer_name or 'კლიენტი'}"
                _bg_wa_text(wa_msg)
                if user_text:
                    text = (
                        f"[კლიენტმა ბმული გამოგზავნა პროდუქტის. მფლობელს გადაეგზავნა. კლიენტი ეკითხება: '{user_text}'. "
                        f"ზუსტად ეს უპასუხე: 'რა ზომა გაინტერესებთ? ✨\\n\\n📦 პატარა (33x25სმ) — 69₾\\n📦 დიდი (37x27სმ) — 74₾']"
                    )
                else:
                    text = "[კლიენტმა ბმული გამოგზავნა. მფლობელს გადაეგზავნა. უთხარი 'გადავამოწმებ ✨']"

        # ── Photo-incoming signal (text only, photo may arrive next) ──
        # Customers often type "ესეც", "ეს გაქვთ?", "შეამოწმეთ ეს", etc. immediately
        # before sending a photo. If the buffer timeout elapses, the agent sees only
        # the text and — when it's mid-order — can jump to "which bank?". Catching
        # these phrases up-front prevents the mis-interpretation. We only trigger on
        # SHORT messages that don't also contain a size word (otherwise "შეამოწმე
        # პატარა ზომაში" would be hijacked away from the size filter).
        if not image_url and text:
            _raw = text.strip().lower()
            _compact = _raw.replace("?", "").replace("!", "").strip()
            _has_size = any(w in _compact for w in ("პატარა", "დიდი", "patara", "didi", "small", "large"))
            _photo_signals = (
                "ესე", "ესეც", "ესეთი", "ეს გაქვთ", "ეს მაქვს", "ესეთი გაქვთ", "ეს?",
                "შეამოწმ", "გადაამოწმ", "გადამიმოწმ",
                "eseti", "esec", "ese", "check this", "this one",
            )
            _is_photo_signal = (not _has_size) and len(_compact) < 30 and (
                any(_compact == w for w in _photo_signals)
                or any(_compact.startswith(w) for w in _photo_signals)
            )
            if _is_photo_signal:
                text = "[კლიენტი ახალი ფოტოს გაგზავნას ცდილობს. ზუსტად ეს უპასუხე: 'გამომიგზავნეთ ფოტო ✨']"

        # ── When user answers size after photo → filter AI results by size ──
        force_inventory_codes = ""  # set when we already know exact codes to show
        if not image_url and text:
            text_lower = text.lower().strip()
            size_wanted = ""
            if any(w in text_lower for w in (
                "პატარა", "პატარე", "პატარ", "small", "33",
                "patara", "patarа", "patra",  # Latin transliterations
            )):
                size_wanted = "პატარა"
            elif any(w in text_lower for w in (
                "დიდი", "დიდ", "large", "big", "37",
                "didi", "did", "didia",  # Latin transliterations
            )):
                size_wanted = "დიდი"

            # If user just confirmed ("კი"/"დიახ") after bot asked "ესეც X ზომაში გადაგიმოწმოთ?",
            # infer size from the last bot message.
            _yes_words = (
                "კი", "დიახ", "ok", "კარგი", "yes", "ჰო", "ჰოო", "კიიი",
                "ki", "diax", "diakh", "ho", "hoo", "kargi", "okey", "okei",
            )
            text_yes = any(text_lower == w for w in _yes_words) or text_lower.startswith(
                ("კი ", "დიახ ", "ki ", "diax ", "ho ", "yes ")
            )
            if not size_wanted and text_yes:
                try:
                    _pool_hist = await get_db()
                    last_bot = await _pool_hist.fetchrow(
                        "SELECT content FROM messages WHERE conversation_id = $1 AND role = 'assistant' ORDER BY created_at DESC LIMIT 1",
                        conversation_id,
                    )
                    last_content = (last_bot["content"] or "").lower() if last_bot else ""
                    if "გადაგიმოწმოთ" in last_content or "გადავამოწმოთ" in last_content:
                        if "პატარა" in last_content:
                            size_wanted = "პატარა"
                        elif "დიდი" in last_content:
                            size_wanted = "დიდი"
                except Exception:
                    pass

            # Size answered after a sold-out match — announce it with the size.
            if size_wanted and conversation_id in _pending_soldout:
                so_code = _pending_soldout.pop(conversation_id)
                print(f"[PHOTO] Sold-out with size={size_wanted} for {so_code}", flush=True)
                text = (
                    f"[AI-მ იპოვა მოდელი ({so_code}) მაგრამ {size_wanted} ზომაში მარაგი ამოწურულია. "
                    f"ზუსტად ეს უპასუხე: 'სამწუხაროდ {size_wanted} ზომაში მარაგი ამოწურულია ✨']"
                )

            # Size answered for a photo that AI previously failed to match →
            # forward it to WhatsApp now, with the size we just learned.
            if size_wanted and conversation_id in _pending_failed_photos:
                pending_bytes = _pending_failed_photos.pop(conversation_id)
                try:
                    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
                    admin_url = f"{public_url}/admin"
                    confirm_url = build_confirm_url("photo-confirm", await create_confirm_token(conversation_id, "photo-confirm"))
                    deny_url = build_confirm_url("photo-deny", await create_confirm_token(conversation_id, "photo-deny"))
                    _bg_wa_image(
                        pending_bytes,
                        caption=(
                            f"📷 კლიენტი ეძებს ამ მოდელს.\n"
                            f"📐 ზომა: {size_wanted}\n"
                            f"AI ვერ იპოვა ზუსტი შესაბამისი.\n\n"
                            f"✅ გვაქვს:\n{confirm_url}\n\n"
                            f"❌ არ გვაქვს:\n{deny_url}\n\n"
                            f"📋 ადმინ პანელი:\n{admin_url}"
                        ),
                    )
                    print(f"[PHOTO] Forwarded to owner with size={size_wanted}", flush=True)
                except Exception as e:
                    print(f"[PHOTO] Delayed owner forward error: {e}", flush=True)
                text = "[AI ვერ იპოვა. ზომა მიეცა მფლობელს. ზუსტად ეს უპასუხე: 'ვამოწმებ და მალე მოგწერთ ✨']"

            if size_wanted:
                try:
                    _pool = await get_db()
                    hint_row = await _pool.fetchrow(
                        "SELECT code FROM ai_photo_hints WHERE conversation_id = $1", conversation_id,
                    )
                    if hint_row and hint_row["code"]:
                        codes = [c.strip() for c in hint_row["code"].split(",")]
                        # Don't delete — keep hints so follow-ups like
                        # "დიდიც გაქვთ?" after asking for small can still resolve
                        # to the linked large version of the same product.

                        # Filter by size — check which linked codes are IN STOCK in this size
                        if codes:
                            placeholders = ",".join(f"${i+1}" for i in range(len(codes)))
                            size_param_idx = len(codes) + 1
                            matching = await _pool.fetch(
                                f"SELECT code, model, size, price, stock FROM inventory WHERE UPPER(code) IN ({placeholders}) AND size ILIKE ${size_param_idx} AND stock > 0",
                                *codes, f"%{size_wanted}%",
                            )
                            if matching:
                                found_codes = ",".join(r["code"] for r in matching)
                                force_inventory_codes = found_codes
                                print(f"[PHOTO] Size filter: wanted={size_wanted} found={found_codes}", flush=True)
                                text = (
                                    f"[{size_wanted} ზომაში ეს მოვიძიეთ: {found_codes}. "
                                    f"გამოიძახე check_inventory(search='{found_codes}') და უპასუხე: "
                                    f"'{size_wanted} ზომაში ეს მოდელი მოვიძიეთ ✨ გნებავთ?']"
                                )
                            else:
                                # Nothing in stock for requested size — is it sold out or
                                # did the product simply never exist in that size?
                                sold_out_in_size = await _pool.fetch(
                                    f"SELECT code FROM inventory WHERE UPPER(code) IN ({placeholders}) AND size ILIKE ${size_param_idx} AND stock = 0",
                                    *codes, f"%{size_wanted}%",
                                )
                                other_size = "დიდი" if size_wanted == "პატარა" else "პატარა"
                                other_in_stock = await _pool.fetch(
                                    f"SELECT code FROM inventory WHERE UPPER(code) IN ({placeholders}) AND size ILIKE ${size_param_idx} AND stock > 0",
                                    *codes, f"%{other_size}%",
                                )
                                if sold_out_in_size:
                                    so_codes = ",".join(r["code"] for r in sold_out_in_size)
                                    print(f"[PHOTO] Size filter: sold out for size={size_wanted} codes={so_codes}", flush=True)
                                    text = (
                                        f"[ეს მოდელი {size_wanted} ზომაში გვქონდა მაგრამ ამჟამად დასრულდა (stock=0). "
                                        f"ზუსტად ეს უპასუხე: 'სამწუხაროდ {size_wanted} ზომაში მარაგი ამოწურულია ✨']"
                                    )
                                elif other_in_stock:
                                    other_codes = ",".join(r["code"] for r in other_in_stock)
                                    force_inventory_codes = other_codes
                                    text = (
                                        f"[AI-მ იპოვა მაგრამ {size_wanted} ზომაში არ გვაქვს. "
                                        f"მხოლოდ {other_size} ზომაში გვაქვს: {other_codes}. "
                                        f"გამოიძახე check_inventory(search='{other_codes}') და უპასუხე: "
                                        f"'სამწუხაროდ {size_wanted} ზომაში არ გვაქვს, მაგრამ {other_size} ზომაში ეს ვიპოვე ✨']"
                                    )
                                else:
                                    text = f"[AI-მ ვერ იპოვა {size_wanted} ზომაში. უპასუხე: 'სამწუხაროდ ეს მოდელი {size_wanted} ზომაში არ გვაქვს ✨ სხვა მოდელები გაჩვენოთ?']"
                except Exception as e:
                    print(f"[PHOTO] Size filter error: {e}", flush=True)

        # ── Customer name + delivery day hint ──
        delivery_day = _get_delivery_day()
        sys_prefix = f"[SYSTEM: delivery_day={delivery_day}]"
        if customer_name:
            sys_prefix = f"[SYSTEM: customer_name={customer_name}; delivery_day={delivery_day}]"
        text = f"{sys_prefix}\n{text}"

        # ── Clear history on greeting so bot always starts fresh ──
        _GREETING_PATTERNS = (
            "გამარჯობა", "სალამი", "სალამ", "გამარჯობა!", "მოგესალმებ",
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        )
        raw_text = text.split("\n", 1)[-1].strip().lower()
        if raw_text in _GREETING_PATTERNS or raw_text in {p + "!" for p in _GREETING_PATTERNS}:
            try:
                _pool = await get_db()
                await _pool.execute(
                    "DELETE FROM messages WHERE conversation_id = $1",
                    conversation_id,
                )
                print(f"[MSG] Greeting detected — cleared history for {conversation_id}", flush=True)
            except Exception as _e:
                print(f"[MSG] History clear error: {_e}", flush=True)

        # ── Run agent ──
        print(f"[MSG] Calling agent...", flush=True)
        agent = await get_support_sales_agent(tenant_id)
        try:
            result = await run_agent(agent, text, conversation_id)
        except Exception as e:
            print(f"[MSG] Agent error: {e}", flush=True)
            _bg_wa_text(f"🚨 აგენტის შეცდომა!\n{text[:200]}\n{str(e)[:300]}")
            result = {"reply": "გადავამოწმებ და მოგწერთ ✨", "tool_calls_made": [], "tool_results_data": {}}

        print(f"[MSG] Agent replied: {result['reply'][:80]}...", flush=True)

        # ── Force-populate inventory when size filter already knows exact codes ──
        if force_inventory_codes:
            trd = result.setdefault("tool_results_data", {})
            inv = trd.get("check_inventory")
            if not inv or not inv.get("found"):
                try:
                    from src.tools.support import check_inventory as _check_inv
                    forced = await _check_inv(search=force_inventory_codes)
                    if forced.get("found"):
                        trd["check_inventory"] = forced
                        print(f"[MSG] Force-injected inventory for codes={force_inventory_codes}", flush=True)
                except Exception as e:
                    print(f"[MSG] Force inventory error: {e}", flush=True)
            # A size-filter match is a fresh answer — bypass the "already sent
            # this category" dedup so the customer actually sees the photos.
            result["_bypass_inventory_dedup"] = True

        # ── Send reply ──
        if not FB_PAGE_TOKEN:
            return

        async with httpx.AsyncClient(timeout=30) as client:
            fb_api = "https://graph.facebook.com/v21.0/me/messages"
            fb_params = {"access_token": FB_PAGE_TOKEN}

            reply_text = result["reply"].strip()
            reply_text = re.sub(r'\[[^\]]{10,}\]', '', reply_text).strip()
            reply_text = re.sub(r'https?://\S+', '', reply_text).strip()
            reply_text = re.sub(r'/static/\S+', '', reply_text).strip()
            reply_text = re.sub(r'\n{3,}', '\n\n', reply_text).strip()

            if not reply_text:
                # Empty reply = bot chose silence (e.g. waiting for address+phone)
                print(f"[MSG] Empty reply — not sending anything", flush=True)
            else:
                fb_resp = await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"text": reply_text[:2000]},
                })
                print(f"[MSG] FB reply: {fb_resp.status_code} {fb_resp.text[:200]}", flush=True)

            await _send_product_images(client, fb_api, fb_params, sender_id, result, conversation_id)

        print(f"[MSG] DONE", flush=True)

    except Exception as e:
        print(f"[MSG] CRASH: {e}", flush=True)
        _tb.print_exc()
        # Always try to respond to customer even on crash
        try:
            if FB_PAGE_TOKEN and sender_id:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        "https://graph.facebook.com/v21.0/me/messages",
                        params={"access_token": FB_PAGE_TOKEN},
                        json={"recipient": {"id": sender_id}, "message": {"text": "ერთი წუთით, გადავამოწმებ ✨"}},
                    )
        except Exception:
            pass


async def _send_product_images(
    client: httpx.AsyncClient, fb_api: str, fb_params: dict,
    sender_id: str, result: dict, conversation_id: str = "",
) -> None:
    tool_data = result.get("tool_results_data", {})
    inventory_data = tool_data.get("check_inventory")
    if not inventory_data or not inventory_data.get("found"):
        return

    items = inventory_data.get("items", [])
    if not items:
        return

    # Check if we already sent this exact category to this conversation.
    # Bypass this check when the caller forced inventory based on a size-filter
    # match — those are fresh answers for a specific photo, not re-listings.
    if conversation_id and len(items) > 1 and not result.get("_bypass_inventory_dedup"):
        first = items[0]
        category_key = f"{first.get('model', '')}|{first.get('size', '')}"
        already_sent = _sent_inventory.get(conversation_id, set())
        if category_key in already_sent:
            print(f"[PHOTO] Skipping — already sent {category_key} to {conversation_id}", flush=True)
            return
        already_sent.add(category_key)
        _sent_inventory[conversation_id] = already_sent

    # Filter out codes the customer has already seen in this conversation.
    # When they ask "დიდიც გაქვთ?" after seeing FP6, we'd resolve to FD2 via
    # pairs — but FP6's photos were already sent. If they haven't seen FD2
    # yet, we still send FD2, otherwise everything is text-only.
    already_seen_codes = _sent_codes.get(conversation_id, set()) if conversation_id else set()
    prefiltered_items = [it for it in items if (it.get("code") or "").upper() not in already_seen_codes]
    if conversation_id and not prefiltered_items:
        print(f"[PHOTO] All codes already shown in this session — skipping photo send", flush=True)
        return
    items_to_send = prefiltered_items if prefiltered_items else items

    sent_codes: set[str] = set()
    photos_sent = 0
    for item in items_to_send:
        code = item.get("code", "")
        front = item.get("image_url", "")
        if not front or not code or code in sent_codes:
            continue
        sent_codes.add(code)
        if conversation_id:
            _sent_codes.setdefault(conversation_id, set()).add(code.upper())

        try:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": f"📌 {code}"},
            })
        except Exception:
            pass

        try:
            img_url = PUBLIC_URL + front if front.startswith("/") else front
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
            })
            photos_sent += 1
        except Exception:
            pass

        back = item.get("image_url_back", "")
        if back:
            try:
                img_url = PUBLIC_URL + back if back.startswith("/") else back
                await client.post(fb_api, params=fb_params, json={
                    "recipient": {"id": sender_id},
                    "message": {"attachment": {"type": "image", "payload": {"url": img_url, "is_reusable": True}}},
                })
            except Exception:
                pass

    # After photos — tell customer to pick a code, but only when >1 new photo
    # was actually sent (don't prompt if only one new code was shown).
    if photos_sent > 1:
        try:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": "შეარჩიეთ მოდელი და შესაბამისი კოდი მოგვწერეთ ✨"},
            })
        except Exception:
            pass


@router.get("/webhook")
async def webhook_verify(request: Request):
    """Meta hits this URL during webhook setup with hub.verify_token.
    We accept either the legacy hardcoded token (Tissu page) or any
    value that matches a tenant's stored fb_verify_token — that way
    every shop can use its own verify string without us redeploying."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode != "subscribe":
        raise HTTPException(status_code=403, detail="Verification failed")
    if token == VERIFY_TOKEN:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)
    # Multi-tenant: is this a verify_token we have on file for any tenant?
    try:
        pool = await get_db()
        match = await pool.fetchval(
            "SELECT tenant_id FROM tenants WHERE fb_verify_token = $1",
            token or "",
        )
    except Exception:
        match = None
    if match:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def webhook_receive(request: Request):
    body = await request.json()

    if body.get("object") not in ("page", "instagram"):
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        # Multi-tenant dispatcher: the page that received the event
        # decides which tenant owns it. Unknown pages are dropped
        # (return 200 to Meta so they don't retry; we don't reply).
        entry_page_id = str(entry.get("id", ""))
        tenant_id_for_entry, _page_token_for_entry = await _resolve_tenant_context(entry_page_id)
        if not tenant_id_for_entry:
            # Unknown page — dropped with a log line in _resolve_tenant_context.
            continue
        # NOTE: the per-tenant token isn't threaded through the rest
        # of this handler yet — that's the next refactor. For now
        # replies still use the env-var FB_PAGE_TOKEN, which is right
        # for the default Tissu tenant. See TISSU-HANDOFF.md §7.
        if tenant_id_for_entry != DEFAULT_TENANT_ID:
            logger.info(
                "[webhook] event for non-default tenant=%s page=%s — "
                "dispatcher recognized but reply path still single-tenant",
                tenant_id_for_entry, entry_page_id,
            )

        for event in entry.get("messaging", []):
            if event.get("delivery") or event.get("read"):
                continue

            sender_id = event.get("sender", {}).get("id", "")
            message = event.get("message", {})
            text = message.get("text", "")
            mid = message.get("mid", "")

            # Drop self-echoes: FB page, IG business account, or Meta's is_echo flag.
            # The page-id check uses the entry's own id, so a second
            # tenant's page doesn't get misclassified as a self-echo.
            if (message.get("is_echo")
                    or sender_id == entry_page_id
                    or sender_id == PAGE_ID
                    or (IG_USER_ID and sender_id == IG_USER_ID)
                    or not sender_id):
                continue

            image_url = ""
            for att in message.get("attachments", []):
                if att.get("type") == "image":
                    image_url = att.get("payload", {}).get("url", "")
                    break

            if not text and not image_url:
                continue

            if mid and mid in _processed_mids:
                continue
            if mid:
                _processed_mids[mid] = time.time()
                _cleanup_old_mids()

            channel = "instagram_dm" if body["object"] == "instagram" else "facebook_messenger"
            conversation_id = f"{channel}_{sender_id}"

            customer_name = ""
            if FB_PAGE_TOKEN:
                try:
                    async with httpx.AsyncClient(timeout=5) as c:
                        resp = await c.get(
                            f"https://graph.facebook.com/v21.0/{sender_id}",
                            params={"fields": "first_name,last_name,name", "access_token": FB_PAGE_TOKEN},
                        )
                        profile = resp.json()
                        customer_name = profile.get("name", "") or f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
                except Exception:
                    pass

            # === BUFFERING: text↔photo pairing ===

            # Case 1: Photo arrived
            if image_url:
                if sender_id in _pending_text:
                    # Text was waiting → combine
                    buffered = _pending_text.pop(sender_id)
                    asyncio.create_task(_process_message(
                        sender_id, buffered["text"], conversation_id, channel, customer_name, image_url,
                    ))
                else:
                    # Photo alone → buffer, wait for text
                    _pending_photo[sender_id] = {
                        "image_url": image_url, "mid": mid, "conv_id": conversation_id,
                        "channel": channel, "customer_name": customer_name,
                    }

                    async def _photo_wait(sid: str, stored_mid: str):
                        await asyncio.sleep(BUFFER_SECONDS)
                        if sid in _pending_photo and _pending_photo[sid]["mid"] == stored_mid:
                            buf = _pending_photo.pop(sid)
                            asyncio.create_task(_process_message(
                                sid, "", buf["conv_id"], buf["channel"], buf["customer_name"], buf["image_url"],
                            ))

                    asyncio.create_task(_photo_wait(sender_id, mid))
                continue

            # Case 2: Text arrived
            if text:
                if sender_id in _pending_photo:
                    # Photo was waiting → combine
                    buffered = _pending_photo.pop(sender_id)
                    asyncio.create_task(_process_message(
                        sender_id, text, conversation_id, channel, customer_name, buffered["image_url"],
                    ))
                elif _PHOTO_HINT.search(text):
                    # Text hints at photo → buffer, wait for photo
                    _pending_text[sender_id] = {
                        "text": text, "mid": mid, "conv_id": conversation_id,
                        "channel": channel, "customer_name": customer_name,
                    }

                    async def _text_wait(sid: str, stored_mid: str):
                        await asyncio.sleep(BUFFER_SECONDS)
                        if sid in _pending_text and _pending_text[sid]["mid"] == stored_mid:
                            buf = _pending_text.pop(sid)
                            asyncio.create_task(_process_message(
                                sid, buf["text"], buf["conv_id"], buf["channel"], buf["customer_name"], "",
                            ))

                    asyncio.create_task(_text_wait(sender_id, mid))
                else:
                    # Normal text → process immediately
                    asyncio.create_task(_process_message(
                        sender_id, text, conversation_id, channel, customer_name, "",
                    ))
                continue

    return {"status": "ok"}
