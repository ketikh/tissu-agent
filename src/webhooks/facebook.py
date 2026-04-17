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
from src.db import get_db
from src.engine import run_agent
from src.notifications import send_whatsapp_image, send_whatsapp_text
from src.tools.support import _pending_photos, _ai_hints
from src.vision import download_image, is_payment_receipt
from src.webhooks.whatsapp import build_confirm_url, create_confirm_token

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_TOKEN = "tissu_verify_2026"
FB_PAGE_TOKEN = os.getenv("FB_PAGE_TOKEN", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
PAGE_ID = "447377388462459"

_processed_mids: dict[str, float] = {}

# Track which inventory categories were already sent to each conversation
# Key: conversation_id, Value: set of "model|size" combos already sent
_sent_inventory: dict[str, set[str]] = {}

# Buffers for text↔photo pairing
_pending_text: dict[str, dict] = {}   # sender_id -> {text, mid, ...} — text waiting for photo
_pending_photo: dict[str, dict] = {}  # sender_id -> {image_url, mid, ...} — photo waiting for text

# When AI fails to match a photo we ask the customer for size first; we keep the
# bytes here and only forward to WhatsApp once we know which size to send.
_pending_failed_photos: dict[str, bytes] = {}

_PHOTO_HINT = re.compile(r'(გაქვთ|მოდელი|ჩანთა|ასეთი|მსგავსი|ეს\s*არის|have|this)', re.IGNORECASE)

BUFFER_SECONDS = 3


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
            image_bytes = await download_image(image_url)
            if not image_bytes:
                text = "[კლიენტმა ფოტო გამოგზავნა მაგრამ ვერ დამუშავდა. უთხარი 'გადავამოწმებ და მოგწერთ ✨']"
            else:
                is_receipt = await is_payment_receipt(image_bytes, conversation_id)
                if is_receipt:
                    cname = customer_name or "კლიენტი"
                    confirm_url = build_confirm_url("owner-confirm", await create_confirm_token(conversation_id, "owner-confirm"))
                    deny_url = build_confirm_url("owner-deny", await create_confirm_token(conversation_id, "owner-deny"))
                    await send_whatsapp_image(
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
                    try:
                        await _log("step2_importing_vision_match")
                        from src.vision_match import analyze_and_match
                        await _log("step3_calling_analyze_and_match")
                        match_result = await asyncio.wait_for(analyze_and_match(image_bytes), timeout=120)
                        await _log(f"step4_result_matched={match_result.get('matched')}_code={match_result.get('code')}_score={match_result.get('score')}")
                        if match_result and match_result.get("matched"):
                            ai_code = match_result["code"]
                            ai_product = match_result.get("product", {})
                            await _log(f"step5_SUCCESS_{ai_code}")
                        else:
                            await _log(f"step5_NO_MATCH_score={match_result.get('score','?')}_closest={match_result.get('closest_code','?')}_msg={match_result.get('message','?')}")
                    except asyncio.TimeoutError:
                        await _log("step_ERROR_TIMEOUT_60s")
                    except Exception as e:
                        await _log(f"step_ERROR_{type(e).__name__}_{str(e)[:100]}")

                    if ai_code:
                        # Only reuse a previously-chosen size when we're *still* inside
                        # a photo flow from recent activity. Otherwise old sessions leak
                        # their size preference into brand-new conversations.
                        prev_size = ""
                        try:
                            # Step A: is the previous bot turn a photo-flow state?
                            photo_flow_keywords = (
                                "გადაგიმოწმოთ", "მოვიძიეთ", "ვიპოვე", "რა ზომაში",
                                "ზომაში ეს მოდელი", "ზომაში არ გვაქვს",
                                "შეარჩიეთ", "კოდი მოგვწერეთ",
                            )
                            last_bot = await _debug_pool.fetchrow(
                                "SELECT content, created_at FROM messages "
                                "WHERE conversation_id = $1 AND role = 'assistant' "
                                "ORDER BY created_at DESC LIMIT 1",
                                conversation_id,
                            )
                            in_photo_flow = False
                            if last_bot and last_bot["content"]:
                                lc = last_bot["content"]
                                if any(kw in lc for kw in photo_flow_keywords):
                                    in_photo_flow = True

                            # Step B: find a recent size choice (within 60 min) by the user
                            if in_photo_flow:
                                prev_msgs = await _debug_pool.fetch(
                                    "SELECT content, created_at FROM messages "
                                    "WHERE conversation_id = $1 AND role = 'user' "
                                    "ORDER BY created_at DESC LIMIT 20",
                                    conversation_id,
                                )
                                now_utc = datetime.now(timezone.utc)
                                for pm in prev_msgs:
                                    created = pm.get("created_at") if isinstance(pm, dict) else pm["created_at"]
                                    if created and hasattr(created, "tzinfo"):
                                        if created.tzinfo is None:
                                            created = created.replace(tzinfo=timezone.utc)
                                        age_min = (now_utc - created).total_seconds() / 60
                                        if age_min > 60:
                                            continue
                                    c = (pm["content"] or "").lower()
                                    if "პატარა" in c or "პატარ" in c:
                                        prev_size = "პატარა"
                                        break
                                    elif "დიდი" in c or "დიდ" in c:
                                        prev_size = "დიდი"
                                        break
                            await _log(f"step5a_prev_size={prev_size or 'NONE'}_flow={in_photo_flow}")
                        except Exception as e:
                            await _log(f"step5a_error={str(e)[:80]}")

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

                        if prev_size:
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
                        # Hold the photo and ask the customer for size first.
                        # Once they answer with "პატარა"/"დიდი" the size-filter branch
                        # below will forward it to WhatsApp with size context.
                        photo_bytes = _pending_photos.get(conversation_id)
                        if photo_bytes:
                            _pending_failed_photos[conversation_id] = photo_bytes
                            _pending_photos.pop(conversation_id, None)
                            await _log("step6_awaiting_size_before_owner")
                        else:
                            await _log("step6_NO_PHOTO_BYTES")
                        text = (
                            "[AI ვერ იპოვა ზუსტი შესაბამისი. ზუსტად ეს უპასუხე: "
                            "'რა ზომა გაინტერესებთ? ✨\\n\\n"
                            "📦 პატარა (33x25სმ) — 69₾\\n"
                            "📦 დიდი (37x27სმ) — 74₾']"
                        )

        # ── Link handling — forward to owner like photo ──
        elif text and re.search(r'https?://', text):
            link_match = re.search(r'https?://\S+', text)
            link_url = link_match.group(0) if link_match else ""
            # Extract user's text without the link
            user_text = re.sub(r'https?://\S+', '', text).strip()
            # Forward link to owner via WhatsApp
            wa_msg = f"🔗 კლიენტმა ბმული გამოგზავნა:\n{link_url}"
            if user_text:
                wa_msg += f"\n💬 {user_text}"
            wa_msg += f"\n\n👤 {customer_name or 'კლიენტი'}"
            await send_whatsapp_text(wa_msg)
            if user_text:
                # User also wrote something ("ეს გაქვთ?") — let bot respond to the question
                text = (
                    f"[კლიენტმა ბმული გამოგზავნა პროდუქტის. მფლობელს გადაეგზავნა. კლიენტი ეკითხება: '{user_text}'. "
                    f"ზუსტად ეს უპასუხე: 'რა ზომა გაინტერესებთ? ✨\\n\\n📦 პატარა (33x25სმ) — 69₾\\n📦 დიდი (37x27სმ) — 74₾']"
                )
            else:
                # Only a link, no text
                text = "[კლიენტმა ბმული გამოგზავნა. მფლობელს გადაეგზავნა. უთხარი 'გადავამოწმებ ✨']"

        # ── When user answers size after photo → filter AI results by size ──
        force_inventory_codes = ""  # set when we already know exact codes to show
        if not image_url and text:
            text_lower = text.lower().strip()
            size_wanted = ""
            if any(w in text_lower for w in ("პატარა", "პატარე", "პატარ", "small", "33")):
                size_wanted = "პატარა"
            elif any(w in text_lower for w in ("დიდი", "დიდ", "large", "big", "37")):
                size_wanted = "დიდი"

            # If user just confirmed ("კი"/"დიახ") after bot asked "ესეც X ზომაში გადაგიმოწმოთ?",
            # infer size from the last bot message.
            if not size_wanted and text_lower in ("კი", "დიახ", "ok", "კარგი", "yes", "ჰო", "ჰოო"):
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

            # Size answered for a photo that AI previously failed to match →
            # forward it to WhatsApp now, with the size we just learned.
            if size_wanted and conversation_id in _pending_failed_photos:
                pending_bytes = _pending_failed_photos.pop(conversation_id)
                try:
                    public_url = os.getenv("PUBLIC_URL", "https://tissu-agent-production.up.railway.app")
                    admin_url = f"{public_url}/admin"
                    confirm_url = build_confirm_url("photo-confirm", await create_confirm_token(conversation_id, "photo-confirm"))
                    deny_url = build_confirm_url("photo-deny", await create_confirm_token(conversation_id, "photo-deny"))
                    await send_whatsapp_image(
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
                        await _pool.execute("DELETE FROM ai_photo_hints WHERE conversation_id = $1", conversation_id)

                        # Filter by size — check which codes exist in this size
                        if codes:
                            placeholders = ",".join(f"${i+1}" for i in range(len(codes)))
                            size_param_idx = len(codes) + 1
                            matching = await _pool.fetch(
                                f"SELECT code, model, size, price FROM inventory WHERE UPPER(code) IN ({placeholders}) AND size ILIKE ${size_param_idx} AND stock > 0",
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
                                # Check other size
                                other_size = "დიდი" if size_wanted == "პატარა" else "პატარა"
                                other = await _pool.fetch(
                                    f"SELECT code FROM inventory WHERE UPPER(code) IN ({placeholders}) AND size ILIKE ${size_param_idx} AND stock > 0",
                                    *codes, f"%{other_size}%",
                                )
                                if other:
                                    other_codes = ",".join(r["code"] for r in other)
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

        # ── Run agent ──
        print(f"[MSG] Calling agent...", flush=True)
        agent = get_support_sales_agent()
        try:
            result = await run_agent(agent, text, conversation_id)
        except Exception as e:
            print(f"[MSG] Agent error: {e}", flush=True)
            await send_whatsapp_text(f"🚨 აგენტის შეცდომა!\n{text[:200]}\n{str(e)[:300]}")
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

    sent_codes: set[str] = set()
    photos_sent = 0
    for item in items:
        code = item.get("code", "")
        front = item.get("image_url", "")
        if not front or not code or code in sent_codes:
            continue
        sent_codes.add(code)

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

    # After photos — tell customer to pick a code
    if photos_sent > 0 and len(items) > 1:
        try:
            await client.post(fb_api, params=fb_params, json={
                "recipient": {"id": sender_id},
                "message": {"text": "შეარჩიეთ მოდელი და შესაბამისი კოდი მოგვწერეთ ✨"},
            })
        except Exception:
            pass


@router.get("/webhook")
async def webhook_verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def webhook_receive(request: Request):
    body = await request.json()

    if body.get("object") not in ("page", "instagram"):
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            if event.get("delivery") or event.get("read"):
                continue

            sender_id = event.get("sender", {}).get("id", "")
            message = event.get("message", {})
            text = message.get("text", "")
            mid = message.get("mid", "")

            if message.get("is_echo") or sender_id == PAGE_ID or not sender_id:
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
