"""Admin insights — surface bot quality signals from the message history.

Three derived views, computed on demand so they always reflect the latest state:

1. Complaints — customer messages showing dissatisfaction (wrong match,
   wants a human operator, "არა ეს არ არის", etc.)
2. FAQ candidates — most frequent customer questions, grouped after
   normalisation. A recurring question is a hint we should encode a
   canned answer.
3. Product requests — customers asking for categories we don't carry
   (iPad / tablet cases, cosmetic bags, belts, etc.) so the owner can
   decide what to stock next.

No new tables — this module only reads ``messages`` and applies heuristics.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from src.db import get_db


# ── Message cleaning ─────────────────────────────────────────────

_SYSTEM_PREFIX_RE = re.compile(r"^\s*\[SYSTEM:[^\]]*\]\s*", re.IGNORECASE)
_BRACKETED_RE = re.compile(r"\[[^\]]*\]")


def clean_user_message(content: str) -> str:
    """Strip system tags and bracketed hints, return the customer's real text.

    Bot-side we prepend `[SYSTEM: ...]` metadata and sometimes inject
    `[კლიენტმა ფოტო გამოგზავნა...]` hints as "user" messages for the LLM.
    Those are not things the customer actually typed, so we filter them out
    before any analysis.
    """
    if not content:
        return ""
    text = _SYSTEM_PREFIX_RE.sub("", content)
    text = _BRACKETED_RE.sub("", text)
    return text.strip()


_EMOJI_RE = re.compile(
    "[\U00010000-\U0010ffff"          # supplementary plane (emoji, symbols)
    "\u2600-\u27bf"                     # misc symbols / dingbats
    "\u2190-\u21ff"                     # arrows
    "]+",
    flags=re.UNICODE,
)
_PUNCT_RE = re.compile(r"[\u0589\.,!?;:\-—–()\"'`«»\[\]{}/\\]+")


def normalize_for_grouping(text: str) -> str:
    """Lowercase + strip punctuation/emoji/extra whitespace for FAQ grouping."""
    t = text.lower()
    t = _EMOJI_RE.sub("", t)
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


# ── Detectors ────────────────────────────────────────────────────

# Phrases that indicate the customer is unhappy with the bot's reply. Kept
# as substrings (no word boundaries) so inflected forms match — Georgian
# is agglutinative and "ოპერატორთან" / "ოპერატორი" should both hit.
COMPLAINT_PHRASES = [
    "არასწორი", "არასწორია",
    "ეს არ არის", "ეს არაა", "ეგ არ არის", "ეგ არაა",
    "არა ეს", "არა ეგ",
    "ოპერატორ", "ცოცხალ ადამიან", "ცოცხალ ოპერატორ", "ადამიანთან",
    "დამაკავშირ", "გადამიერთ", "გადამრთე",
    "არ მომწონ", "არ მიყვარ",
    "განერვიულ",
    "სხვა მინდოდა", "სხვა მქონდა მხედველობაში",
    "არ გესმ",
    "ცუდად ვერ", "ვერ ხვდები",
]

# Product categories we DO NOT carry. Matching any of these in a customer
# message is a signal to add a "requested product" row so the owner can
# decide what to stock next.
UNAVAILABLE_CATEGORIES: dict[str, list[str]] = {
    "აიპადის ქეისი": ["აიპად", "ipad"],
    "პლანშეტის ქეისი": ["პლანშეტ", "tablet"],
    "ტელეფონის ქეისი": ["ტელეფონის ქეი", "ტელეფონის ქი", "phone case", "ქეისი ტელეფონ"],
    "კოსმეტიკის ჩანთა": ["კოსმეტიკ"],
    "საფულე": ["საფულ"],
    "ყელსაბამი": ["ყელსაბამ"],
    "სამაჯური": ["სამაჯურ"],
    "წინსაფარი": ["წინსაფარ"],
    "ბავშვის ჩანთა": ["ბავშვ"],
    "ზურგჩანთა": ["ზურგჩანთ", "backpack"],
    "ჩემოდანი": ["ჩემოდან", "სამგზავრო"],
    "ქამარი": ["ქამარ"],
    "ხელთათმანი": ["ხელთათმან"],
    "ჩექმა": ["ჩექმ"],
    "ქოლგა": ["ქოლგ"],
}


def _match_category(text: str) -> str:
    """Return the canonical label if text mentions an unavailable category, else ''."""
    low = text.lower()
    for label, needles in UNAVAILABLE_CATEGORIES.items():
        for needle in needles:
            if needle.lower() in low:
                return label
    return ""


def _contains_complaint(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in COMPLAINT_PHRASES)


# ── Public queries ───────────────────────────────────────────────

async def list_complaints(limit: int = 50) -> list[dict]:
    """Customer messages that look like complaints, newest first.

    We also grab the bot reply that preceded the complaint so the owner can
    see what the customer was reacting to.
    """
    pool = await get_db()
    rows = await pool.fetch(
        """
        SELECT id, conversation_id, role, content, created_at
        FROM messages
        WHERE conversation_id NOT LIKE 'debug_%'
        ORDER BY conversation_id, created_at ASC
        """
    )
    # Group by conversation so we can find the bot reply right before each
    # flagged user message.
    by_conv: dict[str, list] = defaultdict(list)
    for r in rows:
        by_conv[r["conversation_id"]].append(r)

    out: list[dict] = []
    for conv_id, msgs in by_conv.items():
        for i, m in enumerate(msgs):
            if m["role"] != "user":
                continue
            clean = clean_user_message(m["content"])
            if not clean or not _contains_complaint(clean):
                continue
            # Find the most recent bot reply before this message.
            prev_bot = ""
            for j in range(i - 1, -1, -1):
                if msgs[j]["role"] in ("assistant", "model"):
                    prev_bot = msgs[j]["content"][:500]
                    break
            out.append({
                "conversation_id": conv_id,
                "customer_message": clean,
                "bot_reply_before": prev_bot,
                "created_at": m["created_at"],
            })
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out[:limit]


async def list_faq_candidates(min_count: int = 2, limit: int = 30) -> list[dict]:
    """Most frequent customer questions after normalisation.

    Shown descending by count; the owner can scan this to spot questions
    worth encoding as canned responses in the bot prompt.
    """
    pool = await get_db()
    rows = await pool.fetch(
        """
        SELECT content, MIN(created_at) AS first_seen, MAX(created_at) AS last_seen
        FROM messages
        WHERE role = 'user' AND conversation_id NOT LIKE 'debug_%'
        GROUP BY content
        """
    )
    groups: dict[str, dict] = {}
    for r in rows:
        clean = clean_user_message(r["content"])
        if len(clean) < 2 or len(clean) > 200:
            continue
        key = normalize_for_grouping(clean)
        if not key:
            continue
        g = groups.setdefault(key, {
            "question": clean,
            "count": 0,
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
        })
        g["count"] += 1
        if r["first_seen"] < g["first_seen"]:
            g["first_seen"] = r["first_seen"]
        if r["last_seen"] > g["last_seen"]:
            g["last_seen"] = r["last_seen"]

    ranked = sorted(
        (g for g in groups.values() if g["count"] >= min_count),
        key=lambda g: g["count"],
        reverse=True,
    )
    return ranked[:limit]


async def list_product_requests(limit: int = 100) -> list[dict]:
    """Customer messages mentioning categories we don't carry.

    Aggregated per category so the owner sees "iPad case: 8 asks" rather
    than 8 near-duplicate rows. Also keeps one sample message per category
    for context.
    """
    pool = await get_db()
    rows = await pool.fetch(
        """
        SELECT content, conversation_id, created_at
        FROM messages
        WHERE role = 'user' AND conversation_id NOT LIKE 'debug_%'
        ORDER BY created_at DESC
        """
    )
    agg: dict[str, dict] = {}
    for r in rows:
        clean = clean_user_message(r["content"])
        if not clean:
            continue
        label = _match_category(clean)
        if not label:
            continue
        entry = agg.setdefault(label, {
            "category": label,
            "count": 0,
            "sample": clean[:200],
            "conversations": set(),
            "first_seen": r["created_at"],
            "last_seen": r["created_at"],
        })
        entry["count"] += 1
        entry["conversations"].add(r["conversation_id"])
        if r["created_at"] < entry["first_seen"]:
            entry["first_seen"] = r["created_at"]
        if r["created_at"] > entry["last_seen"]:
            entry["last_seen"] = r["created_at"]

    out = []
    for entry in agg.values():
        out.append({
            "category": entry["category"],
            "count": entry["count"],
            "unique_customers": len(entry["conversations"]),
            "sample": entry["sample"],
            "first_seen": entry["first_seen"],
            "last_seen": entry["last_seen"],
        })
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:limit]
