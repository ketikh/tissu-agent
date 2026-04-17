"""Tissu Shop — Google Cloud Vision based product matching.

Uses Label Detection + Image Properties (dominant colors) + Web Detection
to create rich fingerprints for each product, then matches user photos.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Credentials — support both local file and Railway env variable
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # Option A: local file in secrets/
    _default_cred = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "secrets", "gcloud-key.json"
    )
    if os.path.exists(_default_cred):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _default_cred
    # Option B: Railway — JSON content in GOOGLE_CREDENTIALS_JSON env var
    elif os.getenv("GOOGLE_CREDENTIALS_JSON"):
        import tempfile
        _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        _tmp.write(os.getenv("GOOGLE_CREDENTIALS_JSON"))
        _tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

from google.cloud import vision  # noqa: E402

from src.db import get_db  # noqa: E402

_client: vision.ImageAnnotatorClient | None = None


def _get_client() -> vision.ImageAnnotatorClient:
    global _client
    if _client is None:
        _client = vision.ImageAnnotatorClient()
    return _client


# ─────────────────────────────────────────────────────────────
# Fingerprint extraction
# ─────────────────────────────────────────────────────────────

def _extract_fingerprint_from_image(image: vision.Image) -> dict[str, Any]:
    """Extract labels, colors, and web entities from an image."""
    client = _get_client()

    # Batch request for efficiency — one API call, multiple features
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=15),
        vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
        vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=10),
    ]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)

    # Extract labels
    labels = [
        {"name": lab.description.lower(), "score": lab.score}
        for lab in response.label_annotations
    ]

    # Extract dominant colors (top 5)
    colors = []
    for c in response.image_properties_annotation.dominant_colors.colors[:5]:
        colors.append({
            "r": int(c.color.red),
            "g": int(c.color.green),
            "b": int(c.color.blue),
            "score": c.score,
            "pixel_fraction": c.pixel_fraction,
        })

    # Extract web entities (semantic understanding)
    web_entities = [
        {"name": e.description.lower(), "score": e.score}
        for e in response.web_detection.web_entities
        if e.description
    ]

    return {"labels": labels, "colors": colors, "web_entities": web_entities}


def fingerprint_from_url(image_url: str) -> dict[str, Any]:
    image = vision.Image()
    image.source.image_uri = image_url
    return _extract_fingerprint_from_image(image)


def fingerprint_from_bytes(image_bytes: bytes) -> dict[str, Any]:
    image = vision.Image(content=image_bytes)
    return _extract_fingerprint_from_image(image)


# ─────────────────────────────────────────────────────────────
# Similarity scoring
# ─────────────────────────────────────────────────────────────

def _color_distance(c1: dict, c2: dict) -> float:
    """Euclidean distance in RGB space, normalized 0-1 (0 = identical)."""
    dr = c1["r"] - c2["r"]
    dg = c1["g"] - c2["g"]
    db = c1["b"] - c2["b"]
    dist = (dr * dr + dg * dg + db * db) ** 0.5
    # Max possible distance in RGB = sqrt(3 * 255^2) ≈ 441.67
    return dist / 441.67


def _color_similarity(colors_a: list[dict], colors_b: list[dict]) -> float:
    """Compare two sets of dominant colors. Returns 0-1 (1 = identical)."""
    if not colors_a or not colors_b:
        return 0.0

    # For each color in A, find best match in B
    total_score = 0.0
    total_weight = 0.0
    for ca in colors_a:
        best = 0.0
        for cb in colors_b:
            dist = _color_distance(ca, cb)
            sim = 1 - dist  # 1 = identical, 0 = opposite
            # Weight by how prominent the color is in both
            weighted = sim * min(ca["score"], cb["score"])
            best = max(best, weighted)
        total_score += best * ca["score"]
        total_weight += ca["score"]

    return total_score / total_weight if total_weight > 0 else 0.0


def _label_similarity(labels_a: list[dict], labels_b: list[dict]) -> float:
    """Jaccard-like similarity on label names, weighted by scores."""
    if not labels_a or not labels_b:
        return 0.0

    dict_a = {l["name"]: l["score"] for l in labels_a}
    dict_b = {l["name"]: l["score"] for l in labels_b}

    common = set(dict_a) & set(dict_b)
    if not common:
        return 0.0

    # Sum of min scores for common labels / sum of max scores for all labels
    intersection = sum(min(dict_a[k], dict_b[k]) for k in common)
    union = sum(max(dict_a.get(k, 0), dict_b.get(k, 0)) for k in set(dict_a) | set(dict_b))
    return intersection / union if union > 0 else 0.0


def _entity_similarity(ents_a: list[dict], ents_b: list[dict]) -> float:
    """Same logic as labels but for web entities."""
    return _label_similarity(ents_a, ents_b)


def compute_similarity(fp_a: dict, fp_b: dict) -> float:
    """Weighted combination of color + label + web entity similarity."""
    color_sim = _color_similarity(fp_a.get("colors", []), fp_b.get("colors", []))
    label_sim = _label_similarity(fp_a.get("labels", []), fp_b.get("labels", []))
    entity_sim = _entity_similarity(fp_a.get("web_entities", []), fp_b.get("web_entities", []))

    # Color is most important for pattern-based bags (50%)
    # Labels catch material/type (30%)
    # Web entities catch style/semantic (20%)
    return color_sim * 0.5 + label_sim * 0.3 + entity_sim * 0.2


# ─────────────────────────────────────────────────────────────
# Indexing + matching
# ─────────────────────────────────────────────────────────────

async def ensure_table():
    pool = await get_db()
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS product_fingerprints (
            id SERIAL PRIMARY KEY,
            inventory_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            fingerprint JSONB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    await pool.execute("CREATE INDEX IF NOT EXISTS idx_pf_code ON product_fingerprints(code)")


async def index_product(inventory_id: int, code: str, image_url: str, image_url_back: str = "") -> bool:
    import asyncio
    try:
        pool = await get_db()
        now = datetime.now(timezone.utc).isoformat()

        fp_front = await asyncio.to_thread(fingerprint_from_url, image_url)
        await pool.execute(
            "INSERT INTO product_fingerprints (inventory_id, code, fingerprint, created_at) VALUES ($1, $2, $3, $4)",
            inventory_id, code, json.dumps(fp_front), now,
        )

        if image_url_back:
            try:
                fp_back = await asyncio.to_thread(fingerprint_from_url, image_url_back)
                await pool.execute(
                    "INSERT INTO product_fingerprints (inventory_id, code, fingerprint, created_at) VALUES ($1, $2, $3, $4)",
                    -inventory_id, code, json.dumps(fp_back), now,
                )
            except Exception as e:
                logger.warning(f"Back image failed for {code}: {e}")

        return True
    except Exception as e:
        logger.error(f"Index failed for {code}: {e}")
        return False


async def index_extra_photo(code: str, image_url: str) -> bool:
    """Index a single lifestyle/marketing photo for a product."""
    import asyncio
    try:
        fp = await asyncio.to_thread(fingerprint_from_url, image_url)
        pool = await get_db()
        now = datetime.now(timezone.utc).isoformat()
        # Use negative ID to distinguish from main catalog photos
        await pool.execute(
            "INSERT INTO product_fingerprints (inventory_id, code, tags, embedding, created_at) VALUES ($1, $2, $3, '', $4)",
            -abs(hash(image_url)) % 100000, code, json.dumps(fp), now,
        )
        print(f"[INDEX] Extra photo indexed: {code} → {image_url[:50]}", flush=True)
        return True
    except Exception as e:
        logger.error(f"Extra photo index failed: {e}")
        return False


async def index_all_products() -> dict:
    await ensure_table()
    pool = await get_db()

    # Clear existing
    await pool.execute("DELETE FROM product_fingerprints")

    rows = await pool.fetch("""
        SELECT i.id, i.code, i.image_url, i.image_url_back
        FROM inventory i
        WHERE i.image_url != '' AND i.image_url IS NOT NULL AND i.stock > 0
        ORDER BY i.code
    """)

    success = 0
    failed = 0
    for row in rows:
        print(f"[INDEX] {row['code']}...", flush=True)
        ok = await index_product(row["id"], row["code"], row["image_url"], row.get("image_url_back", ""))
        if ok:
            success += 1
            print(f"[INDEX] ✅ {row['code']}", flush=True)
        else:
            failed += 1
            print(f"[INDEX] ❌ {row['code']}", flush=True)

    return {"indexed": success, "failed": failed, "total": len(rows)}


CLIP_SERVICE_URL = os.getenv("CLIP_SERVICE_URL", "https://katekh12-clip-embed.hf.space")


async def _get_clip_embedding(image_bytes: bytes) -> list[float] | None:
    """Get CLIP embedding from HuggingFace Space."""
    import base64
    try:
        b64 = base64.b64encode(image_bytes).decode()
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{CLIP_SERVICE_URL}/embed",
                json={"image_base64": b64, "remove_bg": True},
            )
            if resp.status_code == 200:
                return resp.json().get("embedding")
    except Exception as e:
        logger.warning(f"CLIP embedding failed: {e}")
    return None


async def _get_clip_embedding_url(image_url: str) -> list[float] | None:
    """Get CLIP embedding for a URL."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{CLIP_SERVICE_URL}/embed",
                json={"image_url": image_url, "remove_bg": True},
            )
            if resp.status_code == 200:
                return resp.json().get("embedding")
    except Exception as e:
        logger.warning(f"CLIP URL embedding failed: {e}")
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def analyze_and_match(image_bytes: bytes, size: str = "") -> dict:
    """Hybrid match: CLIP (visual) + Cloud Vision (semantic). Best of both."""
    import asyncio

    pool = await get_db()
    rows = await pool.fetch("""
        SELECT pf.code, pf.fingerprint, pf.inventory_id,
               i.model, i.size, i.price, i.image_url, i.image_url_back, i.stock
        FROM product_fingerprints pf
        JOIN inventory i ON i.id = ABS(pf.inventory_id)
        WHERE i.stock > 0
    """)
    if not rows:
        return {"matched": False, "message": "მარაგში ვერ მოიძებნა"}

    # ── Method 1: Cloud Vision fingerprint comparison ──
    vision_scores = {}
    user_fp: dict | None = None
    try:
        user_fp = await asyncio.to_thread(fingerprint_from_bytes, image_bytes)
        for row in rows:
            fp = row["fingerprint"]
            if isinstance(fp, str):
                fp = json.loads(fp)
            score = compute_similarity(user_fp, fp)
            code = row["code"]
            if code not in vision_scores or score > vision_scores[code]:
                vision_scores[code] = score
    except Exception as e:
        logger.warning(f"Vision fingerprint failed: {e}")

    # ── Method 2: CLIP visual embedding — use pre-stored embeddings from product_embeddings ──
    clip_scores = {}
    try:
        user_clip = await _get_clip_embedding(image_bytes)
        if user_clip:
            # Read pre-computed CLIP embeddings from product_embeddings table (pgvector)
            clip_rows = await pool.fetch("""
                SELECT code, embedding
                FROM product_embeddings
                WHERE embedding IS NOT NULL
            """)
            for cr in clip_rows:
                code = cr["code"]
                prod_emb = cr["embedding"]
                if prod_emb:
                    # pgvector returns string like "[0.1,0.2,...]" — parse it
                    if isinstance(prod_emb, str):
                        prod_emb = [float(x) for x in prod_emb.strip("[]").split(",")]
                    sim = _cosine_similarity(user_clip, list(prod_emb))
                    if code not in clip_scores or sim > clip_scores[code]:
                        clip_scores[code] = sim
            print(f"[MATCH] CLIP scores: top={max(clip_scores.values()) if clip_scores else 0:.2f} ({len(clip_scores)} products)", flush=True)
    except Exception as e:
        logger.warning(f"CLIP matching failed: {e}")

    # ── Color re-ranking ──
    # CLIP alone matches shape/pattern but ignores color, so an orange bag
    # and a yellow bag with the same pattern score nearly identically.
    # We blend CLIP (shape/semantic, 70%) with dominant-color similarity (30%).
    color_by_code: dict[str, float] = {}
    user_colors: list[dict] = []
    try:
        user_colors = (user_fp or {}).get("colors", [])
    except Exception:
        user_colors = []

    if user_colors:
        for row in rows:
            fp = row["fingerprint"]
            if isinstance(fp, str):
                try:
                    fp = json.loads(fp)
                except Exception:
                    continue
            prod_colors = fp.get("colors", []) if isinstance(fp, dict) else []
            sim = _color_similarity(user_colors, prod_colors)
            code = row["code"]
            if code not in color_by_code or sim > color_by_code[code]:
                color_by_code[code] = sim

    # ── Combine scores: CLIP × color (fallback to whichever exists) ──
    combined: dict[str, float] = {}
    if clip_scores and color_by_code:
        for code, clip_sim in clip_scores.items():
            color_sim = color_by_code.get(code, 0.0)
            combined[code] = clip_sim * 0.7 + color_sim * 0.3
        top_before = max(clip_scores.values())
        top_after = max(combined.values())
        print(f"[MATCH] Color re-rank: CLIP top={top_before:.2f} → blended top={top_after:.2f}", flush=True)
    elif clip_scores:
        combined = dict(clip_scores)
    else:
        combined = dict(vision_scores)

    # Build scored list with row data
    code_to_row = {}
    for row in rows:
        c = row["code"]
        if c not in code_to_row:
            code_to_row[c] = row

    scored = [(combined.get(c, 0), code_to_row[c]) for c in code_to_row if c in combined]

    # Sort best first
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by code (keep best score per code)
    seen = {}
    for score, row in scored:
        if row["code"] not in seen or score > seen[row["code"]][0]:
            seen[row["code"]] = (score, row)

    ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]

    # Threshold for CLIP similarity (0.85+ = exact, 0.75+ = very similar)
    if best_score < 0.75:
        return {
            "matched": False,
            "message": "ზუსტი შესაბამისი ვერ მოიძებნა",
            "closest_code": best["code"],
            "score": round(best_score, 2),
        }

    return {
        "matched": True,
        "code": best["code"],
        "score": round(best_score, 2),
        "product": {
            "code": best["code"],
            "model": best["model"],
            "size": best["size"],
            "price": best["price"],
            "image_url": best["image_url"],
            "image_url_back": best.get("image_url_back", ""),
        },
        "alternatives": [
            {"code": r["code"], "score": round(s, 2)}
            for s, r in ranked[1:3]
            if s > 0.45
        ],
    }
