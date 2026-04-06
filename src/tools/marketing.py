"""Tools for the Marketing + Content + Ads agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from src.db import get_db
from src.engine import Tool


async def save_content(title: str, body: str, content_type: str, tags: list[str] | str = "") -> dict:
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    tags_json = json.dumps(tags if isinstance(tags, list) else [t.strip() for t in tags.split(",") if t.strip()])
    row = await pool.fetchrow(
        "INSERT INTO content (title, body, content_type, tags, status, created_at, updated_at) VALUES ($1, $2, $3, $4, 'draft', $5, $6) RETURNING id",
        title, body, content_type, tags_json, now, now,
    )
    return {"success": True, "content_id": row["id"], "message": f"Content '{title}' saved as draft."}


async def list_content(content_type: str = "", status: str = "", limit: int = 10) -> dict:
    pool = await get_db()
    query = "SELECT id, title, content_type, status, tags, created_at FROM content WHERE 1=1"
    params = []
    idx = 1
    if content_type:
        query += f" AND content_type = ${idx}"
        params.append(content_type)
        idx += 1
    if status:
        query += f" AND status = ${idx}"
        params.append(status)
        idx += 1
    query += f" ORDER BY created_at DESC LIMIT ${idx}"
    params.append(limit)
    rows = await pool.fetch(query, *params)
    return {
        "count": len(rows),
        "content": [dict(r) for r in rows],
    }


async def schedule_content(content_id: int, scheduled_at: str) -> dict:
    pool = await get_db()
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE content SET status = 'scheduled', scheduled_at = $1, updated_at = $2 WHERE id = $3",
        scheduled_at, now, content_id,
    )
    return {"success": True, "message": f"Content #{content_id} scheduled for {scheduled_at}."}


async def get_content_stats() -> dict:
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT content_type, status, COUNT(*) as count FROM content GROUP BY content_type, status"
    )
    total_row = await pool.fetchrow("SELECT COUNT(*) as total FROM content")
    total = total_row["total"]
    return {
        "total_pieces": total,
        "breakdown": [dict(r) for r in rows],
    }


async def get_lead_insights() -> dict:
    pool = await get_db()
    rows = await pool.fetch(
        "SELECT status, COUNT(*) as count, AVG(score) as avg_score FROM leads GROUP BY status"
    )
    sources = await pool.fetch(
        "SELECT source, COUNT(*) as count FROM leads GROUP BY source"
    )
    return {
        "by_status": [dict(r) for r in rows],
        "by_source": [dict(s) for s in sources],
    }


MARKETING_TOOLS = [
    Tool(
        name="save_content",
        description="Save generated content (blog post, social media post, email copy, ad copy) to the database as a draft.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Content title or headline"},
                "body": {"type": "string", "description": "The full content body"},
                "content_type": {
                    "type": "string",
                    "enum": ["blog_post", "social_media", "email", "ad_copy", "landing_page", "newsletter"],
                    "description": "Type of content",
                },
                "tags": {"type": "string", "description": "Comma-separated tags for categorization"},
            },
            "required": ["title", "body", "content_type"],
        },
        handler=save_content,
    ),
    Tool(
        name="list_content",
        description="List existing content pieces. Use to check what's already been created before generating new content.",
        parameters={
            "type": "object",
            "properties": {
                "content_type": {"type": "string", "description": "Filter by type (blog_post, social_media, email, ad_copy)"},
                "status": {"type": "string", "description": "Filter by status (draft, scheduled, published)"},
                "limit": {"type": "integer", "description": "Max results to return"},
            },
            "required": [],
        },
        handler=list_content,
    ),
    Tool(
        name="schedule_content",
        description="Schedule a content piece for publishing at a specific date and time.",
        parameters={
            "type": "object",
            "properties": {
                "content_id": {"type": "integer", "description": "ID of the content to schedule"},
                "scheduled_at": {"type": "string", "description": "ISO datetime for publishing (e.g., 2026-04-01T10:00:00Z)"},
            },
            "required": ["content_id", "scheduled_at"],
        },
        handler=schedule_content,
    ),
    Tool(
        name="get_content_stats",
        description="Get statistics about content: total pieces, breakdown by type and status.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=get_content_stats,
    ),
    Tool(
        name="get_lead_insights",
        description="Get lead analytics: counts by status, average scores, sources. Use for data-driven marketing decisions.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=get_lead_insights,
    ),
]
