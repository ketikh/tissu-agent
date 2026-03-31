"""Tools for the Marketing + Content + Ads agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from src.db import get_db
from src.engine import Tool


async def save_content(title: str, body: str, content_type: str, tags: list[str] | str = "") -> dict:
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(tags if isinstance(tags, list) else [t.strip() for t in tags.split(",") if t.strip()])
        cursor = await db.execute(
            "INSERT INTO content (title, body, content_type, tags, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'draft', ?, ?)",
            (title, body, content_type, tags_json, now, now),
        )
        await db.commit()
        return {"success": True, "content_id": cursor.lastrowid, "message": f"Content '{title}' saved as draft."}
    finally:
        await db.close()


async def list_content(content_type: str = "", status: str = "", limit: int = 10) -> dict:
    db = await get_db()
    try:
        query = "SELECT id, title, content_type, status, tags, created_at FROM content WHERE 1=1"
        params = []
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return {
            "count": len(rows),
            "content": [dict(r) for r in rows],
        }
    finally:
        await db.close()


async def schedule_content(content_id: int, scheduled_at: str) -> dict:
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "UPDATE content SET status = 'scheduled', scheduled_at = ?, updated_at = ? WHERE id = ?",
            (scheduled_at, now, content_id),
        )
        await db.commit()
        return {"success": True, "message": f"Content #{content_id} scheduled for {scheduled_at}."}
    finally:
        await db.close()


async def get_content_stats() -> dict:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT content_type, status, COUNT(*) as count FROM content GROUP BY content_type, status"
        )
        rows = await cursor.fetchall()
        cursor2 = await db.execute("SELECT COUNT(*) as total FROM content")
        total = (await cursor2.fetchone())["total"]
        return {
            "total_pieces": total,
            "breakdown": [dict(r) for r in rows],
        }
    finally:
        await db.close()


async def get_lead_insights() -> dict:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT status, COUNT(*) as count, AVG(score) as avg_score FROM leads GROUP BY status"
        )
        rows = await cursor.fetchall()
        cursor2 = await db.execute(
            "SELECT source, COUNT(*) as count FROM leads GROUP BY source"
        )
        sources = await cursor2.fetchall()
        return {
            "by_status": [dict(r) for r in rows],
            "by_source": [dict(s) for s in sources],
        }
    finally:
        await db.close()


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
