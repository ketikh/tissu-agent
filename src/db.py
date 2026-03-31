from __future__ import annotations
import aiosqlite
import json
from datetime import datetime, timezone
from src.config import DATABASE_PATH


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(DATABASE_PATH))
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    db = await get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                company TEXT,
                phone TEXT,
                source TEXT,
                score INTEGER DEFAULT 0,
                status TEXT DEFAULT 'new',
                notes TEXT,
                conversation_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'open',
                priority TEXT DEFAULT 'medium',
                customer_email TEXT,
                conversation_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                content_type TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                tags TEXT DEFAULT '[]',
                scheduled_at TEXT,
                published_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                model TEXT NOT NULL,
                size TEXT NOT NULL,
                color TEXT,
                style TEXT,
                code TEXT,
                tags TEXT DEFAULT '',
                price REAL NOT NULL,
                stock INTEGER DEFAULT 0,
                image_url TEXT,
                image_url_back TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                customer_phone TEXT,
                customer_address TEXT,
                items TEXT NOT NULL,
                total REAL NOT NULL,
                delivery_fee REAL DEFAULT 6,
                payment_method TEXT,
                payment_confirmed INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                conversation_id TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        await db.commit()
    finally:
        await db.close()


async def save_message(conversation_id: str, role: str, content: str, tool_calls: list | None = None):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_calls, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(tool_calls) if tool_calls else None, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_conversation_messages(conversation_id: str) -> list[dict]:
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT role, content, tool_calls FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        await db.close()


async def ensure_conversation(conversation_id: str, agent_type: str):
    db = await get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT OR IGNORE INTO conversations (id, agent_type, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, agent_type, now, now),
        )
        await db.commit()
    finally:
        await db.close()
