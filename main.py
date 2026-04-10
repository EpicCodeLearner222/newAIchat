"""
AI Chatbot Backend
- Cookie-based guest identity (30-day rolling expiry)
- SQLite chat storage with conversations & messages
- AI API integration (Groq / OpenAI / custom)
- GitHub-hosted config, knowledge, and examples
"""

import os
import uuid
import time
import sqlite3
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────
# ENV & CONFIG
# ──────────────────────────────────────────────

load_dotenv()

AI_API_KEY      = os.getenv("AI_API_KEY", "")
AI_PROVIDER     = os.getenv("AI_PROVIDER", "groq")
AI_MODEL        = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")
AI_CUSTOM_URL   = os.getenv("AI_CUSTOM_URL", "")
AI_TEMPERATURE  = float(os.getenv("AI_TEMPERATURE", "0.7"))
AI_MAX_TOKENS   = int(os.getenv("AI_MAX_TOKENS", "1024"))
HISTORY_LIMIT   = int(os.getenv("HISTORY_LIMIT", "40"))

GH_CONFIG_URL     = os.getenv("GH_CONFIG_URL", "")
GH_KNOWLEDGE_URL  = os.getenv("GH_KNOWLEDGE_URL", "")
GH_EXAMPLES_URL   = os.getenv("GH_EXAMPLES_URL", "")

COOKIE_NAME       = "guest_token"
COOKIE_MAX_AGE    = 30 * 24 * 60 * 60   # 30 days in seconds

DB_PATH = Path("data/chatbot.db")

API_ENDPOINTS = {
    "groq":   "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}

# ──────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                guest_id      TEXT UNIQUE NOT NULL,
                fingerprint   TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen     TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                title         TEXT NOT NULL DEFAULT 'New Chat',
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id            TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role          TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content       TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_msg_conv  ON messages(conversation_id);
        """)


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────
# GUEST IDENTITY
# ──────────────────────────────────────────────

def generate_fingerprint(request: Request) -> str:
    parts = [
        request.headers.get("user-agent", ""),
        request.headers.get("accept-language", ""),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def resolve_guest(request: Request, response: Response) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    fingerprint = generate_fingerprint(request)

    with get_db() as db:
        if token:
            row = db.execute(
                "SELECT * FROM users WHERE guest_id = ?", (token,)
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE users SET last_seen = datetime('now'), fingerprint = ? WHERE id = ?",
                    (fingerprint, row["id"]),
                )
                _set_cookie(response, token)
                return dict(row)

        # Mint new identity
        user_id = str(uuid.uuid4())
        guest_id = str(uuid.uuid4())

        db.execute(
            "INSERT INTO users (id, guest_id, fingerprint) VALUES (?, ?, ?)",
            (user_id, guest_id, fingerprint),
        )
        _set_cookie(response, guest_id)

        return {
            "id": user_id,
            "guest_id": guest_id,
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }


def _set_cookie(response: Response, guest_id: str):
    response.set_cookie(
        key=COOKIE_NAME,
        value=guest_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,       # Set True in production with HTTPS
    )


# ──────────────────────────────────────────────
# GITHUB CONFIG LOADER
# ──────────────────────────────────────────────

_gh_cache = {
    "config": None,
    "knowledge": None,
    "examples": None,
}


async def fetch_json(url: str) -> Optional[dict]:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None


async def load_github_config():
    _gh_cache["config"]    = await fetch_json(GH_CONFIG_URL)
    _gh_cache["knowledge"] = await fetch_json(GH_KNOWLEDGE_URL)
    _gh_cache["examples"]  = await fetch_json(GH_EXAMPLES_URL)

    names = [k for k, v in _gh_cache.items() if v is not None]
    if names:
        print(f"[INFO] Loaded from GitHub: {', '.join(names)}")
    else:
        print("[INFO] No GitHub config loaded — using defaults")


# ──────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ──────────────────────────────────────────────

def build_system_prompt() -> str:
    parts = []
    config = _gh_cache.get("config")
    knowledge = _gh_cache.get("knowledge")

    if config and config.get("personality"):
        parts.append(config["personality"])

    if config and config.get("instructions"):
        lines = "\n".join(f"- {i}" for i in config["instructions"])
        parts.append(f"## Instructions\n{lines}")

    if knowledge:
        parts.append("## Knowledge Base")
        if knowledge.get("context"):
            parts.append(knowledge["context"])
        for topic in knowledge.get("topics", []):
            parts.append(f"### {topic['name']}\n{topic['content']}")

    if not parts:
        parts.append(
            "You are a helpful, friendly AI assistant. "
            "Be concise and clear. Use markdown formatting when it helps."
        )

    return "\n\n".join(parts)


def build_messages(conversation_messages: list[dict], user_message: str) -> list[dict]:
    messages = []

    # 1. System prompt
    messages.append({"role": "system", "content": build_system_prompt()})

    # 2. Few-shot examples
    examples = _gh_cache.get("examples")
    if examples and examples.get("conversations"):
        for ex in examples["conversations"]:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})

    # 3. Conversation history (trimmed)
    trimmed = conversation_messages[-HISTORY_LIMIT:]
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 4. New user message
    messages.append({"role": "user", "content": user_message})

    return messages


# ──────────────────────────────────────────────
# AI API CALLER
# ──────────────────────────────────────────────

async def call_ai(conversation_messages: list[dict], user_message: str) -> str:
    if not AI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="No AI_API_KEY configured on the server."
        )

    if AI_PROVIDER == "custom":
        endpoint = AI_CUSTOM_URL
    else:
        endpoint = API_ENDPOINTS.get(AI_PROVIDER, API_ENDPOINTS["groq"])

    if not endpoint:
        raise HTTPException(status_code=500, detail="No API endpoint configured.")

    config = _gh_cache.get("config") or {}
    payload = {
        "model": AI_MODEL,
        "messages": build_messages(conversation_messages, user_message),
        "temperature": config.get("temperature", AI_TEMPERATURE),
        "max_tokens": config.get("maxTokens", AI_MAX_TOKENS),
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {AI_API_KEY}",
                },
                json=payload,
            )

            if resp.status_code != 200:
                err = resp.json().get("error", {}).get("message", resp.text)
                raise HTTPException(status_code=502, detail=f"AI API error: {err}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI API request timed out.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI API call failed: {str(e)}")


# ──────────────────────────────────────────────
# AUTO TITLE
# ──────────────────────────────────────────────

def auto_title(user_message: str) -> str:
    title = user_message.strip()
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "…"
    return title or "New Chat"


# ══════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════

app = FastAPI(title="AI Chatbot", version="1.0.0")


@app.on_event("startup")
async def startup():
    init_db()
    await load_github_config()


# ── Request Models ──

class CreateConversation(BaseModel):
    title: Optional[str] = "New Chat"

class RenameConversation(BaseModel):
    title: str

class ChatMessage(BaseModel):
    message: str


# ──────────────────────────────────────────────
# ROUTES: Identity
# ──────────────────────────────────────────────

@app.get("/api/me")
def get_me(request: Request, response: Response):
    user = resolve_guest(request, response)
    with get_db() as db:
        conv_count = db.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE user_id = ?",
            (user["id"],)
        ).fetchone()["c"]
        msg_count = db.execute(
            """SELECT COUNT(*) as c FROM messages m
               JOIN conversations c ON m.conversation_id = c.id
               WHERE c.user_id = ?""",
            (user["id"],)
        ).fetchone()["c"]

    return {
        "guest_id": user["guest_id"],
        "created_at": user["created_at"],
        "last_seen": user["last_seen"],
        "conversations": conv_count,
        "total_messages": msg_count,
    }


# ──────────────────────────────────────────────
# ROUTES: Conversations
# ──────────────────────────────────────────────

@app.get("/api/conversations")
def list_conversations(request: Request, response: Response):
    user = resolve_guest(request, response)
    with get_db() as db:
        rows = db.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
               FROM conversations c
               WHERE c.user_id = ?
               ORDER BY c.updated_at DESC""",
            (user["id"],)
        ).fetchall()
    return {"conversations": [dict(r) for r in rows]}


@app.post("/api/conversations", status_code=201)
def create_conversation(
    body: CreateConversation, request: Request, response: Response
):
    user = resolve_guest(request, response)
    conv_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (conv_id, user["id"], body.title),
        )
    return {"id": conv_id, "title": body.title}


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(
    conv_id: str, body: RenameConversation, request: Request, response: Response
):
    user = resolve_guest(request, response)
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        db.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (body.title, conv_id),
        )
    return {"id": conv_id, "title": body.title}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, request: Request, response: Response):
    user = resolve_guest(request, response)
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user["id"]),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    return {"deleted": conv_id}


# ──────────────────────────────────────────────
# ROUTES: Messages & Chat
# ──────────────────────────────────────────────

@app.get("/api/conversations/{conv_id}/messages")
def get_messages(conv_id: str, request: Request, response: Response):
    user = resolve_guest(request, response)
    with get_db() as db:
        conv = db.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user["id"]),
        ).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        rows = db.execute(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        ).fetchall()

    return {
        "conversation_id": conv_id,
        "title": conv["title"],
        "messages": [dict(r) for r in rows],
    }


@app.post("/api/conversations/{conv_id}/chat")
async def chat(
    conv_id: str, body: ChatMessage, request: Request, response: Response
):
    user_message = body.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user = resolve_guest(request, response)

    with get_db() as db:
        conv = db.execute(
            "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user["id"]),
        ).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        history_rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

        user_msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
            (user_msg_id, conv_id, user_message, now),
        )

    # Call AI (outside DB lock)
    ai_response = await call_ai(history, user_message)

    with get_db() as db:
        asst_msg_id = str(uuid.uuid4())
        asst_time = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
            (asst_msg_id, conv_id, ai_response, asst_time),
        )

        msg_count = db.execute(
            "SELECT COUNT(*) as c FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()["c"]
        new_title = conv["title"]
        if msg_count <= 2 and conv["title"] == "New Chat":
            new_title = auto_title(user_message)
            db.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (new_title, conv_id),
            )
        else:
            db.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conv_id,),
            )

    return {
        "conversation_id": conv_id,
        "title": new_title,
        "user_message": {
            "id": user_msg_id,
            "role": "user",
            "content": user_message,
            "created_at": now,
        },
        "assistant_message": {
            "id": asst_msg_id,
            "role": "assistant",
            "content": ai_response,
            "created_at": asst_time,
        },
    }


# ──────────────────────────────────────────────
# ROUTES: Config & Health
# ──────────────────────────────────────────────

@app.post("/api/reload-config")
async def reload_config():
    await load_github_config()
    loaded = {k: v is not None for k, v in _gh_cache.items()}
    return {"reloaded": True, "sources": loaded}


@app.get("/api/health")
def health_check():
    db_exists = DB_PATH.exists()
    has_key = bool(AI_API_KEY)
    return {
        "status": "ok" if (db_exists and has_key) else "degraded",
        "database": db_exists,
        "ai_api_key_set": has_key,
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "github_config": {k: v is not None for k, v in _gh_cache.items()},
    }


# ──────────────────────────────────────────────
# STATIC FILES (frontend)
# ──────────────────────────────────────────────

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

@app.get("/")
def serve_index():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Drop your frontend files in the /static folder."}

if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
