"""
db.py — Async SQLite logging for reflection sessions and dialogues.

A single aiosqlite connection is opened once at startup (init_db) and
reused for all queries. WAL mode + NORMAL synchronous gives the best
latency/durability trade-off for a single-process bot.

Schema
------
sessions
    session_id   TEXT  PK   – "{user_id}_{unix_timestamp}"
    user_id      INT         – Telegram user ID
    started_at   TEXT        – ISO-8601 UTC
    finished_at  TEXT        – ISO-8601 UTC (filled when DONE)
    final_state  TEXT        – last BotState value
    p1_summary   TEXT        – layer_data from p1_summary skill
    p2_summary   TEXT        – layer_data from p2_summary skill
    p3_summary   TEXT        – layer_data from p3_summary skill

turns  (one row per bot-question + student-answer exchange)
    id           INT   PK autoincrement
    session_id   TEXT        – FK → sessions
    user_id      INT
    state        TEXT        – BotState when bot sent the question
    ts_bot       TEXT        – ISO-8601 UTC, when bot message was logged
    bot_text     TEXT        – bot question / reply (may be multi-message, joined by \n\n)
    ts_student   TEXT        – ISO-8601 UTC, when student replied (NULL until replied)
    student_text TEXT        – student's answer (NULL until replied)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# DB lives in the logs/ folder at project root
_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "reflections.db")
)

# Single persistent connection — opened in init_db(), closed in close_db()
_db: Optional[aiosqlite.Connection] = None


async def init_db() -> None:
    """Open the DB connection and create tables. Call once at startup."""
    global _db
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    _db = await aiosqlite.connect(_DB_PATH)
    _db.row_factory = aiosqlite.Row
    # WAL mode: readers don't block writers and vice-versa
    await _db.execute("PRAGMA journal_mode=WAL")
    # NORMAL sync is safe with WAL and much faster than FULL
    await _db.execute("PRAGMA synchronous=NORMAL")
    await _db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            user_id      INTEGER NOT NULL,
            started_at   TEXT    NOT NULL,
            finished_at  TEXT,
            final_state  TEXT,
            p1_summary   TEXT,
            p2_summary   TEXT,
            p3_summary   TEXT
        );

        CREATE TABLE IF NOT EXISTS turns (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            user_id      INTEGER NOT NULL,
            state        TEXT,
            ts_bot       TEXT    NOT NULL,
            bot_text     TEXT    NOT NULL,
            ts_student   TEXT,
            student_text TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
    """)
    await _db.commit()
    logger.info("[DB] Initialized at %s", _DB_PATH)


async def close_db() -> None:
    """Close the DB connection gracefully. Call on shutdown."""
    global _db
    if _db:
        await _db.close()
        _db = None
        logger.info("[DB] Connection closed.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("DB not initialized — call await init_db() first")
    return _db


_SESSIONS_COLUMNS = frozenset({"finished_at", "final_state", "p1_summary", "p2_summary", "p3_summary"})


async def create_session(session_id: str, user_id: int) -> None:
    """Register a new reflection session."""
    db = _require_db()
    await db.execute(
        "INSERT OR IGNORE INTO sessions (session_id, user_id, started_at)"
        " VALUES (?, ?, ?)",
        (session_id, user_id, _now()),
    )
    await db.commit()


async def log_turn(
    session_id: str,
    user_id: int,
    role: str,
    text: str,
    state: str = "",
) -> None:
    """
    Log one message. role="bot" creates/appends a turn row;
    role="user" fills in student_text on the latest open turn.
    """
    if not session_id:
        return
    db = _require_db()
    if role == "bot":
        cursor = await db.execute(
            "SELECT id, bot_text FROM turns"
            " WHERE session_id = ? AND student_text IS NULL"
            " ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row:
            await db.execute(
                "UPDATE turns SET bot_text = ?, state = ? WHERE id = ?",
                (row["bot_text"] + "\n\n" + text, state or "", row["id"]),
            )
        else:
            await db.execute(
                "INSERT INTO turns (session_id, user_id, state, ts_bot, bot_text)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, state or "", _now(), text),
            )
    else:
        await db.execute(
            """UPDATE turns
               SET student_text = ?, ts_student = ?
               WHERE id = (
                   SELECT MAX(id) FROM turns
                   WHERE session_id = ? AND student_text IS NULL
               )""",
            (text, _now(), session_id),
        )
    await db.commit()


async def update_session(session_id: str, **kwargs) -> None:
    """Update arbitrary columns of an existing session row."""
    if not session_id or not kwargs:
        return
    unknown = set(kwargs) - _SESSIONS_COLUMNS
    if unknown:
        raise ValueError(f"Unknown session column(s): {unknown}")
    db = _require_db()
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    await db.execute(
        f"UPDATE sessions SET {fields} WHERE session_id = ?", vals
    )
    await db.commit()


async def close_session(session_id: str, final_state: str) -> None:
    """Mark a session as finished."""
    await update_session(
        session_id,
        finished_at=_now(),
        final_state=final_state,
    )
