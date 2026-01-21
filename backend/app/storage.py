import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .config import settings


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enquiries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_sanitized TEXT NOT NULL,
                theme TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enquiries_raw (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_raw TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                theme TEXT PRIMARY KEY,
                pattern_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def insert_enquiries(
    sanitized: Iterable[Dict[str, Any]],
    raw: Optional[Iterable[str]] = None,
) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO enquiries (text_sanitized, theme, created_at) VALUES (?, ?, ?)",
            [(item["text_sanitized"], item["theme"], now) for item in sanitized],
        )
        if settings.store_raw and raw:
            conn.executemany(
                "INSERT INTO enquiries_raw (text_raw, created_at) VALUES (?, ?)",
                [(text, now) for text in raw],
            )


def upsert_patterns(theme: str, pattern: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO patterns (theme, pattern_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(theme) DO UPDATE SET
                pattern_json = excluded.pattern_json,
                updated_at = excluded.updated_at
            """,
            (theme, json.dumps(pattern), now),
        )


def insert_generated(theme: str, kind: str, content: Dict[str, Any]) -> int:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO generated (theme, kind, content_json, created_at) VALUES (?, ?, ?, ?)",
            (theme, kind, json.dumps(content), now),
        )
        return int(cursor.lastrowid)


def insert_audit(event_type: str, payload: Dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit (event_type, payload_json, created_at) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), now),
        )


def get_themes_summary() -> List[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT theme, pattern_json FROM patterns")
        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            pattern = json.loads(row["pattern_json"])
            results.append(
                {
                    "theme": row["theme"],
                    "count": pattern.get("count", 0),
                }
            )
        return results


def get_pattern(theme: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM patterns WHERE theme = ?", (theme,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "theme": row["theme"],
            "pattern": json.loads(row["pattern_json"]),
            "updated_at": datetime.fromisoformat(row["updated_at"]),
        }


def get_generated(theme: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect() as conn:
        if theme:
            cursor = conn.execute(
                "SELECT * FROM generated WHERE theme = ? ORDER BY created_at DESC",
                (theme,),
            )
        else:
            cursor = conn.execute("SELECT * FROM generated ORDER BY created_at DESC")
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "theme": row["theme"],
                    "kind": row["kind"],
                    "content": json.loads(row["content_json"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )
        return results


def get_sanitized_texts(theme: Optional[str] = None) -> List[str]:
    with _connect() as conn:
        if theme:
            cursor = conn.execute(
                "SELECT text_sanitized FROM enquiries WHERE theme = ?", (theme,)
            )
        else:
            cursor = conn.execute("SELECT text_sanitized FROM enquiries")
        return [row["text_sanitized"] for row in cursor.fetchall()]


def get_audit_recent(limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT * FROM audit ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )
        return results


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM enquiries")
        conn.execute("DELETE FROM enquiries_raw")
        conn.execute("DELETE FROM patterns")
        conn.execute("DELETE FROM generated")
        conn.execute("DELETE FROM audit")
