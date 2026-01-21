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
            CREATE TABLE IF NOT EXISTS theme_counts (
                theme TEXT PRIMARY KEY,
                count_low INTEGER NOT NULL,
                count_medium INTEGER NOT NULL,
                count_high INTEGER NOT NULL,
                count_total INTEGER NOT NULL,
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
        # Insert sanitized enquiries and get their IDs
        sanitized_list = list(sanitized)
        cursor = conn.executemany(
            "INSERT INTO enquiries (text_sanitized, theme, created_at) VALUES (?, ?, ?)",
            [(item["text_sanitized"], item["theme"], now) for item in sanitized_list],
        )
        
        if settings.store_raw and raw:
            # Get the starting ID for the batch we just inserted
            last_id = cursor.lastrowid
            if last_id:
                start_id = last_id - len(sanitized_list) + 1
                raw_list = list(raw)
                # Insert raw texts with matching IDs
                raw_with_ids = [(start_id + i, text, now) for i, text in enumerate(raw_list[:len(sanitized_list)])]
                conn.executemany(
                    "INSERT INTO enquiries_raw (id, text_raw, created_at) VALUES (?, ?, ?)",
                    raw_with_ids,
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


def upsert_theme_counts(counts: Iterable[Dict[str, Any]]) -> None:
    now = datetime.utcnow().isoformat()
    payload = []
    for entry in counts:
        theme = entry["theme"]
        count_low = int(entry.get("count_low", 0))
        count_medium = int(entry.get("count_medium", 0))
        count_high = int(entry.get("count_high", 0))
        count_total = count_low + count_medium + count_high
        payload.append((theme, count_low, count_medium, count_high, count_total, now))
    if not payload:
        return
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO theme_counts (
                theme,
                count_low,
                count_medium,
                count_high,
                count_total,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(theme) DO UPDATE SET
                count_low = count_low + excluded.count_low,
                count_medium = count_medium + excluded.count_medium,
                count_high = count_high + excluded.count_high,
                count_total = count_total + excluded.count_total,
                updated_at = excluded.updated_at
            """,
            payload,
        )


def get_theme_counts() -> List[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM theme_counts")
        results_by_theme: Dict[str, Dict[str, Any]] = {}
        for row in cursor.fetchall():
            results_by_theme[row["theme"]] = (
                {
                    "theme": row["theme"],
                    "count_low": row["count_low"],
                    "count_medium": row["count_medium"],
                    "count_high": row["count_high"],
                    "count_total": row["count_total"],
                }
            )
        cursor = conn.execute("SELECT theme, pattern_json FROM patterns")
        for row in cursor.fetchall():
            if row["theme"] in results_by_theme:
                continue
            pattern = json.loads(row["pattern_json"])
            count_total = int(pattern.get("count_total") or pattern.get("count", 0))
            results_by_theme[row["theme"]] = (
                {
                    "theme": row["theme"],
                    "count_low": count_total,
                    "count_medium": 0,
                    "count_high": 0,
                    "count_total": count_total,
                }
            )
        return list(results_by_theme.values())


def get_theme_count(theme: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM theme_counts WHERE theme = ?", (theme,))
        row = cursor.fetchone()
        if not row:
            cursor = conn.execute(
                "SELECT theme, pattern_json FROM patterns WHERE theme = ?", (theme,)
            )
            pattern_row = cursor.fetchone()
            if not pattern_row:
                return None
            pattern = json.loads(pattern_row["pattern_json"])
            count_total = int(pattern.get("count_total") or pattern.get("count", 0))
            return {
                "theme": theme,
                "count_low": count_total,
                "count_medium": 0,
                "count_high": 0,
                "count_total": count_total,
            }
        return {
            "theme": row["theme"],
            "count_low": row["count_low"],
            "count_medium": row["count_medium"],
            "count_high": row["count_high"],
            "count_total": row["count_total"],
        }


def get_themes_summary() -> List[Dict[str, Any]]:
    with _connect() as conn:
        cursor = conn.execute("SELECT * FROM theme_counts")
        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            results.append(
                {
                    "theme": row["theme"],
                    "count": row["count_total"],
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


def get_raw_texts(theme: Optional[str] = None) -> List[str]:
    """Get raw texts for pattern extraction when available."""
    with _connect() as conn:
        if theme:
            # Join with enquiries to get theme-filtered raw texts
            cursor = conn.execute(
                """
                SELECT er.text_raw 
                FROM enquiries_raw er
                JOIN enquiries e ON er.id = e.id
                WHERE e.theme = ?
                """, (theme,)
            )
        else:
            cursor = conn.execute("SELECT text_raw FROM enquiries_raw")
        return [row["text_raw"] for row in cursor.fetchall()]


def get_texts_for_patterns(theme: Optional[str] = None) -> List[str]:
    """Get the best available texts for pattern extraction - raw if available, sanitized otherwise."""
    if settings.store_raw:
        raw_texts = get_raw_texts(theme)
        if raw_texts:
            return raw_texts
    return get_sanitized_texts(theme)


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
        conn.execute("DELETE FROM theme_counts")
        conn.execute("DELETE FROM generated")
        conn.execute("DELETE FROM audit")
