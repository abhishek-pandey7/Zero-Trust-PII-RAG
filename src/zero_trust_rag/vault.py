"""
vault.py – Thread-safe PII vault with SQLite persistence for Zero-Trust RAG.

This module provides secure storage and resolution of PII tokens.
Data is persisted to SQLite for resume capability after restarts.
"""
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from .config import Config

# <<ENTITY_TYPE_NN_hhhh>>, tolerating missing angle brackets in LLM output
TOKEN_PATTERN = re.compile(r"<{0,2}(?P<core>\b[A-Z][A-Z_]*_\d{2,}_[0-9a-f]{4}\b)>{0,2}")

config = Config()
logger = logging.getLogger(__name__)


class PIIVault:
    """
    Thread-safe vault for storing PII values with opaque token mapping.
    Supports SQLite persistence for resume capability after restarts.
    """

    def __init__(self, db_path: str | None = None):
        self._lock = threading.Lock()
        self._db_path = db_path or config.get("security.vault_db_path", "./vault.db")
        self._counters: dict[str, int] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_mappings (
                    token TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    real_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS value_mappings (
                    real_value TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    token TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS counters (
                    entity_type TEXT PRIMARY KEY,
                    count INTEGER NOT NULL
                )
            """)
            conn.commit()

            cursor = conn.execute("SELECT entity_type, count FROM counters")
            for entity_type, count in cursor:
                self._counters[entity_type] = count

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def store(self, entity_type: str, real_value: str) -> str:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT token FROM value_mappings WHERE real_value = ?",
                    (real_value,)
                )
                row = cursor.fetchone()
                if row:
                    return row["token"]

                count = self._counters.get(entity_type, 0) + 1
                self._counters[entity_type] = count

                short_id = uuid.uuid4().hex[:4]
                token = f"<<{entity_type}_{count:02d}_{short_id}>>"

                conn.execute(
                    "INSERT INTO token_mappings (token, entity_type, real_value) VALUES (?, ?, ?)",
                    (token, entity_type, real_value)
                )
                conn.execute(
                    "INSERT INTO value_mappings (real_value, entity_type, token) VALUES (?, ?, ?)",
                    (real_value, entity_type, token)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO counters (entity_type, count) VALUES (?, ?)",
                    (entity_type, count)
                )
                conn.commit()

                logger.info(f"Stored PII token: {token} for {entity_type}")
                return token

    def resolve(self, token: str) -> str | None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT real_value FROM token_mappings WHERE token = ?",
                    (token,)
                )
                row = cursor.fetchone()
                return row["real_value"] if row else None

    def unmask_text(self, text: str) -> str:
        """Replace every token in text with its real value.

        Looks up only the tokens that actually appear, instead of scanning the whole vault. Also
        accepts tokens an LLM has echoed without their angle brackets (PERSON_01_a1b2) or with one
        bracket lost; unknown tokens are left as they are.
        """
        found = {m.group("core") for m in TOKEN_PATTERN.finditer(text)}
        if not found:
            return text
        tokens = [f"<<{core}>>" for core in found]
        placeholders = ",".join("?" * len(tokens))
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT token, real_value FROM token_mappings WHERE token IN ({placeholders})", tokens
            ).fetchall()
        values = {row["token"][2:-2]: row["real_value"] for row in rows}
        return TOKEN_PATTERN.sub(lambda m: values.get(m.group("core"), m.group(0)), text)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def known_values(self) -> list[tuple[str, str]]:
        """Every (real value, token) pair, for the egress guard's known-value sweep."""
        with self._lock, self._get_connection() as conn:
            return [(row["real_value"], row["token"]) for row in conn.execute(
                "SELECT real_value, token FROM token_mappings"
            )]

    def get_all_mappings(self) -> list[dict]:
        """Get all token mappings with full details for the UI."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT token, entity_type, real_value, created_at FROM token_mappings ORDER BY created_at DESC")
                return [dict(row) for row in cursor]