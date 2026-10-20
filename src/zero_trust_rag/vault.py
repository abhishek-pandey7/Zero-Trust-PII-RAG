"""
vault.py – Thread-safe PII vault with SQLite persistence for Zero-Trust RAG.

This module provides secure storage and resolution of PII tokens.
Data is persisted to SQLite for resume capability after restarts.
"""
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from .config import Config

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
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT token, real_value FROM token_mappings")
                mapping = {row["token"]: row["real_value"] for row in cursor}

        for token, real_value in mapping.items():
            text = text.replace(token, real_value)
        return text

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def get_all_tokens(self) -> dict[str, str]:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT token, real_value FROM token_mappings")
                return {row["token"]: row["real_value"] for row in cursor}

    def get_all_mappings(self) -> list[dict]:
        """Get all token mappings with full details for the UI."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT token, entity_type, real_value, created_at FROM token_mappings ORDER BY created_at DESC")
                return [dict(row) for row in cursor]