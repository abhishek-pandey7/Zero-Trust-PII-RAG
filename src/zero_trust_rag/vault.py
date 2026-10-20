"""
vault.py – Thread-safe PII vault with SQLite persistence for Zero-Trust RAG.

Maps real values to opaque tokens and back. Every row belongs to a workspace, and every lookup is
scoped to one: a token minted in workspace A can't be resolved from workspace B, so a question in
B can never be answered with A's data even if B somehow obtained A's tokens.
"""
import logging
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager

from .config import Config
from .exceptions import VaultError

config = Config()
logger = logging.getLogger(__name__)

# <<ENTITY_TYPE_NN_hhhh>>, tolerating missing angle brackets in LLM output
TOKEN_PATTERN = re.compile(r"<{0,2}(?P<core>\b[A-Z][A-Z_]*_\d{2,}_[0-9a-f]{4}\b)>{0,2}")

DEFAULT_WORKSPACE = "default"
SCHEMA_VERSION = 2


class PIIVault:
    """
    Thread-safe vault for storing PII values with opaque token mapping.
    Supports SQLite persistence for resume capability after restarts.
    """

    def __init__(self, db_path: str | None = None):
        self._lock = threading.Lock()
        self._db_path = db_path or config.get("security.vault_db_path", "./vault.db")
        self._init_db()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            legacy = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='value_mappings'"
            ).fetchone()
            if legacy or version not in (0, SCHEMA_VERSION):
                # Refuse rather than guess: silently dropping a vault makes every stored chunk unreadable
                raise VaultError(
                    f"{self._db_path} uses an older vault schema. It holds no workspace column, so its "
                    "rows can't be assigned safely. Move it aside and restart to create a new vault."
                )
            conn.executescript(f"""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS token_mappings (
                    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                    token TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    real_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace_id, token),
                    UNIQUE (workspace_id, real_value)
                );
                CREATE TABLE IF NOT EXISTS counters (
                    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (workspace_id, entity_type)
                );
                PRAGMA user_version = {SCHEMA_VERSION};
            """)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def touch(self, workspace: str = DEFAULT_WORKSPACE) -> None:
        """Create the workspace if needed and record that it was used."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "INSERT INTO workspaces (workspace_id) VALUES (?) "
                "ON CONFLICT(workspace_id) DO UPDATE SET last_seen_at = CURRENT_TIMESTAMP",
                (workspace,),
            )
            conn.commit()

    def store(self, entity_type: str, real_value: str, workspace: str = DEFAULT_WORKSPACE) -> str:
        """Return the workspace's token for this value, minting one the first time it's seen."""
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT token FROM token_mappings WHERE workspace_id = ? AND real_value = ?",
                (workspace, real_value),
            ).fetchone()
            if row:
                return row["token"]

            conn.execute("INSERT OR IGNORE INTO workspaces (workspace_id) VALUES (?)", (workspace,))
            count = conn.execute(
                "INSERT INTO counters (workspace_id, entity_type, count) VALUES (?, ?, 1) "
                "ON CONFLICT(workspace_id, entity_type) DO UPDATE SET count = count + 1 RETURNING count",
                (workspace, entity_type),
            ).fetchone()["count"]
            token = f"<<{entity_type}_{count:02d}_{secrets.token_hex(2)}>>"
            conn.execute(
                "INSERT INTO token_mappings (workspace_id, token, entity_type, real_value) VALUES (?, ?, ?, ?)",
                (workspace, token, entity_type, real_value),
            )
            conn.commit()
            logger.info("Stored PII token %s for %s", token, entity_type)
            return token

    def resolve(self, token: str, workspace: str = DEFAULT_WORKSPACE) -> str | None:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT real_value FROM token_mappings WHERE workspace_id = ? AND token = ?",
                (workspace, token),
            ).fetchone()
            return row["real_value"] if row else None

    def unmask_text(self, text: str, workspace: str = DEFAULT_WORKSPACE) -> str:
        """Replace every token in text with its real value.

        Looks up only the tokens that actually appear, instead of scanning the whole vault. Also
        accepts tokens an LLM has echoed without their angle brackets (PERSON_01_a1b2) or with one
        bracket lost; unknown tokens, including other workspaces' tokens, are left as they are.
        """
        found = {m.group("core") for m in TOKEN_PATTERN.finditer(text)}
        if not found:
            return text
        tokens = [f"<<{core}>>" for core in found]
        placeholders = ",".join("?" * len(tokens))
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT token, real_value FROM token_mappings "
                f"WHERE workspace_id = ? AND token IN ({placeholders})",
                [workspace, *tokens],
            ).fetchall()
        values = {row["token"][2:-2]: row["real_value"] for row in rows}
        return TOKEN_PATTERN.sub(lambda m: values.get(m.group("core"), m.group(0)), text)

    def get_stats(self, workspace: str = DEFAULT_WORKSPACE) -> dict[str, int]:
        with self._lock, self._get_connection() as conn:
            return {row["entity_type"]: row["count"] for row in conn.execute(
                "SELECT entity_type, count FROM counters WHERE workspace_id = ?", (workspace,)
            )}

    def known_values(self, workspace: str = DEFAULT_WORKSPACE) -> list[tuple[str, str]]:
        """Every (real value, token) pair in the workspace, for the egress guard's known-value sweep."""
        with self._lock, self._get_connection() as conn:
            return [(row["real_value"], row["token"]) for row in conn.execute(
                "SELECT real_value, token FROM token_mappings WHERE workspace_id = ?", (workspace,)
            )]

    def get_all_mappings(self, workspace: str = DEFAULT_WORKSPACE) -> list[dict]:
        """Get all token mappings with full details for the UI."""
        with self._lock, self._get_connection() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT token, entity_type, real_value, created_at FROM token_mappings "
                "WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace,),
            )]

    def delete_workspace(self, workspace: str) -> int:
        """Erase every mapping in a workspace. Returns how many values were deleted."""
        with self._lock, self._get_connection() as conn:
            deleted = conn.execute(
                "SELECT COUNT(*) FROM token_mappings WHERE workspace_id = ?", (workspace,)
            ).fetchone()[0]
            conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace,))
            conn.commit()
            return deleted

    def stale_workspaces(self, idle_hours: float) -> list[str]:
        """Workspaces not used for idle_hours, oldest first."""
        with self._lock, self._get_connection() as conn:
            return [row["workspace_id"] for row in conn.execute(
                "SELECT workspace_id FROM workspaces "
                "WHERE last_seen_at < datetime('now', ?) ORDER BY last_seen_at",
                (f"-{idle_hours} hours",),
            )]
