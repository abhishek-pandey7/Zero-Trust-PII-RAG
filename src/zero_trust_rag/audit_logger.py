"""
audit_logger.py – Compliance audit logging for Zero-Trust RAG.
"""
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from .config import Config

config = Config()
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class AuditLogger:
    """
    Structured audit logger for PII redaction events.
    Outputs JSON lines for easy parsing by SIEM tools.

    Each instance appends to its own file. Events never contain the values being protected: only
    entity types, tokens, counts and the workspace they belong to.
    """

    def __init__(self, log_path: str | None = None):
        self._log_path = Path(log_path or config.get("security.audit_log_path", "./audit.log"))
        self._lock = threading.Lock()

    def _write(self, event: dict) -> None:
        line = json.dumps({"timestamp": _utc_now(), **event})
        with self._lock, open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_redaction(self, event_type: str, entity_type: str, token: str, action: str = "mask",
                      workspace: str | None = None) -> None:
        """Record that a value was tokenized. Only the token is logged, never the value or its text."""
        self._write({
            "event_type": event_type,
            "workspace": workspace,
            "entity_type": entity_type,
            "token": token,
            "action": action,
        })

    def log_operation(self, event_type: str, status: str, details: dict | None = None,
                      workspace: str | None = None) -> None:
        self._write({
            "event_type": event_type,
            "workspace": workspace,
            "status": status,
            "details": details or {},
        })

    def log_access(self, operation: str, pii_accessed: bool = False, details: dict | None = None,
                   workspace: str | None = None) -> None:
        """Record a read of protected data, such as revealing a vault value."""
        self._write({
            "event_type": "access",
            "workspace": workspace,
            "operation": operation,
            "pii_accessed": pii_accessed,
            "details": details or {},
        })

    def get_recent_logs(self, limit: int = 50, workspace: str | None = None) -> list[dict]:
        """Recent events for the UI, newest first. With a workspace, only that workspace's events."""
        if not self._log_path.exists():
            return []
        with self._lock, open(self._log_path, encoding="utf-8") as f:
            lines = f.readlines()
        logs = []
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed audit log line")
                continue
            if workspace is not None and event.get("workspace") != workspace:
                continue
            logs.append(event)
            if len(logs) == limit:
                break
        return logs
