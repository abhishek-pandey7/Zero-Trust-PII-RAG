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

    Each instance appends to its own file. (It used to share one logging.Logger, so every
    instance after the first silently wrote to the first instance's file.)
    """

    def __init__(self, log_path: str | None = None):
        self._log_path = Path(log_path or config.get("security.audit_log_path", "./audit.log"))
        self._lock = threading.Lock()

    def _write(self, event: dict) -> None:
        line = json.dumps(event)
        with self._lock, open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_redaction(self, event_type: str, entity_type: str, token: str,
                      text_preview: str = "", action: str = "mask") -> None:
        self._write({
            "timestamp": _utc_now(),
            "event_type": event_type,
            "entity_type": entity_type,
            "token": token,
            "text_preview": text_preview[:50],
            "action": action,
        })

    def log_operation(self, event_type: str, status: str, details: dict | None = None) -> None:
        self._write({
            "timestamp": _utc_now(),
            "event_type": event_type,
            "status": status,
            "details": details or {},
        })

    def log_access(self, operation: str, pii_accessed: bool = False) -> None:
        self._write({
            "timestamp": _utc_now(),
            "event_type": "access",
            "operation": operation,
            "pii_accessed": pii_accessed,
        })

    def get_recent_logs(self, limit: int = 50) -> list[dict]:
        """Read recent logs from the file for the UI, newest first."""
        if not self._log_path.exists():
            return []
        with self._lock, open(self._log_path, encoding="utf-8") as f:
            lines = f.readlines()
        logs = []
        for line in reversed(lines[-limit:]):
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed audit log line")
        return logs
