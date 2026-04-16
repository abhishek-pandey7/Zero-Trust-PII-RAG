"""
audit_logger.py – Compliance audit logging for Zero-Trust RAG.
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from .config import Config

config = Config()


class AuditLogger:
    """
    Structured audit logger for PII redaction events.
    Outputs JSON lines for easy parsing by SIEM tools.
    """

    def __init__(self, log_path: Optional[str] = None):
        self._log_path = log_path or config.get("security.audit_log_path", "./audit.log")
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.INFO)
        
        if not self._logger.handlers:
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)

    def log_redaction(self, event_type: str, entity_type: str, token: str,
                      text_preview: str = "", action: str = "mask") -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "entity_type": entity_type,
            "token": token,
            "text_preview": text_preview[:50],
            "action": action
        }
        self._logger.info(json.dumps(event))

    def log_operation(self, event_type: str, status: str, 
                    details: Optional[dict] = None) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "status": status,
            "details": details or {}
        }
        self._logger.info(json.dumps(event))

    def log_access(self, operation: str, pii_accessed: bool = False) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "access",
            "operation": operation,
            "pii_accessed": pii_accessed
        }
        self._logger.info(json.dumps(event))