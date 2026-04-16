"""
zero_trust_rag – Zero-Trust PII Redacting RAG Pipeline.

A production-ready RAG pipeline that masks PII before sending to external APIs.
"""
from .pipeline import ZeroTrustRAG
from .vault import PIIVault
from .pii_redactor import PIIRedactor
from .exceptions import (
    ZeroTrustRAGError,
    ConfigurationError,
    APIError,
    VaultError,
    ValidationError,
    RedactionError,
)

__version__ = "1.0.0"
__all__ = [
    "ZeroTrustRAG",
    "PIIVault",
    "PIIRedactor",
    "ZeroTrustRAGError",
    "ConfigurationError",
    "APIError",
    "VaultError",
    "ValidationError",
    "RedactionError",
]