"""
exceptions.py – Custom exceptions for Zero-Trust RAG.
"""


class ZeroTrustRAGError(Exception):
    """Base exception for Zero-Trust RAG."""
    pass


class ConfigurationError(ZeroTrustRAGError):
    """Configuration related errors."""
    pass


class APIError(ZeroTrustRAGError):
    """API-related errors (LLM, Embeddings)."""
    def __init__(self, message: str, provider: str = None, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class VaultError(ZeroTrustRAGError):
    """Vault/PII storage errors."""
    pass


class ValidationError(ZeroTrustRAGError):
    """Input validation errors."""
    pass


class RedactionError(ZeroTrustRAGError):
    """PII redaction errors."""
    pass


class EgressBlockedError(ZeroTrustRAGError):
    """Raised when text about to leave the trust boundary still contains identifiers."""
    pass
