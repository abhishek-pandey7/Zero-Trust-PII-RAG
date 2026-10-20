"""
pii_redactor.py – Uses Microsoft Presidio to detect and mask PII entities.

This module provides PII detection and redaction with configurable
entity types and audit logging.
"""
import logging
import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .audit_logger import AuditLogger
from .config import Config
from .vault import PIIVault

config = Config()
logger = logging.getLogger(__name__)

ENTITY_TYPES = config.get("presidio.entity_types", [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "LOCATION",
    "DATE_TIME",
    "NRP",
    "MEDICAL_LICENSE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IN_PAN",
    "IN_AADHAAR",
])

SCORE_THRESHOLD = config.get("presidio.score_threshold", 0.3)


class AadhaarRecognizer(PatternRecognizer):
    """Custom recognizer for Indian Aadhaar numbers."""
    
    def __init__(self):
        patterns = [
            Pattern("Aadhaar separator", r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b", 0.85)
        ]
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=patterns
        )
    
    def invalidate_cached_results(self):
        pass


class IndianPhoneRecognizer(PatternRecognizer):
    """Custom recognizer for Indian phone numbers."""
    
    def __init__(self):
        patterns = [
            Pattern("India phone +91", r"\+91[-\s]?\d{5}[-\s]?\d{5}", 0.7),
            Pattern("India phone 10 digit", r"\b\d{10}\b", 0.5),
            Pattern("India phone split", r"\b\d{5}[-\s]?\d{5}\b", 0.6)
        ]
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=patterns
        )


class PANRecognizer(PatternRecognizer):
    """Custom recognizer for Indian PAN numbers."""
    
    def __init__(self):
        patterns = [
            Pattern("PAN format", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.8)
        ]
        super().__init__(
            supported_entity="IN_PAN",
            patterns=patterns
        )


class PIIRedactor:
    """
    Detects PII in text and replaces each hit with a Vault-managed token.
    Supports audit logging for compliance.
    """

    def __init__(self, vault: PIIVault, audit_logger: AuditLogger | None = None):
        self.vault = vault
        self.audit = audit_logger or AuditLogger()
        
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        
        self.analyzer = AnalyzerEngine()
        self.analyzer.nlp_engine = nlp_engine
        self.analyzer.registry.add_recognizer(AadhaarRecognizer())
        self.analyzer.registry.add_recognizer(IndianPhoneRecognizer())
        self.analyzer.registry.add_recognizer(PANRecognizer())

    @staticmethod
    def sanitize_input(text: str) -> str:
        if not text:
            return ""
        
        sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        sanitized = re.sub(r'[\x80-\x9f]', '', sanitized)
        
        return sanitized.strip()

    def redact(self, text: str, event_type: str = "ingest") -> tuple[str, list[dict]]:
        sanitized = self.sanitize_input(text)
        
        if not sanitized:
            return text, []

        results = self.analyzer.analyze(
            text=sanitized,
            entities=ENTITY_TYPES,
            language='en',
            score_threshold=SCORE_THRESHOLD,
        )

        if not results:
            return sanitized, []

        corrected_results = []
        for result in results:
            matched_text = sanitized[result.start:result.end]
            
            # Prevent entities from spanning across newlines (common issue in structured records)
            if '\n' in matched_text:
                # Truncate at the newline
                result.end = result.start + matched_text.index('\n')
            elif '\r' in matched_text:
                result.end = result.start + matched_text.index('\r')
                
            # Entity-specific post-processing
            if result.entity_type == "PERSON":
                # Trim trailing spaces if any
                while result.end > result.start and sanitized[result.end-1].isspace():
                    result.end -= 1
                    
                # Trim trailing possessives to ensure consistent token mapping
                current_match = sanitized[result.start:result.end].lower()
                if current_match.endswith("'s"):
                    result.end -= 2
                elif current_match.endswith("'"):
                    result.end -= 1
            
            # Only keep if it still has length
            if result.end > result.start:
                corrected_results.append(result)

        results = corrected_results

        results = self._remove_overlaps(results)
        results = sorted(results, key=lambda r: r.start, reverse=True)

        log = []
        safe_text = sanitized
        
        for result in results:
            real_value = sanitized[result.start:result.end]
            token = self.vault.store(result.entity_type, real_value)
            safe_text = safe_text[:result.start] + token + safe_text[result.end:]
            
            log.append({
                "original_value": real_value,
                "entity_type": result.entity_type,
                "token": token
            })
            
            self.audit.log_redaction(
                event_type=event_type,
                entity_type=result.entity_type,
                token=token,
                text_preview=safe_text[:50]
            )

        logger.info(f"Redacted {len(log)} PII entities ({event_type})")
        return safe_text, log

    def _remove_overlaps(self, results: list[RecognizerResult]) -> list[RecognizerResult]:
        """Remove overlapping results, keeping the highest scoring ones."""
        if not results:
            return []
        
        results = sorted(results, key=lambda r: (r.start, -r.score))
        
        filtered = []
        last_end = -1
        
        for result in results:
            if result.start >= last_end:
                filtered.append(result)
                last_end = result.end
        
        return filtered

    def redact_batch(self, texts: list[str], event_type: str = "batch") -> list[tuple[str, list[dict]]]:
        return [self.redact(text, event_type) for text in texts]

    def unmask(self, text: str) -> str:
        return self.vault.unmask_text(text)