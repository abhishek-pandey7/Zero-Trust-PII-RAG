"""
egress.py – The last check before anything leaves the trust boundary.

NER-based redaction has no recall guarantee: a lowercase name or an odd format can slip through.
EgressGuard runs on the exact text about to be sent and does two things:

1. Known-value sweep. Any value the vault already holds that still appears in clear text (say a
   name tokenized in one document but missed in this question) is replaced by its token.
2. Hard-identifier scan. If anything that still looks like an SSN, card number, Aadhaar, PAN,
   email or phone number remains outside a token, the request is blocked, not sent.

What it can't catch: a never-seen-before value with no recognisable shape (a new name the NER
model missed). That's the residual risk, and why the audit trail records every egress.
"""
import logging
import re
from dataclasses import dataclass

from .audit_logger import AuditLogger
from .exceptions import EgressBlockedError
from .vault import TOKEN_PATTERN, PIIVault

logger = logging.getLogger(__name__)

# Values this short ("Rao", "UK") would match ordinary words; they are left to the redactor
MIN_SWEEP_LENGTH = 4


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# Shapes that are never acceptable in clear text on the way out
HARD_IDENTIFIERS: dict[str, re.Pattern] = {
    "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IN_AADHAAR": re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b"),
    "IN_PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "PHONE_NUMBER": re.compile(r"(?:\+91[-\s]?\d{5}[-\s]?\d{5}|\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b)"),
}
CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


@dataclass
class EgressReport:
    text: str
    remasked: int


class EgressGuard:
    def __init__(self, vault: PIIVault, audit: AuditLogger):
        self.vault = vault
        self.audit = audit
        self._sweep_version = -1
        self._sweep_pattern: re.Pattern | None = None
        self._sweep_tokens: dict[str, str] = {}

    def _sweep(self) -> tuple[re.Pattern | None, dict[str, str]]:
        """A regex over every known value, rebuilt only when the vault has grown."""
        version = sum(self.vault.get_stats().values())
        if version != self._sweep_version:
            pairs = [(v, t) for v, t in self.vault.known_values() if len(v) >= MIN_SWEEP_LENGTH]
            pairs.sort(key=lambda p: len(p[0]), reverse=True)  # longest first: "Priya Sharma" before "Priya"
            self._sweep_tokens = {v.lower(): t for v, t in pairs}
            self._sweep_pattern = (
                re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v, _ in pairs) + r")(?!\w)", re.IGNORECASE)
                if pairs else None
            )
            self._sweep_version = version
        return self._sweep_pattern, self._sweep_tokens

    @staticmethod
    def find_hard_identifiers(text: str) -> list[str]:
        """Entity types of identifier-shaped strings left outside tokens."""
        bare = TOKEN_PATTERN.sub(" ", text)
        found = [name for name, pattern in HARD_IDENTIFIERS.items() if pattern.search(bare)]
        for match in CARD_CANDIDATE.finditer(bare):
            digits = re.sub(r"\D", "", match.group())
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                found.append("CREDIT_CARD")
                break
        return found

    def remask_known_values(self, text: str) -> tuple[str, int]:
        """Replace any clear-text value the vault already holds with its token."""
        pattern, tokens = self._sweep()
        if pattern is None:
            return text, 0
        return pattern.subn(lambda m: tokens[m.group().lower()], text)

    def enforce(self, text: str, purpose: str) -> EgressReport:
        """Return text that is safe to send, or raise EgressBlockedError."""
        text, remasked = self.remask_known_values(text)

        leftovers = self.find_hard_identifiers(text)
        if leftovers:
            self.audit.log_operation("egress", "blocked", {"purpose": purpose, "entity_types": leftovers})
            logger.warning("Blocked egress for %s: %s still in clear text", purpose, leftovers)
            raise EgressBlockedError(
                "Blocked: the request still contained identifiers after masking "
                f"({', '.join(leftovers)}). Nothing was sent."
            )

        self.audit.log_operation("egress", "allowed", {
            "purpose": purpose, "chars": len(text), "remasked_known_values": remasked,
        })
        return EgressReport(text, remasked)
