"""
pii_redactor.py – Uses Microsoft Presidio to detect and mask PII entities.

Every detected span is replaced with a vault token before the text goes anywhere else. The rules
that decide what counts as a span lean toward recall: when two recognizers disagree about where an
entity starts or ends, the union of their spans is masked, because an unmasked character is a leak
and an over-masked one only costs some answer quality.
"""
import logging
import re

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
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

SCORE_THRESHOLD = config.get("presidio.score_threshold", 0.4)

# Longest piece of text handed to spaCy at once (see PIIRedactor._blocks)
BLOCK_CHARS = 4000

# When recognizers claim overlapping text, the label comes from the most specific one. A checksum-
# or format-backed identifier (SSN, Aadhaar, account number) outranks a name, which outranks the
# generic NER labels that fire on almost any number or place.
GENERIC_ENTITIES = {"DATE_TIME", "LOCATION", "NRP", "URL"}


YEARS_ONLY = re.compile(r"^\s*(?:19|20)\d{2}(?:\s*[-–—]\s*(?:(?:19|20)\d{2}|present))?\s*$", re.IGNORECASE)
DURATION = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:h|hrs?|hours?|m|mins?|minutes?|s|secs?|seconds?|d|days?|wks?|weeks?|"
    r"mos?|months?|yrs?|years?)\s*$",
    re.IGNORECASE,
)


def _looks_like_abbreviation(span: str) -> bool:
    """True for 'B.Tech' or 'Node.js': one word, no scheme, no 'www.', no path, and a capital letter.
    People write real domains in lowercase ('example.com'), and those are still masked."""
    s = span.strip()
    return (
        " " not in s and "/" not in s and "@" not in s
        and not s.lower().startswith(("http", "www."))
        and any(c.isupper() for c in s)
    )


def _priority(entity_type: str) -> int:
    if entity_type in GENERIC_ENTITIES:
        return 0
    if entity_type == "PERSON":
        return 1
    return 2


class AadhaarRecognizer(PatternRecognizer):
    """Indian Aadhaar numbers written in three groups of four digits."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_AADHAAR",
            patterns=[Pattern("Aadhaar grouped", r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b", 0.85)],
            context=["aadhaar", "aadhar", "uid", "uidai"],
        )


class IndianPhoneRecognizer(PatternRecognizer):
    """Indian mobile numbers. A bare 10-digit run only counts when a word like 'phone' is nearby."""

    def __init__(self):
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern("India phone +91", r"\+91[-\s]?[6-9]\d{4}[-\s]?\d{5}\b", 0.7),
                Pattern("India phone split", r"\b[6-9]\d{4}[-\s]\d{5}\b", 0.6),
                # 0.3 is below the threshold on its own; the context boost (+0.35) lifts it over
                Pattern("India phone 10 digit", r"\b[6-9]\d{9}\b", 0.3),
            ],
            context=["phone", "mobile", "cell", "contact", "tel", "whatsapp"],
        )


class PANRecognizer(PatternRecognizer):
    """Indian Permanent Account Numbers: five letters, four digits, one letter."""

    def __init__(self):
        super().__init__(
            supported_entity="IN_PAN",
            patterns=[Pattern("PAN format", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.8)],
            context=["pan", "permanent account"],
        )


# Labelled fields in structured records ("Name: Amit Sharma"). The label is stronger evidence than
# any NER model: spaCy's small model misses plenty of Indian and uncommon names, and an address or
# date of birth on a labelled line is PII whatever its shape.
FIELD_LABELS = {
    "PERSON": [
        "name", "full name", "client", "client name", "customer", "customer name", "account holder",
        "joint holder", "joint account holder", "beneficiary", "nominee", "advisor", "financial advisor",
        "advisor name", "representative", "rep", "employee", "employee name", "manager", "spouse",
        "emergency contact", "contact person", "primary contact", "guardian", "trustee", "reports to",
    ],
    "LOCATION": ["address", "mailing address", "home address", "residential address", "street address"],
    "DATE_TIME": ["dob", "d.o.b", "d.o.b.", "date of birth", "birth date", "birthdate"],
}
_LABEL_TO_ENTITY = {label: entity for entity, labels in FIELD_LABELS.items() for label in labels}
_LABEL_ALTERNATION = "|".join(sorted((re.escape(lbl) for lbl in _LABEL_TO_ENTITY), key=len, reverse=True))
FIELD_LINE = re.compile(
    rf"^[ \t\-*•]*(?P<label>{_LABEL_ALTERNATION})[ \t]*[:=][ \t]*(?P<value>[^\r\n]+?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# A name is up to five words of letters, apostrophes, dots and hyphens; stop at anything else
NAME_VALUE = re.compile(r"[A-Za-z][A-Za-z'.\-]+(?:[ \t]+[A-Za-z][A-Za-z'.\-]*){0,4}")


class LabelledFieldRecognizer(EntityRecognizer):
    """Names, addresses and dates of birth that follow a field label on their own line."""

    def __init__(self):
        super().__init__(supported_entities=list(FIELD_LABELS), name="LabelledFieldRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts=None) -> list[RecognizerResult]:
        results = []
        for match in FIELD_LINE.finditer(text):
            entity = _LABEL_TO_ENTITY[match.group("label").lower()]
            if entity not in entities:
                continue
            start, end = match.span("value")
            if entity == "PERSON":
                name = NAME_VALUE.match(text, start, end)
                if not name or name.end() - start < 2:
                    continue  # "Name: N/A", "Advisor: -"
                end = name.end()
            results.append(RecognizerResult(
                entity, start, end, 0.9,
                recognition_metadata={
                    RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                },
            ))
        return results


class PIIRedactor:
    """
    Detects PII in text and replaces each hit with a Vault-managed token.
    Supports audit logging for compliance.
    """

    def __init__(self, vault: PIIVault, audit_logger: AuditLogger | None = None):
        self.vault = vault
        self.audit = audit_logger or AuditLogger()

        # Pass the engine in: AnalyzerEngine() on its own loads en_core_web_lg (and downloads it if
        # it is missing), which costs ~600 MB before being thrown away.
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self.analyzer.registry.add_recognizer(AadhaarRecognizer())
        self.analyzer.registry.add_recognizer(IndianPhoneRecognizer())
        self.analyzer.registry.add_recognizer(PANRecognizer())
        self.analyzer.registry.add_recognizer(LabelledFieldRecognizer())

    @staticmethod
    def sanitize_input(text: str) -> str:
        if not text:
            return ""
        sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        sanitized = re.sub(r'[\x80-\x9f]', '', sanitized)
        return sanitized.strip()

    @staticmethod
    def _blocks(text: str, max_chars: int = BLOCK_CHARS):
        """Yield (offset, block) pieces of at most ~max_chars, cut only at line breaks.

        spaCy's memory grows with the length of the document it parses (~100 MB for 25,000
        characters), so long inputs are analysed a block at a time. Cutting at line breaks can't
        split an entity we'd keep: _trim already ends every span at its first line break.
        """
        start = 0
        while start < len(text):
            end = start + max_chars
            if end < len(text):
                cut = text.rfind("\n", start, end)
                end = cut + 1 if cut > start else text.find("\n", end) + 1 or len(text)
            else:
                end = len(text)
            yield start, text[start:end]
            start = end

    def detect(self, text: str) -> list[RecognizerResult]:
        """Find PII spans in already-sanitized text, cleaned up and free of overlaps."""
        results: list[RecognizerResult] = []
        for offset, block in self._blocks(text):
            for r in self.analyzer.analyze(
                text=block,
                entities=ENTITY_TYPES,
                language="en",
                score_threshold=SCORE_THRESHOLD,
            ):
                results.append(RecognizerResult(r.entity_type, r.start + offset, r.end + offset, r.score))
        cleaned = [r for r in (self._trim(text, r) for r in results) if r is not None]
        return self._merge_overlaps(cleaned)

    @staticmethod
    def _trim(text: str, result: RecognizerResult) -> RecognizerResult | None:
        """Tighten a span to the entity itself, or drop it if nothing identifying is left."""
        start, end = result.start, result.end

        # Structured records put one field per line; NER sometimes runs a name into the next line
        line_break = re.search(r"[\r\n]", text[start:end])
        if line_break:
            end = start + line_break.start()
        while end > start and text[end - 1].isspace():
            end -= 1
        # An unbalanced bracket belongs to the sentence, not the entity: "(+91-98700-12345" -> number
        if end > start and text[start] == "(" and ")" not in text[start:end]:
            start += 1
        if end > start and text[end - 1] == ")" and "(" not in text[start:end]:
            end -= 1

        if result.entity_type == "PERSON":
            # "Priya Sharma's" and "Priya Sharma" must map to the same token
            if text[start:end].lower().endswith("'s"):
                end -= 2
            elif text[start:end].endswith("'"):
                end -= 1

        if result.entity_type == "DATE_TIME" and not re.search(r"\d", text[start:end]):
            # "quarterly", "last year", "Monday": relative dates identify no one
            return None

        if result.entity_type == "DATE_TIME" and (YEARS_ONLY.match(text[start:end]) or DURATION.match(text[start:end])):
            # "2026", "2024 – 2028", "24h": a bare year or a duration identifies no one (HIPAA Safe
            # Harbor keeps years too). Dates with a month or a day are still masked.
            return None

        if result.entity_type == "URL" and _looks_like_abbreviation(text[start:end]):
            # "B.Tech", "Ph.D", "Node.js": .tech and .js look like domains, but these aren't addresses
            return None

        if text[start:end].lower() in _LABEL_TO_ENTITY and text[end:end + 2].lstrip(" \t")[:1] in (":", "="):
            # A field label ("Client Name:") is structure, not data; NER sometimes tags it as a name
            return None

        if end <= start:
            return None
        return RecognizerResult(result.entity_type, start, end, result.score)

    @staticmethod
    def _merge_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
        """Merge overlapping spans into one span covering all of them.

        Masking the union means no character any recognizer flagged is left in clear text. The
        merged span takes the most specific label (see _priority), then the highest score.
        """
        merged: list[RecognizerResult] = []
        for r in sorted(results, key=lambda r: (r.start, r.end)):
            if merged and r.start < merged[-1].end:
                current = merged[-1]
                best = max(current, r, key=lambda x: (_priority(x.entity_type), x.score))
                merged[-1] = RecognizerResult(
                    best.entity_type, current.start, max(current.end, r.end), best.score
                )
            else:
                merged.append(r)
        return merged

    def redact(self, text: str, event_type: str = "ingest") -> tuple[str, list[dict]]:
        sanitized = self.sanitize_input(text)
        if not sanitized:
            return text, []

        spans = self.detect(sanitized)

        # Build the output in one pass; slicing a new string per entity is quadratic on big documents
        pieces: list[str] = []
        log: list[dict] = []
        cursor = 0
        for span in spans:
            real_value = sanitized[span.start:span.end]
            token = self.vault.store(span.entity_type, real_value)
            pieces.append(sanitized[cursor:span.start])
            pieces.append(token)
            cursor = span.end
            log.append({"original_value": real_value, "entity_type": span.entity_type, "token": token})
        pieces.append(sanitized[cursor:])

        # Audit after masking is complete and record only tokens: the audit trail must never hold
        # the values it exists to protect
        for entry in log:
            self.audit.log_redaction(event_type, entry["entity_type"], entry["token"])

        logger.info("Redacted %d PII entities (%s)", len(log), event_type)
        return "".join(pieces), log

    def redact_batch(self, texts: list[str], event_type: str = "batch") -> list[tuple[str, list[dict]]]:
        return [self.redact(text, event_type) for text in texts]

    def anonymize(self, text: str) -> tuple[str, dict[str, int]]:
        """Replace each detected value with its type label, e.g. '[US_SSN]'. Irreversible.

        Unlike redact(), nothing is written to the vault: there is no token to resolve later, so the
        output can leave the organisation. Returns the text and a count per entity type.
        """
        sanitized = self.sanitize_input(text)
        pieces: list[str] = []
        counts: dict[str, int] = {}
        cursor = 0
        for span in self.detect(sanitized):
            pieces.append(sanitized[cursor:span.start])
            pieces.append(f"[{span.entity_type}]")
            counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
            cursor = span.end
        pieces.append(sanitized[cursor:])
        return "".join(pieces), counts

    def unmask(self, text: str) -> str:
        return self.vault.unmask_text(text)
