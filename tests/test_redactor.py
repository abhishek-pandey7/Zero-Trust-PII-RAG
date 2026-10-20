"""
Redaction tests. The core security property is that no detected value survives in the masked text,
and that unmasking restores the original exactly.
"""
import pytest
from presidio_analyzer import RecognizerResult

pytestmark = pytest.mark.slow

RECORD = (
    "EMPLOYEE RECORD #1042\n"
    "Name: Priya Sharma\n"
    "Email: priya.sharma@example.in\n"
    "Phone: +91-98765-43210\n"
    "PAN Card: ABCDE1234F\n"
    "Aadhaar: 5425-2144-4943\n"
    "SSN: 536-22-1874\n"
    "Card: 4111 1111 1111 1111"
)
SECRETS = [
    "Priya Sharma",
    "priya.sharma@example.in",
    "98765-43210",
    "ABCDE1234F",
    "5425-2144-4943",
    "536-22-1874",
    "4111 1111 1111 1111",
]


def test_no_secret_survives_in_masked_text(redactor):
    masked, log = redactor.redact(RECORD)
    for secret in SECRETS:
        assert secret not in masked
    assert len(log) >= len(SECRETS)


def test_unmask_restores_the_original(redactor):
    masked, _ = redactor.redact(RECORD)
    assert redactor.unmask(masked) == RECORD


def test_entities_do_not_swallow_newlines(redactor):
    masked, _ = redactor.redact("Name: John Doe\nEmail: john.doe@example.com")
    assert masked.count("\n") == 1
    assert masked.splitlines()[1].startswith("Email: <<EMAIL_ADDRESS_")


def test_possessive_is_kept_outside_the_token(redactor):
    masked, log = redactor.redact("Priya Sharma's account is overdrawn.")
    assert masked.endswith("'s account is overdrawn.")
    assert [e["original_value"] for e in log] == ["Priya Sharma"]


def test_same_person_in_two_documents_gets_one_token(redactor):
    a, _ = redactor.redact("Priya Sharma opened an account.")
    b, _ = redactor.redact("Transfer approved for Priya Sharma.")
    token = a.split(" ")[0]
    assert token in b


def test_text_without_pii_is_unchanged(redactor):
    text = "The portfolio review is attached."
    masked, log = redactor.redact(text)
    assert masked == text
    assert log == []


def test_relative_date_words_are_not_tokenized(redactor):
    text = "The quarterly report is attached."
    assert redactor.redact(text)[0] == text


def test_calendar_dates_are_still_tokenized(redactor):
    masked, _ = redactor.redact("DOB: 24-11-1987")
    assert "1987" not in masked


def test_control_characters_are_stripped():
    from src.zero_trust_rag.pii_redactor import PIIRedactor

    assert PIIRedactor.sanitize_input("a\x00b\x1fc\nd") == "abc\nd"


def test_aadhaar_is_labelled_as_aadhaar(redactor):
    _, log = redactor.redact("Aadhaar: 5425-2144-4943")
    assert [e["entity_type"] for e in log] == ["IN_AADHAAR"]


@pytest.mark.parametrize("order", [0, 1])
def test_specific_recognizer_beats_generic_ner_on_a_tie(redactor, order):
    """Presidio returns both IN_AADHAAR and DATE_TIME for '5425-2144-4943' with equal scores."""
    spans = [
        RecognizerResult("DATE_TIME", start=9, end=23, score=0.85),
        RecognizerResult("IN_AADHAAR", start=9, end=23, score=0.85),
    ]
    if order:
        spans.reverse()
    assert [r.entity_type for r in redactor._merge_overlaps(spans)] == ["IN_AADHAAR"]


def test_overlapping_spans_are_merged_so_no_flagged_character_leaks(redactor):
    """Keeping only one of two partly overlapping spans would leave the other's tail in clear text."""
    generic = RecognizerResult("DATE_TIME", start=0, end=12, score=0.9)
    specific = RecognizerResult("IN_AADHAAR", start=1, end=15, score=0.85)
    (merged,) = redactor._merge_overlaps([generic, specific])
    assert (merged.entity_type, merged.start, merged.end) == ("IN_AADHAAR", 0, 15)


def test_unformatted_ssn_with_context_is_caught(redactor):
    masked, _ = redactor.redact("Social Security Number 536221874")
    assert "536221874" not in masked


def test_bank_account_number_is_labelled_as_one(redactor):
    _, log = redactor.redact("Account number: 004512339871")
    assert [e["entity_type"] for e in log] == ["US_BANK_NUMBER"]


def test_bare_mobile_number_with_context_is_caught(redactor):
    assert "9876543210" not in redactor.redact("Mobile: 9876543210")[0]


def test_audit_log_never_contains_detected_values(redactor, tmp_path):
    redactor.redact(RECORD)
    audit_text = (tmp_path / "audit.log").read_text()
    for secret in SECRETS:
        assert secret not in audit_text
