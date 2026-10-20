import pytest

from src.zero_trust_rag.egress import EgressGuard
from src.zero_trust_rag.exceptions import EgressBlockedError
from src.zero_trust_rag.pipeline import ZeroTrustRAG

from .fakes import HashingEmbedder, RecordingLLM


@pytest.fixture
def guard(vault, audit):
    return EgressGuard(vault, audit)


class NoOpRedactor:
    """Simulates a total redaction failure: every value goes through untouched."""

    def __init__(self, vault):
        self.vault = vault

    def redact(self, text, event_type="ingest"):
        return text.strip(), []

    def unmask(self, text):
        return self.vault.unmask_text(text)


@pytest.mark.parametrize("leak, entity", [
    ("SSN 536-22-1874", "US_SSN"),
    ("mail priya@example.in", "EMAIL_ADDRESS"),
    ("Aadhaar 5425 2144 4943", "IN_AADHAAR"),
    ("PAN ABCDE1234F", "IN_PAN"),
    ("call +91-98765-43210", "PHONE_NUMBER"),
    ("card 4111 1111 1111 1111", "CREDIT_CARD"),
])
def test_hard_identifiers_block_the_request(guard, leak, entity):
    with pytest.raises(EgressBlockedError, match=entity):
        guard.enforce(f"Context: {leak}\n\nQuestion: what is it?", "test")


def test_numbers_that_fail_luhn_are_not_cards(guard):
    assert guard.enforce("Order 4111111111111112 shipped", "test").text.startswith("Order")


def test_identifiers_inside_tokens_are_fine(guard):
    text = "Context: <<US_SSN_01_ab12>> belongs to <<PERSON_01_cd34>>"
    assert guard.enforce(text, "test").text == text


def test_known_values_are_remasked_case_insensitively(guard, vault):
    token = vault.store("PERSON", "Priya Sharma")
    report = guard.enforce("Question: what does priya sharma own?", "test")
    assert report.text == f"Question: what does {token} own?"
    assert report.remasked == 1


def test_longest_known_value_wins(guard, vault):
    full = vault.store("PERSON", "Priya Sharma")
    vault.store("PERSON", "Priya")
    assert guard.enforce("Ask Priya Sharma", "test").text == f"Ask {full}"


def test_known_values_only_match_whole_words(guard, vault):
    vault.store("LOCATION", "Pune")
    assert guard.enforce("Punesh called", "test").text == "Punesh called"


def test_sweep_picks_up_values_added_after_first_use(guard, vault):
    guard.enforce("warm", "test")
    token = vault.store("PERSON", "Rohan Mehra")
    assert guard.enforce("rohan mehra", "test").text == token


def test_blocked_and_allowed_egress_are_audited(guard, audit):
    guard.enforce("nothing sensitive", "test")
    with pytest.raises(EgressBlockedError):
        guard.enforce("SSN 536-22-1874", "test")
    statuses = [e["status"] for e in audit.get_recent_logs() if e["event_type"] == "egress"]
    assert statuses == ["blocked", "allowed"]
    assert "536-22-1874" not in str(audit.get_recent_logs())


@pytest.mark.slow
def test_total_redaction_failure_still_sends_no_hard_identifiers(tmp_path, vault, audit):
    """Even with redaction switched off entirely, an SSN never reaches the LLM."""
    llm = RecordingLLM()
    rag = ZeroTrustRAG(
        persist_directory=str(tmp_path / "chroma"),
        llm=llm, embedder=HashingEmbedder(), vault=vault, audit=audit, redactor=NoOpRedactor(vault),
    )
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874")
    with pytest.raises(EgressBlockedError):
        rag.ask("What is the SSN?")
    assert llm.calls == []


@pytest.mark.slow
def test_lowercase_name_missed_by_ner_is_remasked_before_sending(rag_with_llm):
    rag, llm = rag_with_llm
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874")
    rag.ask("what is priya sharma's ssn?")
    assert "priya sharma" not in llm.sent_text.lower()


@pytest.fixture
def rag_with_llm(tmp_path, redactor, vault, audit):
    llm = RecordingLLM()
    rag = ZeroTrustRAG(
        persist_directory=str(tmp_path / "chroma"),
        llm=llm, embedder=HashingEmbedder(), vault=vault, audit=audit, redactor=redactor,
    )
    return rag, llm
