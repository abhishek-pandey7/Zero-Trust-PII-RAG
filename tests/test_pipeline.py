"""
End-to-end pipeline tests with a recording fake LLM, so the security property can be checked on
exactly what would have left the process.
"""
import pytest

from src.zero_trust_rag.exceptions import ValidationError
from src.zero_trust_rag.pipeline import ZeroTrustRAG

from .fakes import HashingEmbedder, RecordingLLM

pytestmark = pytest.mark.slow

RECORDS = """CLIENT RECORD #2001
Name: Priya Sharma
Email: priya.sharma@example.in
SSN: 536-22-1874
Account number: 004512339871
Advisor: Rohan Mehra

---

CLIENT RECORD #2002
Name: Daniel Brooks
Email: dbrooks@example.com
SSN: 287-41-0954
Account number: 778201945563
Advisor: Rohan Mehra"""

SECRETS = [
    "Priya Sharma", "priya.sharma@example.in", "536-22-1874", "004512339871",
    "Daniel Brooks", "dbrooks@example.com", "287-41-0954", "778201945563", "Rohan Mehra",
]


@pytest.fixture
def llm():
    return RecordingLLM()


@pytest.fixture
def rag(tmp_path, redactor, vault, audit, llm):
    return ZeroTrustRAG(
        persist_directory=str(tmp_path / "chroma"),
        llm=llm,
        embedder=HashingEmbedder(),
        vault=vault,
        audit=audit,
        redactor=redactor,
    )


def test_ingest_stores_only_masked_text(rag):
    result = rag.ingest(RECORDS)
    stored = "\n".join(rag.collection.get()["documents"])
    assert result.chunks_created >= 1
    for secret in SECRETS:
        assert secret not in stored
        assert secret not in result.masked_text


def test_reingesting_the_same_text_does_not_duplicate_chunks(rag):
    first = rag.ingest(RECORDS)
    rag.ingest(RECORDS)
    assert rag.collection.count() == first.chunks_created


def test_nothing_sent_to_the_llm_contains_pii(rag, llm):
    rag.ingest(RECORDS)
    rag.ask("What is Priya Sharma's SSN?")
    rag.ask("Which client has account number 778201945563?")
    assert len(llm.calls) == 2
    for secret in SECRETS:
        assert secret not in llm.sent_text


def test_answer_is_unmasked_for_the_caller(rag, vault):
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874")
    ssn_token = next(t for t, v in [(m["token"], m["real_value"]) for m in vault.get_all_mappings()]
                     if v == "536-22-1874")
    rag.llm.reply = f"Her SSN is {ssn_token}."
    result = rag.ask("What is Priya Sharma's SSN?")
    assert result.answer == "Her SSN is 536-22-1874."
    assert result.masked_answer == f"Her SSN is {ssn_token}."
    assert "Priya Sharma" not in result.masked_question


def test_question_entities_are_reported(rag):
    rag.ingest(RECORDS)
    result = rag.ask("What is Priya Sharma's email?")
    assert [e["entity_type"] for e in result.question_entities] == ["PERSON"]


def test_asking_before_ingesting_does_not_call_the_llm(rag, llm):
    result = rag.ask("What is Priya Sharma's SSN?")
    assert result.context_chunks == 0
    assert llm.calls == []


def test_empty_input_is_rejected(rag):
    with pytest.raises(ValidationError):
        rag.ingest("   ")
    with pytest.raises(ValidationError):
        rag.ask("")


def test_reset_clears_stored_chunks(rag):
    rag.ingest(RECORDS)
    rag.reset_vector_db()
    assert rag.collection.count() == 0
