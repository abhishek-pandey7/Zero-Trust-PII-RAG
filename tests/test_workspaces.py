"""Workspace isolation: nothing one workspace stores can be read, retrieved or unmasked from another."""
import sqlite3

import pytest

from src.zero_trust_rag.exceptions import VaultError
from src.zero_trust_rag.pipeline import ZeroTrustRAG
from src.zero_trust_rag.vault import PIIVault

from .fakes import HashingEmbedder, RecordingLLM

A, B = "workspace-a", "workspace-b"


def test_tokens_are_per_workspace(vault):
    token_a = vault.store("PERSON", "Priya Sharma", A)
    vault.store("PERSON", "Priya Sharma", B)
    assert vault.resolve(token_a, A) == "Priya Sharma"
    assert vault.resolve(token_a, B) is None


def test_unmask_ignores_another_workspaces_tokens(vault):
    token_a = vault.store("US_SSN", "536-22-1874", A)
    assert vault.unmask_text(f"SSN is {token_a}", B) == f"SSN is {token_a}"


def test_stats_and_known_values_are_scoped(vault):
    vault.store("PERSON", "Priya Sharma", A)
    vault.store("PERSON", "Rohan Mehra", A)
    vault.store("EMAIL_ADDRESS", "b@example.com", B)
    assert vault.get_stats(A) == {"PERSON": 2}
    assert [v for v, _ in vault.known_values(B)] == ["b@example.com"]


def test_deleting_a_workspace_erases_only_its_values(vault):
    token_a = vault.store("PERSON", "Priya Sharma", A)
    token_b = vault.store("PERSON", "Rohan Mehra", B)
    assert vault.delete_workspace(A) == 1
    assert vault.resolve(token_a, A) is None
    assert vault.get_stats(A) == {}
    assert vault.resolve(token_b, B) == "Rohan Mehra"


def test_stale_workspaces_are_found_by_last_use(vault, tmp_path):
    vault.touch(A)
    vault.touch(B)
    with sqlite3.connect(tmp_path / "vault.db") as conn:
        conn.execute("UPDATE workspaces SET last_seen_at = datetime('now', '-3 hours') WHERE workspace_id = ?", (A,))
    assert vault.stale_workspaces(idle_hours=2) == [A]


def test_a_pre_workspace_vault_is_refused_not_dropped(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE value_mappings (real_value TEXT PRIMARY KEY, entity_type TEXT, token TEXT)")
    with pytest.raises(VaultError, match="older vault schema"):
        PIIVault(str(path))


@pytest.fixture
def rag(tmp_path, redactor, vault, audit):
    return ZeroTrustRAG(
        persist_directory=str(tmp_path / "chroma"),
        llm=RecordingLLM(), embedder=HashingEmbedder(), vault=vault, audit=audit, redactor=redactor,
    )


@pytest.mark.slow
def test_another_workspace_has_nothing_to_retrieve(rag):
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874", workspace=A)
    result = rag.ask("What is Priya Sharma's SSN?", workspace=B)
    assert result.context_chunks == 0
    assert rag.llm.calls == []


@pytest.mark.slow
def test_retrieval_only_returns_the_callers_chunks(rag):
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874", workspace=A)
    rag.ingest("Name: Rohan Mehra\nSSN: 287-41-0954", workspace=B)
    rag.ask("What is the SSN?", workspace=B)
    (_, prompt), = rag.llm.calls
    b_ssn = next(t for v, t in rag.vault.known_values(B) if v == "287-41-0954")
    assert prompt.count("<<US_SSN_") == 1 and b_ssn in prompt
    for _, token in rag.vault.known_values(A):
        assert token not in prompt


@pytest.mark.slow
def test_answers_never_unmask_into_another_workspaces_values(rag):
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874", workspace=A)
    rag.ingest("Name: Rohan Mehra\nSSN: 287-41-0954", workspace=B)
    a_token = next(t for v, t in rag.vault.known_values(A) if v == "536-22-1874")
    rag.llm.reply = f"It is {a_token}."  # a model that somehow echoes another workspace's token
    assert rag.ask("What is the SSN?", workspace=B).answer == f"It is {a_token}."


@pytest.mark.slow
def test_erasing_a_workspace_removes_its_chunks_and_values(rag):
    rag.ingest("Name: Priya Sharma\nSSN: 536-22-1874", workspace=A)
    rag.ingest("Name: Rohan Mehra\nSSN: 287-41-0954", workspace=B)
    erased = rag.delete_workspace(A)
    assert erased["chunks_deleted"] == 1 and erased["values_deleted"] >= 2
    assert rag.get_stats(A) == {"vault_stats": {}, "chunks_stored": 0}
    assert rag.get_stats(B)["chunks_stored"] == 1


@pytest.mark.slow
def test_audit_events_can_be_filtered_by_workspace(rag, audit):
    rag.ingest("Name: Priya Sharma", workspace=A)
    rag.ingest("Name: Rohan Mehra", workspace=B)
    assert {e["workspace"] for e in audit.get_recent_logs(workspace=A)} == {A}
