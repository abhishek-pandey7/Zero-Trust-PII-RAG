import re
import threading

from src.zero_trust_rag.vault import PIIVault

TOKEN_RE = re.compile(r"^<<[A-Z_]+_\d{2,}_[0-9a-f]{4}>>$")


def test_store_returns_an_opaque_token(vault):
    token = vault.store("PERSON", "Priya Sharma")
    assert TOKEN_RE.match(token)
    assert "Priya" not in token


def test_same_value_always_gets_the_same_token(vault):
    """Deterministic tokens are what keep retrieval and unmasking working across documents."""
    first = vault.store("PERSON", "Priya Sharma")
    second = vault.store("PERSON", "Priya Sharma")
    assert first == second


def test_different_values_get_different_tokens(vault):
    assert vault.store("PERSON", "Priya Sharma") != vault.store("PERSON", "Rohan Mehra")


def test_resolve_and_unmask_round_trip(vault):
    name = vault.store("PERSON", "Priya Sharma")
    email = vault.store("EMAIL_ADDRESS", "priya@example.in")
    masked = f"{name} can be reached at {email}."
    assert vault.resolve(name) == "Priya Sharma"
    assert vault.unmask_text(masked) == "Priya Sharma can be reached at priya@example.in."


def test_unknown_token_resolves_to_none(vault):
    assert vault.resolve("<<PERSON_99_ffff>>") is None


def test_tokens_and_counters_survive_a_restart(tmp_path):
    path = str(tmp_path / "vault.db")
    token = PIIVault(path).store("PERSON", "Priya Sharma")

    reopened = PIIVault(path)
    assert reopened.resolve(token) == "Priya Sharma"
    assert reopened.store("PERSON", "Priya Sharma") == token
    assert reopened.get_stats() == {"PERSON": 1}


def test_concurrent_stores_of_one_value_mint_one_token(vault):
    tokens = []

    def worker():
        tokens.append(vault.store("US_SSN", "536-22-1874"))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(tokens)) == 1
    assert vault.get_stats() == {"US_SSN": 1}
