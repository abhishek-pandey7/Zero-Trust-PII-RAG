import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.zero_trust_rag.audit_logger import AuditLogger  # noqa: E402
from src.zero_trust_rag.vault import PIIVault  # noqa: E402


@pytest.fixture
def vault(tmp_path):
    return PIIVault(db_path=str(tmp_path / "vault.db"))


@pytest.fixture
def audit(tmp_path):
    return AuditLogger(log_path=str(tmp_path / "audit.log"))


@pytest.fixture(scope="session")
def _shared_redactor(tmp_path_factory):
    """Building the Presidio analyzer loads spaCy (~2 s), so build it once per test run."""
    from src.zero_trust_rag.pii_redactor import PIIRedactor

    base = tmp_path_factory.mktemp("redactor")
    return PIIRedactor(PIIVault(str(base / "vault.db")), AuditLogger(str(base / "audit.log")))


@pytest.fixture
def redactor(_shared_redactor, vault, audit):
    """The shared analyzer, wired to a fresh vault and audit log for each test."""
    _shared_redactor.vault = vault
    _shared_redactor.audit = audit
    return _shared_redactor
