import pytest

from src.zero_trust_rag.config import Config
from src.zero_trust_rag.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    yield
    monkeypatch.delenv("ZT_CONFIG_PATH", raising=False)
    Config.reload()


def test_reads_dotted_keys_from_packaged_config():
    cfg = Config.reload()
    assert cfg.get("splitter.chunk_size") == 500
    assert "PERSON" in cfg.get("presidio.entity_types")


def test_missing_key_returns_default():
    cfg = Config.reload()
    assert cfg.get("presidio.no_such_key", "fallback") == "fallback"
    with pytest.raises(KeyError):
        cfg["presidio.no_such_key"]


def test_missing_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ZT_CONFIG_PATH", str(tmp_path / "absent.yaml"))
    with pytest.raises(ConfigurationError):
        Config.reload()


def test_empty_file_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    monkeypatch.setenv("ZT_CONFIG_PATH", str(path))
    with pytest.raises(ConfigurationError):
        Config.reload()


def test_override_path(tmp_path, monkeypatch):
    path = tmp_path / "custom.yaml"
    path.write_text("retrieval:\n  top_k: 7\n")
    monkeypatch.setenv("ZT_CONFIG_PATH", str(path))
    assert Config.reload().get("retrieval.top_k") == 7
