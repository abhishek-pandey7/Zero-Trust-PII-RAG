import os

import pytest

os.environ["ZT_EAGER_INIT"] = "0"  # don't load models in a background thread during tests

import app as web  # noqa: E402


@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    return web.app.test_client()


def test_health_reports_version_and_mode(client, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef")
    body = client.get("/api/health").get_json()
    assert body["status"] == "healthy"
    assert body["version"] == "0123456"
    assert body["async_mode"] == "threading"


def test_policies_show_the_real_threshold(client):
    security = client.get("/api/policies").get_json()["policies"]["security"]
    assert security["score_threshold"] == 0.4
    assert "US_BANK_NUMBER" in security["entity_types"]


@pytest.mark.parametrize("path, body", [("/api/ingest", {"data": "  "}), ("/api/ask", {"question": ""})])
def test_empty_input_is_a_400(client, path, body):
    assert client.post(path, json=body).status_code == 400


def test_oversized_input_is_refused_with_a_clear_message(client, monkeypatch):
    monkeypatch.setenv("ZT_MAX_INPUT_CHARS", "50")
    response = client.post("/api/ask", json={"question": "x" * 60})
    assert response.status_code == 400
    assert "accepts up to 50" in response.get_json()["error"]
