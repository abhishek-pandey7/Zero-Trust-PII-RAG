import json

from src.zero_trust_rag.audit_logger import AuditLogger


def test_events_are_json_lines(audit, tmp_path):
    audit.log_operation("ingest", "success", {"chunks_created": 3})
    audit.log_redaction("ingest", "PERSON", "<<PERSON_01_abcd>>")

    lines = (tmp_path / "audit.log").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["event_type"] for e in events] == ["ingest", "ingest"]
    assert events[0]["details"] == {"chunks_created": 3}


def test_recent_logs_are_newest_first(audit):
    for i in range(5):
        audit.log_operation("op", "success", {"i": i})
    assert [e["details"]["i"] for e in audit.get_recent_logs(limit=3)] == [4, 3, 2]


def test_each_logger_writes_to_its_own_file(tmp_path):
    a = AuditLogger(str(tmp_path / "a.log"))
    b = AuditLogger(str(tmp_path / "b.log"))
    a.log_operation("from_a", "success")
    b.log_operation("from_b", "success")
    assert "from_b" in (tmp_path / "b.log").read_text()
    assert "from_b" not in (tmp_path / "a.log").read_text()
