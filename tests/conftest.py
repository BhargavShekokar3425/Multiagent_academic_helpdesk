"""Keep offline tests from writing real unanswered-question logs."""
import pytest


@pytest.fixture(autouse=True)
def isolate_application_log(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test_helpdesk.sqlite3"))
