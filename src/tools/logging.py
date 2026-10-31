"""Local unanswered-question log, created only when explicitly invoked."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from ._records import ROOT, response


def log_unanswered(question: str, category: str, reason: str,
                   agent_name: str = "academic_helpdesk_baseline") -> dict:
    """Record an unanswered question for human review; does not contact anyone."""
    if not question.strip():
        return response("error", message="Question must not be empty.")
    try:
        load_dotenv(ROOT / ".env")
        path = Path(os.getenv("DATABASE_PATH", "data/helpdesk.sqlite3"))
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS unanswered_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
                    timestamp TEXT NOT NULL, category TEXT NOT NULL, reason TEXT NOT NULL,
                    agent_name TEXT NOT NULL, status TEXT NOT NULL
                )
            """)
            cursor = connection.execute(
                "INSERT INTO unanswered_questions "
                "(question,timestamp,category,reason,agent_name,status) VALUES (?,?,?,?,?,?)",
                (question, datetime.now(timezone.utc).isoformat(), category, reason,
                 agent_name, "pending_review"))
            row_id = cursor.lastrowid
        return response("success", {"record_id": row_id, "review_status": "pending_review"})
    except (OSError, sqlite3.Error) as exc:
        return response("error", message=str(exc))
