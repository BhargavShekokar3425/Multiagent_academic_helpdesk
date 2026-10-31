"""Course-slot routing and observable lookup status without a live model."""
import asyncio
from types import SimpleNamespace
from google.genai import types

from src.single_agent import reset_turn, answer_policy, RESULT_KEY
from src.tools.routing import parse_slot_question, semester_label
from src.tools.schedule import get_course_slot


def context(question, semester=""):
    return SimpleNamespace(
        state={"requested_semester": semester},
        user_content=types.Content(role="user", parts=[types.Part(text=question)]))


def test_missing_semester_calls_slot_tool_and_clarifies():
    ctx = context("What is the slot for Autonomous Systems?")
    reset_turn(ctx)
    result = ctx.state[RESULT_KEY]
    assert result["answer"] == "Which academic year and semester?"
    assert result["tool"] == "course_slot"
    assert result["lookup_status"] == "ambiguous"
    assert ctx.state["temp:tool_calls"][0]["tool"] == "course_slot"


def test_semester_produces_specific_missing_record(monkeypatch):
    monkeypatch.setattr("src.tools.schedule.course_rows", lambda: [])
    result = get_course_slot("Autonomous Systems", "2026-27-S1")
    assert result["message"] == (
        "No verified timetable record for Autonomous Systems in 2026–27 Semester I.")
    assert result["status"] == "not_found"


def test_cli_context_question_is_parsed():
    ctx = context('Student context: {"semester": "2026-27-S1"}\nQuestion: What is the slot for Autonomous Systems?',
                  "2026-27-S1")
    reset_turn(ctx)
    assert ctx.state[RESULT_KEY]["tool"] == "course_slot"


def test_slot_parsing_does_not_capture_policy_query():
    assert parse_slot_question("What is the attendance policy?") is None
    assert parse_slot_question("What slot is CSL7570 in?") == "CSL7570"
    assert semester_label("2026-27-S2") == "2026–27 Semester II"


def test_end_to_end_session_preserves_debug_trace(monkeypatch):
    monkeypatch.setattr("src.tools.schedule.course_rows", lambda: [])
    # A model call here is an integration regression.
    async def forbidden(*args, **kwargs):
        raise AssertionError("Slot routing must not call the language model")
        yield
    from src.single_agent import baseline_agent
    monkeypatch.setattr(type(baseline_agent.model), "generate_content_async", forbidden)
    result = asyncio.run(answer_policy(
        "What is the slot for Autonomous Systems?",
        semester="2026-27-S1", debug=True))
    assert result["status"] == "insufficient_evidence"
    assert result["tool"] == "course_slot"
    assert result["lookup_status"] == "not_found"
    assert result["debug"]["tool_calls"][0]["semester"] == "2026-27-S1"
    assert result["human_review_log"]["status"] == "success"
