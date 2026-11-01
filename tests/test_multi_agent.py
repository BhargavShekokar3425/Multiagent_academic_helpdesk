"""Offline integration through the real ADK Runner and synthetic tool records."""
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src import single_agent as baseline
from src.agent import build_root_agent
from src.orchestration import build_orchestrator, route_request, validate_handoff
from src.agents.common import guard_tool_arguments, REQUEST_KEY
from src.tools.agent_tools import KEY
from src.tools._records import response

SEMESTER = "2026-27-S1"
MIXED = ("I missed my Major exam due to illness. Explain the applicable makeup-exam "
         "policy and draft an email to the instructor.")
TEXT = "A student who missed the major examination due to illness may apply for a makeup examination."
EVIDENCE = [{"chunk_id": "test-chunk", "document_id": "test-policy", "page_start": 1,
             "text": TEXT, "source_url": "https://example.invalid/policy",
             "needs_review": False, "policy_status": "unknown"}]


@pytest.fixture
def records(monkeypatch):
    from src.tools import schedule, faculty
    source = {"source_doc_id": "synthetic", "source_page": "1"}
    course = dict(source, course_code="TEST401", course_name="Test Systems",
                  semester=SEMESTER, department="Humanities", level="400",
                  course_type="Elective", offered="true")
    monkeypatch.setattr(schedule, "course_rows", lambda: [course, dict(course), dict(course, course_code="TEST402")])
    monkeypatch.setattr(schedule, "timetable_rows", lambda: [])
    monkeypatch.setattr(schedule, "coverage_metadata", lambda semester: {})
    def lookup(code, semester):
        if not code or not semester:
            return response("ambiguous", message="Specify course and semester.")
        if code == "TEST401" and semester == SEMESTER:
            row = dict(source, course_code=code, semester=semester,
                       instructor_name="Synthetic Instructor", email="teacher@example.invalid")
            return response("success", {"instructors": [row]}, [{"document_id": "synthetic", "page": "1"}])
        return response("not_found", {"instructors": []})
    monkeypatch.setattr(faculty, "lookup_instructor", lookup)
    monkeypatch.setattr("src.tools.email.lookup_instructor", lookup)


@pytest.fixture
def model(monkeypatch):
    """Mock only model inference; ADK callbacks, tools, sessions and Runner are real."""
    options = {"tool": None, "arguments": {}, "fail": set(), "empty_policy": False}
    monkeypatch.setattr(baseline, "search_policy", lambda *a, **k: {
        "status": "success", "evidence": [] if options["empty_policy"] else EVIDENCE,
        "data": {}, "warnings": []})

    async def generate(self, llm_request, **kwargs):
        instruction = str(llm_request.config.system_instruction)
        role = ("policy" if "SPECIALIST ROLE" in instruction else
                "email" if "Handle only the email part" in instruction else "schedule")
        if role in options["fail"]:
            raise RuntimeError("Synthetic specialist failure")
        names = {"policy": {"retrieve_policy"}, "email": {"email_draft", "instructor", "course_lookup"},
                 "schedule": {"course_slot", "courses_in_slot", "elective_count", "instructor",
                              "course_lookup", "semester_offerings"}}[role]
        replies = [p.function_response for content in llm_request.contents
                   for p in (content.parts or []) if p.function_response and p.function_response.name in names]
        if not replies:
            name = ("course_lookup" if role == "email" and options.get("resolve_first") else
                    "retrieve_policy" if role == "policy" else options["tool"] or "email_draft")
            args = ({"query": "makeup examination policy"} if role == "policy" else
                    {"course_code": "TEST401", "semester": SEMESTER, "request_type": "sick_leave",
                     **options["arguments"]})
            if name == "course_lookup" and options.get("resolve_first"):
                args = {"course": "Test Systems", "semester": SEMESTER}
            yield LlmResponse(content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(name=name, args=args, id="test-call"))]))
        else:
            reply = replies[-1].response
            if role == "email" and options.get("resolve_first") and replies[-1].name == "course_lookup":
                yield LlmResponse(content=types.Content(role="model", parts=[
                    types.Part(function_call=types.FunctionCall(name="email_draft", args={
                        "course_code": reply["data"]["course_code"], "semester": SEMESTER,
                        "request_type": "sick_leave"}, id="draft-call"))]))
                return
            if role == "policy":
                if options["empty_policy"]:
                    payload = {"status": "insufficient_evidence", "claims": []}
                else:
                    payload = {"status": "answer", "claims": [
                        {"excerpt_id": reply["excerpts"][0]["excerpt_id"]}]}
            else:
                payload = {"status": "tool_result", "result_id": reply["result_id"],
                           "answer": "This untrusted model answer must never be displayed."}
            yield LlmResponse(content=types.Content(role="model", parts=[
                types.Part(text=json.dumps(payload))]))
    monkeypatch.setattr(type(baseline.baseline_agent.model), "generate_content_async", generate)
    return options


def run(question, **kwargs):
    return asyncio.run(baseline.answer_policy(question, mode="multi", extractive=True, debug=True, **kwargs))


@pytest.mark.parametrize("question,expected", [
    ("What is the attendance policy?", ["policy"]),
    ("What are minor and major exam durations and minimum Major examination weightage?", ["policy"]),
    ("What is the slot for Autonomous Systems?", ["schedule"]),
    ("Draft a sick-leave email to the instructor.", ["email"]),
    (MIXED, ["policy", "email"]),
    ("How many Humanities 400 electives are offered?", ["schedule"]),
    ("What are the available dining options on campus?", []),
    ("Tell me the weather.", []),
])
def test_routing(question, expected):
    assert route_request(question) == expected


def test_factories_and_mode_selection(monkeypatch):
    monkeypatch.delenv("HELPDESK_AGENT_MODE", raising=False)
    assert build_root_agent("single") is baseline.baseline_agent
    root = build_root_agent("multi")
    assert [a.name for a in root.sub_agents] == ["policy_agent", "schedule_agent", "email_agent"]
    assert [t.__name__ for t in root.sub_agents[0].tools] == ["retrieve_policy"]
    assert {t.__name__ for t in baseline.build_agent().tools} == {
        "retrieve_policy", "course_slot", "courses_in_slot", "elective_count", "instructor", "email_draft", "course_lookup", "semester_offerings"}
    monkeypatch.setenv("HELPDESK_AGENT_MODE", "multi")
    assert build_root_agent().name == "academic_helpdesk_orchestrator"
    with pytest.raises(ValueError):
        build_root_agent("unsupported")


def test_policy_runner_citations_survive_handoff(model):
    result = run("What is the makeup examination policy?")
    assert result["status"] == "answer", result
    assert result["citations"][0]["chunk_id"] == "test-chunk"
    assert result["citations"][0]["quote"] == TEXT
    assert result["debug"]["routing"] == ["policy"]
    assert result["sections"][0]["validation"] == "verbatim_excerpt_checks_passed"


def test_slot_without_semester_is_specific(model, records):
    result = run("What is the slot for Test Systems?")
    assert result["status"] == "requires_clarification", result
    assert result["answer"] == "Which academic year and semester?"
    assert result["debug"]["tool_calls"][0]["tool"] == "course_slot"


def test_empty_timetable_never_invents_slot(model, records):
    result = run("What is the slot for TEST401?", semester=SEMESTER)
    assert result["status"] == "insufficient_evidence", result
    assert result["lookup_status"] == "not_found"
    assert "2026–27 Semester I" in result["answer"]
    assert result["debug"]["agent_calls"] == [{"agent": "schedule_agent", "status": "insufficient_evidence"}]


def test_isolated_synthetic_timetable_success(model, records, monkeypatch):
    monkeypatch.setattr("src.tools.schedule.timetable_rows", lambda: [{
        "course_code": "TEST401", "semester": SEMESTER, "section": "A", "slot": "E",
        "day": "Monday", "start_time": "10:00", "end_time": "11:00",
        "source_doc_id": "synthetic", "source_page": "1"}])
    result = run("What is the slot for TEST401?", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert "slot E" in result["answer"]


def test_email_verified_contact_and_placeholders(model, records):
    result = run("Draft a sick-leave email for TEST401.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["to"] == ["teacher@example.invalid"]
    assert result["data"]["sent"] is False
    assert "[Student name]" in result["data"]["body"]
    assert "[Start/exam date]" in result["data"]["body"]
    assert result["citations"][0]["document_id"] == "synthetic"
    assert "untrusted model" not in result["answer"]


def test_mixed_request_runs_policy_and_generic_email(model, records):
    result = run(MIXED)
    assert result["status"] == "answer", result
    assert [s["agent"] for s in result["sections"]] == ["policy_agent", "email_agent"]
    email = result["sections"][1]
    assert email["data"]["to"] == ["[Instructor email]"]
    assert "[Course code]" in email["data"]["subject"]
    assert "makeup examination" in email["answer"]
    assert result["citations"][0]["chunk_id"] == "test-chunk"


def test_mixed_request_with_known_instructor(model, records):
    result = run(MIXED + " The course is TEST401.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["sections"][1]["data"]["to"] == ["teacher@example.invalid"]
    assert "makeup examination" in result["sections"][1]["data"]["subject"]


def test_out_of_scope_no_model(model):
    model["fail"] = {"policy", "schedule", "email"}
    result = run("What are the dining options on campus?")
    assert result["status"] == "out_of_scope"
    assert result["debug"]["agent_calls"] == []


def test_missing_evidence_abstains(model):
    model["empty_policy"] = True
    result = run("Explain the examination policy.")
    assert result["status"] == "insufficient_evidence", result
    assert result["citations"] == []


def test_partial_elective_count(model, records):
    model.update(tool="elective_count", arguments={"department": "Humanities", "level": 400})
    result = run("How many Humanities 400 electives are offered?", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["count"] == 2
    assert result["data"]["coverage_verified"] is False
    assert "official total is not established" in result["answer"]


def test_missing_faculty_address_remains_placeholder(model, records):
    result = run("Draft a sick-leave email for TEST499.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["to"] == ["[Instructor email]"]


def test_specialist_error_logs_once_and_other_specialist_continues(model, records, monkeypatch, tmp_path):
    db = tmp_path / "review.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    model["fail"] = {"policy"}
    result = run(MIXED)
    assert result["status"] == "partial_answer", result
    assert result["sections"][0]["status"] == "error"
    assert result["sections"][1]["status"] == "answer"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unanswered_questions").fetchone()[0] == 1


def test_two_specialist_errors_log_once(model, records, monkeypatch, tmp_path):
    db = tmp_path / "review.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    model["fail"] = {"policy", "email"}
    def fail_draft(*args, **kwargs):
        raise RuntimeError("Synthetic email tool failure")
    monkeypatch.setattr("src.agents.common.email_draft", fail_draft)
    result = run(MIXED + " Course TEST401.", semester=SEMESTER)
    assert result["status"] == "error", result
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unanswered_questions").fetchone()[0] == 1


def test_missing_evidence_and_timetable_log_once(model, records, monkeypatch, tmp_path):
    db = tmp_path / "review.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    model["empty_policy"] = True
    model.update(tool="course_slot", arguments={"course": "TEST401"})
    result = run("Explain the exam policy and find the slot for TEST401.", semester=SEMESTER)
    assert result["status"] == "insufficient_evidence", result
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unanswered_questions").fetchone()[0] == 1


def test_handoff_rejects_fabricated_tool_and_citation():
    assert validate_handoff("schedule", {"status": "answer", "answer": "slot invented",
        "validation": "deterministic_tool_result"}, [], {}, "")["validation"] == "rejected"
    fake = {"status": "answer", "claims": [{"text": "Invented rule.", "citations": [
        {"chunk_id": "test-chunk", "quote": "Invented rule."}]}],
        "support_check": {"supported": True}, "validation": "verbatim_excerpt_checks_passed"}
    assert validate_handoff("policy", fake, EVIDENCE, {}, "")["validation"] == "rejected"


def test_argument_guard_preserves_only_supplied_facts():
    ctx = SimpleNamespace(state={REQUEST_KEY: {
        "question": "Draft a sick-leave email for TEST401.", "semester": "",
        "email_type": "sick_leave"}, KEY: {}})
    args = {"course_code": "TEST401", "semester": "invented", "request_type": "makeup_exam",
            "student_name": "Invented Name", "reason": "hospitalised", "start_date": "2026-01-01"}
    guard_tool_arguments(SimpleNamespace(name="email_draft"), args, ctx)
    assert args["semester"] == ""
    assert args["request_type"] == "sick_leave"
    assert args["student_name"] == args["reason"] == args["start_date"] == ""


def test_single_mode_direct_slot_still_works(model, records):
    result = asyncio.run(baseline.answer_policy(
        "What is the slot for Autonomous Systems?", semester=SEMESTER, mode="single"))
    assert result["status"] == "insufficient_evidence"
    assert result["validation"] == "deterministic_tool_result"
    assert "sections" not in result


def test_policy_trace_is_observable(model):
    result = run("Explain the exam policy.")
    assert result["debug"]["tool_calls"][0]["tool"] == "retrieve_policy"
    assert result["debug"]["tool_calls"][0]["lookup_status"] == "success"


def test_course_code_lookup_uses_existing_records(model, records):
    model.update(tool="course_lookup", arguments={"course": "TEST401"})
    result = run("What is the course title for TEST401?", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["course_name"] == "Test Systems"


def test_offerings_are_explicitly_partial(model, records):
    model.update(tool="semester_offerings", arguments={})
    result = run("List courses offered this semester.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["coverage_verified"] is False
    assert "complete offering list is not established" in result["answer"]


def test_instructor_mapping_stays_in_requested_semester(model, records):
    model.update(tool="instructor", arguments={"course_code": "TEST401"})
    result = run("Who is the instructor for TEST401?", semester="2026-27-S2")
    assert result["status"] == "insufficient_evidence", result
    assert result["lookup_status"] == "not_found"
    assert "teacher@example.invalid" not in result["answer"]


def test_unknown_slot_argument_cannot_be_invented():
    ctx = SimpleNamespace(state={REQUEST_KEY: {"question": "List courses in a slot.", "semester": SEMESTER}})
    returned = guard_tool_arguments(SimpleNamespace(name="courses_in_slot"), {"slot": "E"}, ctx)
    assert returned["status"] == "ambiguous"


def test_email_requires_final_draft_not_intermediate_contact(model, records):
    model["tool"] = "instructor"
    result = run("Draft a sick-leave email for TEST401.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["tool"] == "email_draft"
    # An intermediate contact is still forbidden as the final email handoff.
    assert validate_handoff("email", {
        "status": "answer", "validation": "deterministic_tool_result",
        "tool": "instructor", "data": {}, "citations": []},
        [], {}, "")["validation"] == "rejected"


def test_no_internal_debug_fields_without_debug(model):
    result = asyncio.run(baseline.answer_policy(
        "Explain the exam policy.", mode="multi", extractive=True))
    assert "debug" not in result
    assert not any(k.startswith("_") for section in result["sections"] for k in section)


def test_excerpts_preserve_condition_after_etc_abbreviation():
    from src.validation.excerpts import excerpt_catalog
    text = ("A student who has missed the major examination due to genuine reasons like illness "
            "etc. may be permitted to write a make-up examination for the missed components.")
    source = {**EVIDENCE[0], "text": text}
    excerpts = excerpt_catalog([source])
    assert len(excerpts) == 1
    assert excerpts[0]["text"] == text


def test_existing_session_replaces_previous_specialist_result(model, records):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    async def exercise():
        service = InMemorySessionService()
        await service.create_session(app_name="test", user_id="student", session_id="session")
        runner = Runner(agent=build_orchestrator(extractive=True),
                        app_name="test", session_service=service)
        results = []
        try:
            for question in ("Draft a sick-leave email.", "Explain the exam policy.",
                             "What are the dining options on campus?"):
                async for event in runner.run_async(
                        user_id="student", session_id="session",
                        new_message=types.Content(role="user", parts=[types.Part(text=question)])):
                    pass
                session = await service.get_session(app_name="test", user_id="student", session_id="session")
                results.append(dict(session.state[baseline.RESULT_KEY]))
        finally:
            await runner.close()
        return results
    results = asyncio.run(exercise())
    assert results[0]["sections"][0]["agent"] == "email_agent"
    assert results[1]["sections"][0]["agent"] == "policy_agent"
    assert results[2]["status"] == "out_of_scope"
    assert results[2]["citations"] == []


def test_email_can_resolve_supplied_title_before_drafting(model, records, monkeypatch):
    from src.tools import schedule
    existing = schedule.course_rows()
    monkeypatch.setattr(schedule, "course_rows", lambda: [existing[0]])
    model["resolve_first"] = True
    result = run("Draft a sick-leave email for Test Systems.", semester=SEMESTER)
    assert result["status"] == "answer", result
    assert result["data"]["to"] == ["teacher@example.invalid"]
    assert [c["tool"] for c in result["debug"]["tool_calls"]] == ["course_lookup", "email_draft"]


def test_each_failed_request_gets_one_distinct_review_entry(model, records, monkeypatch, tmp_path):
    db = tmp_path / "reviews.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    model["fail"] = {"policy"}
    for _ in range(2):
        result = run(MIXED)
        assert result["status"] == "partial_answer"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM unanswered_questions").fetchone()[0] == 2


def test_specialist_invalid_excerpts_still_use_only_one_revision(model, monkeypatch):
    from unittest.mock import AsyncMock
    from src.validation.excerpts import excerpt_catalog
    repair = AsyncMock(return_value={"raw": json.dumps({
        "status": "answer", "claims": [{"excerpt_id": "invented-id"}]})})
    monkeypatch.setattr(baseline, "revise_policy_draft", repair)
    # The model selects a real ID, but only an empty catalogue was shown.
    monkeypatch.setattr(baseline, "focused_excerpts", lambda *a, **k: [
        {**excerpt_catalog(EVIDENCE)[0], "excerpt_id": "not-in-source-catalogue"}])
    result = run("Explain the exam policy.")
    assert result["status"] == "insufficient_evidence", result
    assert repair.await_count == 1
    assert result["sections"][0]["revision_attempted"] is True
    assert result["citations"] == []


def test_missing_specialist_final_response_is_controlled(model, monkeypatch):
    async def no_final(*args, **kwargs):
        return
        yield
    monkeypatch.setattr(type(baseline.baseline_agent.model), "generate_content_async", no_final)
    result = run("Explain the exam policy.")
    assert result["status"] == "insufficient_evidence", result
    assert result["sections"][0]["failure_type"] == "specialist_validation_failed"
    assert result["human_review_log"]["status"] == "success"
