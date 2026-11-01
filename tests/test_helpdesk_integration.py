"""Single-agent integration without a live model or real student records."""
import asyncio
import json
from types import SimpleNamespace

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.single_agent import reset_turn, validate_final, RESULT_KEY, build_agent
from src.tools.agent_tools import KEY, remember
from src.integration import record_unanswered


def ctx(question):
    return SimpleNamespace(state={}, user_content=types.Content(
        role="user", parts=[types.Part(text=question)]))


def final(raw):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=raw)]))


def test_dining_refused_before_model_and_logged():
    context = ctx("What are the available dining options on campus?")
    content = reset_turn(context)
    result = context.state[RESULT_KEY]
    assert "don't know" in content.parts[0].text
    assert result["status"] == "out_of_scope"
    assert result["human_review_log"]["status"] == "success"


def test_supported_academic_question_is_not_short_circuited():
    assert reset_turn(ctx("What is the attendance requirement?")) is None


def test_exact_count_rendered_from_python_not_model():
    context = ctx("How many electives?")
    reset_turn(context)
    record = remember(context, "elective_count", {
        "status": "success", "data": {"count": 2, "semester": "TEST-S1",
                                    "course_codes": ["TEST401", "TEST402"]},
        "evidence": [{"document_id": "test_source", "page": "1"}], "message": ""})
    output = asyncio.run(validate_final(context, final(json.dumps({
        "status": "tool_result", "result_id": record["result_id"],
        "answer": "Invented 99 courses"}))))
    assert "2 distinct" in output.content.parts[0].text
    assert "99" not in output.content.parts[0].text


def test_fabricated_result_id_fails_and_logs():
    context = ctx("What slot is my course?")
    reset_turn(context)
    asyncio.run(validate_final(context, final('{"status":"tool_result","result_id":"fake"}')))
    result = context.state[RESULT_KEY]
    assert result["status"] == "insufficient_evidence"
    assert result["human_review_log"]["status"] == "success"


def test_previous_turn_tool_result_is_cleared():
    context = ctx("What slot?")
    context.state[KEY] = {"result_1": {"status": "success"}}
    reset_turn(context)
    assert context.state[KEY] == {}


def test_unknown_course_does_not_get_a_fabricated_slot():
    context = ctx("What is the slot of Autonomous Systems?")
    reset_turn(context)
    record = remember(context, "course_slot", {
        "status": "not_found", "data": {}, "evidence": [],
        "message": "No timetable record for this course and semester."})
    asyncio.run(validate_final(context, final(json.dumps({
        "status": "tool_result", "result_id": record["result_id"]}))))
    assert context.state[RESULT_KEY]["status"] == "insufficient_evidence"


def test_logging_is_once_per_turn():
    context = ctx("An unanswered question")
    reset_turn(context)
    result = {"status": "insufficient_evidence", "answer": "No evidence"}
    record_unanswered(context, result, "An unanswered question")
    first = result["human_review_log"]["data"]["record_id"]
    record_unanswered(context, result, "An unanswered question")
    assert result["human_review_log"]["data"]["record_id"] == first


def test_agent_registers_all_assignment_tools():
    agent = build_agent(extractive=True)
    names = {tool.__name__ for tool in agent.tools}
    assert names == {"retrieve_policy", "course_slot", "courses_in_slot",
                     "elective_count", "instructor", "email_draft", "course_lookup", "semester_offerings"}
