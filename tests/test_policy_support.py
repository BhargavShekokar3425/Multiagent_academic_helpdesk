"""Offline regression checks for claim support; no live LLM calls."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.validation.support import attendance_cutoff_error, parse_verdict, verify_claim_support

EVIDENCE = [{
    "chunk_id": "attendance", "document_id": "circular_51_2024", "page_start": 1,
    "text": (
        "Revised Policy: students' examination performance is no longer linked "
        "to their course attendance. In any case, the attendance of a student "
        "should not fall below 75% in a course."
    ),
    "policy_status": "unknown", "needs_review": True,
}]
QUOTE = "In any case, the attendance of a student should not fall below 75% in a course."


def test_known_bad_attendance_answer_is_blocked_without_model():
    claims = [{
        "text": "The attendance requirement to sit in the Major examination is a minimum of 75% in a course.",
        "supports": [{"chunk_id": "attendance", "quote": QUOTE}],
    }]
    result = asyncio.run(verify_claim_support(claims, EVIDENCE))
    assert result["supported"] is False
    assert result["method"] == "attendance_regression_guard"


def test_attendance_obligation_is_not_rejected_by_cutoff_guard():
    assert attendance_cutoff_error(
        "The revised policy states attendance should not fall below 75%.", EVIDENCE) is None


def test_model_review_requires_every_claim():
    with pytest.raises(ValueError):
        parse_verdict('{"claims":[]}', 1)


def test_string_true_is_not_a_boolean_verdict():
    with pytest.raises(ValueError):
        parse_verdict('{"claims":[{"index":0,"supported":"true","reason":"yes"}]}', 1)


def test_duplicate_claim_indices_are_invalid():
    raw = json.dumps({"claims": [
        {"index": 0, "supported": True, "reason": "supported"},
        {"index": 0, "supported": True, "reason": "supported"}]})
    with pytest.raises(ValueError):
        parse_verdict(raw, 2)


def test_one_unsupported_claim_fails_whole_answer():
    raw = json.dumps({"claims": [
        {"index": 0, "supported": True, "reason": "supported"},
        {"index": 1, "supported": False, "reason": "wrong applicability"}]})
    assert parse_verdict(raw, 2)["supported"] is False


def test_support_rejection_prevents_final_claim_release(monkeypatch):
    from src.single_agent import validate_final, EVIDENCE_KEY, RESULT_KEY
    monkeypatch.setattr("src.single_agent.verify_claim_support", AsyncMock(
        return_value={"supported": False, "reason": "Unsupported eligibility inference"}))
    monkeypatch.setattr("src.single_agent.revise_policy_draft", AsyncMock(
        return_value={"raw": None, "error": "offline simulated revision failure"}))
    raw = json.dumps({"status": "answer", "claims": [{
        "text": "The minimum attendance to sit the exam is 75%.",
        "supports": [{"chunk_id": "attendance", "quote": QUOTE}],
    }]})
    context = SimpleNamespace(state={EVIDENCE_KEY: {"attendance": EVIDENCE[0]}})
    response = asyncio.run(validate_final(context, LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=raw)]))))
    assert "sufficiently supported answer" in response.content.parts[0].text
    assert context.state[RESULT_KEY]["citations"] == []
    assert context.state[RESULT_KEY]["reason"] == "Revision attempt failed."


def test_malformed_model_review_fails_closed(monkeypatch):
    from litellm import acompletion
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))])))
    claims = [{"text": "Attendance should not fall below 75%.",
               "supports": [{"chunk_id": "attendance", "quote": QUOTE}]}]
    result = asyncio.run(verify_claim_support(claims, EVIDENCE))
    assert result["supported"] is False
    assert result["method"] == "support_check_error"
