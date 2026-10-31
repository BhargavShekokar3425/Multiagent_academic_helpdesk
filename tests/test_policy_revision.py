"""Mocked integration tests for the one-attempt revision limit."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.single_agent import EVIDENCE_KEY, RESULT_KEY, REVISION_KEY, reset_turn, validate_final

SOURCE = {"chunk_id": "exam", "document_id": "exam_policy", "page_start": 1,
          "text": "The Major examination duration is 3 hours.", "needs_review": False,
          "policy_status": "unknown"}


def draft(text="The Major examination lasts 3 hours.", chunk_id="exam"):
    return json.dumps({"status": "answer", "claims": [{
        "text": text, "supports": [{
            "chunk_id": chunk_id, "quote": SOURCE["text"]}]}]})


def context():
    return SimpleNamespace(
        state={EVIDENCE_KEY: {"exam": SOURCE}, REVISION_KEY: False},
        user_content=types.Content(role="user", parts=[types.Part(text="How long is the Major exam?")]))


def response(raw):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=raw)]))


def setup(monkeypatch, verdicts, revised):
    checker = AsyncMock(side_effect=verdicts)
    reviser = AsyncMock(return_value=revised)
    monkeypatch.setattr("src.single_agent.verify_claim_support", checker)
    monkeypatch.setattr("src.single_agent.revise_policy_draft", reviser)
    return checker, reviser


FAIL = {"supported": False, "method": "local_model_support_review",
        "claims": [{"index": 0, "supported": False, "reason": "Wrong duration"}]}
PASS = {"supported": True, "method": "local_model_support_review"}


def test_successful_revision_is_fully_rechecked(monkeypatch):
    checker, reviser = setup(monkeypatch, [FAIL, PASS], {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft("The Major examination lasts 2 hours."))))
    result = ctx.state[RESULT_KEY]
    assert result["status"] == "answer"
    assert result["revision_attempted"] and result["revision_succeeded"]
    assert checker.await_count == 2 and reviser.await_count == 1
    assert reviser.call_args.args[0] == "How long is the Major exam?"
    assert reviser.call_args.args[2] == FAIL
    assert "2 hours" not in result["answer"]


def test_failed_revision_abstains_without_third_attempt(monkeypatch):
    checker, reviser = setup(monkeypatch, [FAIL, FAIL], {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    result = ctx.state[RESULT_KEY]
    assert result["status"] == "insufficient_evidence" and result["citations"] == []
    assert result["revision_attempted"] and not result["revision_succeeded"]
    assert checker.await_count == 2 and reviser.await_count == 1


def test_invented_citation_in_revision_is_rejected_before_support_check(monkeypatch):
    checker, reviser = setup(monkeypatch, [FAIL], {"raw": draft(chunk_id="fake"), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    assert ctx.state[RESULT_KEY]["status"] == "insufficient_evidence"
    assert checker.await_count == 1 and reviser.await_count == 1


def test_already_supported_answer_never_revises(monkeypatch):
    checker, reviser = setup(monkeypatch, [PASS], {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    assert ctx.state[RESULT_KEY]["status"] == "answer"
    assert not ctx.state[RESULT_KEY]["revision_attempted"]
    reviser.assert_not_awaited()


def test_support_service_error_does_not_trigger_revision(monkeypatch):
    _, reviser = setup(monkeypatch, [{"supported": False, "method": "support_check_error"}],
                       {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    assert ctx.state[RESULT_KEY]["status"] == "insufficient_evidence"
    reviser.assert_not_awaited()


def test_revision_budget_cannot_be_reused_in_same_turn(monkeypatch):
    _, reviser = setup(monkeypatch, [FAIL, FAIL, FAIL], {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    asyncio.run(validate_final(ctx, response(draft())))
    assert reviser.await_count == 1
    reset_turn(ctx)
    assert ctx.state[REVISION_KEY] is False


def test_revision_model_error_abstains(monkeypatch):
    _, reviser = setup(monkeypatch, [FAIL], {"raw": None, "error": "timeout"})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    assert ctx.state[RESULT_KEY]["status"] == "insufficient_evidence"
    assert ctx.state[RESULT_KEY]["reason"] == "Revision attempt failed."
    assert reviser.await_count == 1


def test_ellipsis_quote_gets_one_revision_then_support_review(monkeypatch):
    source = {
        **SOURCE,
        "text": (
            "Absence from all academic activities for four or more weeks, "
            "not necessarily contiguous, results in semester withdrawal."
        ),
    }
    quote = source["text"]
    bad = json.dumps({"status": "answer", "claims": [{
        "text": "The specified absence leads to semester withdrawal.",
        "supports": [{"chunk_id": "exam",
                      "quote": "Absence from all academic activities...results in semester withdrawal."}],
    }]})
    good = json.dumps({"status": "answer", "claims": [{
        "text": "Absence from all academic activities for four or more weeks, not necessarily contiguous, results in semester withdrawal.",
        "supports": [{"chunk_id": "exam", "quote": quote}],
    }]})
    checker, reviser = setup(monkeypatch, [PASS], {"raw": good, "error": None})
    ctx = context()
    ctx.state[EVIDENCE_KEY] = {"exam": source}
    asyncio.run(validate_final(ctx, response(bad)))
    result = ctx.state[RESULT_KEY]
    assert result["status"] == "answer"
    assert result["revision_succeeded"]
    assert checker.await_count == 1 and reviser.await_count == 1
    feedback = reviser.call_args.args[2]
    assert feedback["stage"] == "citation_validation"
    assert feedback["details"]["source_text"] == quote
    assert "..." in feedback["details"]["model_quote"]


def test_quote_then_support_failure_does_not_allow_second_revision(monkeypatch):
    malformed_quote = json.loads(draft())
    malformed_quote["claims"][0]["supports"][0]["quote"] = "The Major...3 hours."
    checker, reviser = setup(monkeypatch, [FAIL], {"raw": draft(), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(json.dumps(malformed_quote))))
    result = ctx.state[RESULT_KEY]
    assert result["status"] == "insufficient_evidence"
    assert "sufficiently supported answer" in result["answer"]
    assert result["failure_type"] == "answer_validation_failed"
    assert checker.await_count == 1 and reviser.await_count == 1


def test_revised_quote_mismatch_still_abstains(monkeypatch):
    wrong = json.loads(draft())
    wrong["claims"][0]["supports"][0]["quote"] = "The Major...3 hours."
    checker, reviser = setup(monkeypatch, [FAIL], {"raw": json.dumps(wrong), "error": None})
    ctx = context()
    asyncio.run(validate_final(ctx, response(draft())))
    result = ctx.state[RESULT_KEY]
    assert result["validation"] == "rejected"
    assert result["reason"] == "Supporting quote does not match source."
    assert reviser.await_count == 1 and checker.await_count == 1


def test_no_retrieved_evidence_never_triggers_revision(monkeypatch):
    checker, reviser = setup(monkeypatch, [], {"raw": draft(), "error": None})
    ctx = context()
    ctx.state[EVIDENCE_KEY] = {}
    asyncio.run(validate_final(ctx, response(draft())))
    result = ctx.state[RESULT_KEY]
    assert result["failure_type"] == "no_usable_evidence"
    assert not result["revision_attempted"]
    assert "don't know" in result["answer"]
    checker.assert_not_awaited()
    reviser.assert_not_awaited()
