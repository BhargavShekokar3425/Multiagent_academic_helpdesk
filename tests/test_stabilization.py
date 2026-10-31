"""Malformed-model and service-failure regressions using isolated local logs."""
import asyncio
import json
import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src import single_agent as app
from src.integration import record_unanswered, unanswered_logging_scope
from src.validation.excerpts import excerpt_catalog

SOURCE = {"chunk_id": "exam", "document_id": "test_policy", "page_start": 1,
          "text": "The Major examination duration is 3 hours.",
          "needs_review": False, "policy_status": "unknown"}


def context(with_evidence=True):
    return SimpleNamespace(
        state={app.EVIDENCE_KEY: {"exam": SOURCE} if with_evidence else {},
               app.REVISION_KEY: False, app.ANSWER_MODE_KEY: "extractive",
               "temp:available_excerpts": excerpt_catalog([SOURCE]) if with_evidence else []},
        user_content=types.Content(role="user", parts=[types.Part(text="Major exam duration?")]))


def response(raw):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=raw)]))


def rows():
    with sqlite3.connect(os.environ["DATABASE_PATH"]) as db:
        return db.execute("SELECT question,category,reason FROM unanswered_questions ORDER BY id").fetchall()


BAD_PAYLOADS = [
    {"status": "answer", "claims": None},
    {"status": "answer", "claims": {}},
    {"status": "answer", "claims": "claim"},
    {"status": "answer", "claims": 1},
    {"status": "answer", "claims": []},
    {"status": "answer", "claims": [None]},
    {"status": "answer", "claims": [[]]},
    {"status": "answer", "claims": [{}]},
    *[{"status": "answer", "claims": [{"excerpt_id": value}]}
      for value in (None, [], {}, 1, True, "", "   ", "invented")],
    {"status": "answer"},
    {"claims": [{"excerpt_id": "invented"}]},
    {},
]


@pytest.mark.parametrize("payload", BAD_PAYLOADS)
@pytest.mark.parametrize("mode", ["extractive", "paraphrase"])
def test_malformed_output_fails_closed_with_one_revision(monkeypatch, payload, mode):
    raw = json.dumps(payload)
    revise = AsyncMock(return_value={"raw": raw})
    monkeypatch.setattr(app, "revise_policy_draft", revise)
    checker = AsyncMock(side_effect=AssertionError("Malformed output must not reach support review"))
    monkeypatch.setattr(app, "verify_claim_support", checker)
    ctx = context()
    ctx.state[app.ANSWER_MODE_KEY] = mode
    asyncio.run(app.validate_final(ctx, response(raw)))
    result = ctx.state[app.RESULT_KEY]
    assert result["status"] == "insufficient_evidence"
    assert result["validation"] == "rejected"
    assert result["revision_attempted"] is True
    assert result["revision_succeeded"] is False
    assert result["citations"] == []
    # A second callback in the same turn cannot acquire another revision.
    asyncio.run(app.validate_final(ctx, response(raw)))
    assert revise.await_count == 1
    checker.assert_not_awaited()
    assert len(rows()) == 1


def test_malformed_claims_can_revise_to_a_valid_source_excerpt(monkeypatch):
    excerpt = excerpt_catalog([SOURCE])[0]
    good = json.dumps({"status": "answer", "claims": [{"excerpt_id": excerpt["excerpt_id"]}]})
    revise = AsyncMock(return_value={"raw": good})
    monkeypatch.setattr(app, "revise_policy_draft", revise)
    ctx = context()
    asyncio.run(app.validate_final(ctx, response('{"status":"answer","claims":null}')))
    result = ctx.state[app.RESULT_KEY]
    assert result["status"] == "answer"
    assert result["revision_succeeded"] is True
    assert result["citations"][0]["quote"] == SOURCE["text"]
    revise.assert_awaited_once()


@pytest.mark.parametrize("with_evidence", [False, True])
@pytest.mark.parametrize("missing", [
    None,
    LlmResponse(content=None),
    LlmResponse(content=types.Content(role="model", parts=[])),
    LlmResponse(content=types.Content(role="model", parts=[types.Part(text="")])),
])
def test_missing_model_output_is_validated_and_logged(monkeypatch, with_evidence, missing):
    revise = AsyncMock(return_value={"raw": None, "error": "offline repair failed"})
    monkeypatch.setattr(app, "revise_policy_draft", revise)
    ctx = context(with_evidence)
    guarded = asyncio.run(app.validate_final(ctx, missing))
    result = ctx.state[app.RESULT_KEY]
    assert guarded is not None
    assert result["status"] == "insufficient_evidence"
    assert result["validation"] == "rejected"
    assert result["revision_attempted"] is with_evidence
    assert revise.await_count == int(with_evidence)
    assert result["human_review_log"]["status"] == "success"
    assert len(rows()) == 1


def install_service_stubs(monkeypatch, failure):
    observed = {"closed": 0}

    class Service:
        def __init__(self):
            if failure == "service_init":
                raise RuntimeError("session service unavailable")
            self.state = {}

        async def create_session(self, **kwargs):
            if failure == "create_session":
                raise RuntimeError("session creation failed")
            self.state = kwargs["state"]

        async def get_session(self, **kwargs):
            if failure in {"get_session", "callback_then_session_error"}:
                raise RuntimeError("session read failed")
            if failure == "missing_session":
                return None
            return SimpleNamespace(state=self.state)

    class Runner:
        def __init__(self, **kwargs):
            if failure == "runner_init":
                raise RuntimeError("runner initialization failed")
            self.service = kwargs["session_service"]

        async def run_async(self, **kwargs):
            if failure.startswith("callback_then"):
                async def callback():
                    result = {"status": "insufficient_evidence", "answer": "No supported evidence."}
                    record_unanswered(SimpleNamespace(state={}), result, "test question")
                # ADK may execute callbacks in a separate async task.
                await asyncio.create_task(callback())
            if failure in {"run_async", "run_and_close", "callback_then_run_error"}:
                raise RuntimeError("runner execution failed")
            if failure in {"close", "success"}:
                self.service.state[app.RESULT_KEY] = {
                    "status": "answer", "answer": "Supported answer.", "citations": []}
                yield SimpleNamespace(
                    is_final_response=lambda: True,
                    content=types.Content(role="model", parts=[types.Part(text="Supported answer.")]))
            elif failure == "empty_final_event":
                yield SimpleNamespace(is_final_response=lambda: True, content=None)

        async def close(self):
            observed["closed"] += 1
            if failure in {"close", "run_and_close"}:
                raise RuntimeError("runner cleanup failed")

    monkeypatch.setattr(app, "InMemorySessionService", Service)
    monkeypatch.setattr(app, "Runner", Runner)
    if failure == "agent_build":
        def broken_agent(**kwargs):
            raise RuntimeError("agent initialization failed")
        monkeypatch.setattr(app, "build_agent", broken_agent)
    return observed


@pytest.mark.parametrize("failure", [
    "service_init", "create_session", "agent_build", "runner_init", "run_async",
    "get_session", "missing_result", "missing_session", "empty_final_event", "close",
    "run_and_close", "callback_then_run_error", "callback_then_session_error",
])
def test_service_failures_return_controlled_errors_and_log_once(monkeypatch, failure):
    observed = install_service_stubs(monkeypatch, failure)
    result = asyncio.run(app.answer_policy("test question", extractive=failure == "agent_build"))
    assert result["status"] == "error"
    assert result["human_review_log"]["status"] == "success"
    assert result["display_answer"] == result["answer"]
    logged = rows()
    assert len(logged) == 1
    assert logged[0][0] == "test question"
    if failure in {"missing_result", "missing_session", "empty_final_event"}:
        assert result["failure_type"] == "missing_final_response"
    if failure == "run_and_close":
        assert "runner execution failed" in result["error"]
        assert "runner cleanup failed" in result["cleanup_error"]
    assert observed["closed"] == int(failure not in {
        "service_init", "create_session", "agent_build", "runner_init"})


def test_successful_runner_does_not_create_an_unanswered_log(monkeypatch):
    install_service_stubs(monkeypatch, "success")
    result = asyncio.run(app.answer_policy("test question"))
    assert result["status"] == "answer"
    assert "human_review_log" not in result
    assert not os.path.exists(os.environ["DATABASE_PATH"])


def test_request_scope_reuses_callback_log_across_different_contexts():
    with unanswered_logging_scope():
        first = {"status": "insufficient_evidence", "answer": "No supported evidence."}
        second = {"status": "error", "answer": "Service failed."}
        record_unanswered(SimpleNamespace(state={}), first, "test question")
        record_unanswered(SimpleNamespace(state={}), second, "test question")
    assert len(rows()) == 1
    assert first["human_review_log"] == second["human_review_log"]


def test_returning_a_logged_result_does_not_log_it_again():
    result = {"status": "error", "answer": "Service failed."}
    record_unanswered(SimpleNamespace(state={}), result, "test question")
    record_unanswered(SimpleNamespace(state={}), result, "test question")
    assert len(rows()) == 1


def test_concurrent_requests_do_not_share_log_entries(monkeypatch):
    install_service_stubs(monkeypatch, "create_session")

    async def both():
        return await asyncio.gather(app.answer_policy("first question"),
                                    app.answer_policy("second question"))

    results = asyncio.run(both())
    ids = [r["human_review_log"]["data"]["record_id"] for r in results]
    assert len(set(ids)) == 2
    assert {r[0] for r in rows()} == {"first question", "second question"}


def test_later_requests_with_the_same_question_are_logged_separately(monkeypatch):
    install_service_stubs(monkeypatch, "create_session")
    asyncio.run(app.answer_policy("test question"))
    asyncio.run(app.answer_policy("test question"))
    assert len(rows()) == 2
