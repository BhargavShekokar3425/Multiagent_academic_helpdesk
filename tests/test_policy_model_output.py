"""Offline response-shape and non-thinking revision regression tests."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.validation.model_output import completion_output
from src.validation.revision import revise_policy_draft


def fake_response(content, finish="stop", reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(content=content, reasoning_content=reasoning))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def test_empty_final_content_is_not_reported_as_json_error():
    result = completion_output(fake_response(""))
    assert result["raw"] is None
    assert result["response_metadata"]["failure"] == "empty_final_content"


def test_reasoning_is_not_used_as_final_answer():
    result = completion_output(fake_response(None, reasoning="internal reasoning"))
    assert result["raw"] is None
    assert result["response_metadata"]["has_reasoning_content"]
    assert "internal reasoning" not in str(result)


def test_truncated_output_is_rejected_even_if_json_looks_valid():
    result = completion_output(fake_response('{"claims":[]}', finish="length"))
    assert result["raw"] is None
    assert result["response_metadata"]["failure"] == "output_limit"


def test_valid_final_content_is_preserved():
    result = completion_output(fake_response('{"status":"insufficient_evidence","claims":[]}'))
    assert result["error"] is None
    assert result["response_metadata"]["completion_tokens"] == 50


def test_missing_choices_is_explicit():
    assert completion_output(SimpleNamespace(choices=[]))["response_metadata"]["failure"] == "no_choices"


def test_revision_disables_thinking_and_remains_one_call(monkeypatch):
    call = AsyncMock(return_value=fake_response('{"status":"insufficient_evidence","claims":[]}'))
    monkeypatch.setattr("litellm.acompletion", call)
    result = asyncio.run(revise_policy_draft(
        "Attendance?", "rejected", {"reason": "missing qualifier"},
        [{"chunk_id": "one", "text": "Source rule", "notes": "unused extra metadata"}]))
    assert result["error"] is None
    assert call.await_count == 1
    assert call.call_args.kwargs["reasoning_effort"] == "none"
    assert call.call_args.kwargs["max_tokens"] == 2400
    assert "unused extra metadata" not in call.call_args.kwargs["messages"][1]["content"]
