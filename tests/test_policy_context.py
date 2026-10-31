"""Offline prompt-budget and natural-question coverage checks."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.validation.context_budget import focused_excerpts
from src.validation.revision import revise_policy_draft


def sources():
    return [{"chunk_id": "exam", "document_id": "policy", "page_start": 1,
             "text": (
                 "The Minor examination will be a written examination for 2 hours. "
                 "The duration of the major examination will be 3 hours. "
                 "The Major examination has weightage not less than 40%."
             )}]


def test_natural_question_covers_all_three_values():
    result = focused_excerpts(sources(),
        "What are minor and major exam durations and minimum Major examination weightage?")
    combined = " ".join(e["text"] for e in result)
    assert all(term in combined for term in ["2 hours", "3 hours", "40%"])
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 9000


def test_oversized_draft_is_bounded_and_schema_is_unambiguous(monkeypatch):
    call = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content='{"status":"insufficient_evidence","claims":[]}'))],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=20)))
    monkeypatch.setattr("litellm.acompletion", call)
    result = asyncio.run(revise_policy_draft(
        "exam durations?", "irrelevant summary " * 4000,
        {"answer_mode": "extractive"}, sources()))
    assert result["error"] is None
    args = call.call_args.kwargs
    assert args["response_format"]["type"] == "json_schema"
    payload = json.loads(args["messages"][1]["content"])
    assert len(payload["rejected_draft_excerpt"]) <= 1200
    assert result["response_metadata"]["prompt_bytes"] < 18000


def test_context_limit_is_reported_as_input_failure(monkeypatch):
    call = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content='{"status":"insufficient_evidence","claims":[]}'))],
        usage=SimpleNamespace(prompt_tokens=8192, completion_tokens=20)))
    monkeypatch.setattr("litellm.acompletion", call)
    result = asyncio.run(revise_policy_draft(
        "exam durations?", "draft", {"answer_mode": "extractive"}, sources()))
    assert result["raw"] is None
    assert result["response_metadata"]["failure"] == "context_limit"
