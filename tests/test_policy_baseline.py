"""Offline citation validation tests; no live model or database required."""
import json
import asyncio
from src.validation.citations import validate_policy_output

EVIDENCE = [{
    "chunk_id": "real_chunk", "document_id": "exam_circular",
    "page_start": 1, "text": "The Major examination duration is 3 hours.",
    "source_url": "https://example.org/source", "needs_review": False,
    "policy_status": "unknown",
}]


def payload(chunk_id="real_chunk", quote="The Major examination duration is 3 hours."):
    return json.dumps({"status": "answer", "claims": [{
        "text": "The Major examination lasts 3 hours.",
        "supports": [{"chunk_id": chunk_id, "quote": quote}],
    }]})


def test_valid_citation_is_rendered():
    result = validate_policy_output(payload(), EVIDENCE)
    assert result["status"] == "answer"
    assert "[exam_circular, PDF p. 1]" in result["answer"]
    assert result["warnings"]  # Unknown policy status must survive.


def test_invented_chunk_is_rejected():
    assert validate_policy_output(payload("fake"), EVIDENCE)["validation"] == "rejected"


def test_invented_quote_is_rejected():
    result = validate_policy_output(payload(quote="The Major examination lasts 2 hours."), EVIDENCE)
    assert result["validation"] == "rejected"


def test_previous_turn_evidence_cannot_be_used():
    assert validate_policy_output(payload(), [])["validation"] == "rejected"


def test_missing_citations_is_rejected():
    raw = json.dumps({"status": "answer", "claims": [{"text": "Three hours."}]})
    assert validate_policy_output(raw, EVIDENCE)["validation"] == "rejected"


def test_malformed_model_output_abstains():
    assert validate_policy_output("The answer is 3 hours.", EVIDENCE)["status"] == "insufficient_evidence"


def test_refusal_does_not_repeat_untrusted_model_message():
    raw = json.dumps({"status": "out_of_scope", "message": "Try this restaurant"})
    result = validate_policy_output(raw, EVIDENCE)
    assert result["status"] == "out_of_scope"
    assert "restaurant" not in result["answer"]


def test_clarification_is_supported():
    raw = json.dumps({"status": "requires_clarification", "claims": []})
    assert "degree track" in validate_policy_output(raw, EVIDENCE)["answer"]


def test_turn_reset_discards_previous_evidence():
    from types import SimpleNamespace
    from src.single_agent import reset_turn, EVIDENCE_KEY, RESULT_KEY
    context = SimpleNamespace(state={EVIDENCE_KEY: {"old": EVIDENCE[0]},
                                     RESULT_KEY: {"status": "answer"}})
    reset_turn(context)
    assert context.state[EVIDENCE_KEY] == {}
    assert context.state[RESULT_KEY] is None


def test_callback_rejects_uncited_final_answer():
    from types import SimpleNamespace
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from src.single_agent import validate_final, EVIDENCE_KEY, RESULT_KEY
    context = SimpleNamespace(state={EVIDENCE_KEY: {}})
    raw = LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=payload())]))
    guarded = asyncio.run(validate_final(context, raw))
    assert "don't know" in guarded.content.parts[0].text
    assert context.state[RESULT_KEY]["validation"] == "rejected"


def test_callback_preserves_tool_calls():
    from types import SimpleNamespace
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from src.single_agent import validate_final
    raw = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(
            name="retrieve_policy", args={"query": "exam policy"}))]))
    assert asyncio.run(validate_final(SimpleNamespace(state={}), raw)) is None


def test_callback_renders_verified_citation(monkeypatch):
    from unittest.mock import AsyncMock
    monkeypatch.setattr(
        "src.single_agent.verify_claim_support",
        AsyncMock(return_value={"supported": True, "method": "offline_mock"}))

    from types import SimpleNamespace
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from src.single_agent import validate_final, EVIDENCE_KEY, RESULT_KEY
    context = SimpleNamespace(state={EVIDENCE_KEY: {"real_chunk": EVIDENCE[0]}})
    raw = LlmResponse(content=types.Content(role="model", parts=[types.Part(text=payload())]))
    guarded = asyncio.run(validate_final(context, raw))
    assert "[exam_circular, PDF p. 1]" in guarded.content.parts[0].text
    assert context.state[RESULT_KEY]["status"] == "answer"
