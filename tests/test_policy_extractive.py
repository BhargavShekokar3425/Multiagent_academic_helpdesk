"""Offline tests for the optional verbatim-excerpt path."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src.validation.extractive import verify_extractive_claims
from src.single_agent import ANSWER_MODE_KEY, EVIDENCE_KEY, RESULT_KEY, validate_final

TEXT = "In any case, the attendance of a student should not fall below 75% in a course."
SOURCE = {"chunk_id": "rule", "document_id": "policy", "page_start": 1,
          "text": TEXT, "needs_review": True, "policy_status": "unknown"}


def claims(text=TEXT, quote=TEXT):
    return [{"text": text, "supports": [{"chunk_id": "rule", "quote": quote}]}]


def test_verbatim_rule_is_accepted_without_model_judgment():
    assert verify_extractive_claims(claims(), [SOURCE])["supported"]


def test_paraphrase_cannot_use_extractive_path():
    result = verify_extractive_claims(claims("Attendance must be at least 75%."), [SOURCE])
    assert not result["supported"]


def test_internal_sentence_fragment_is_rejected():
    fragment = "the attendance of a student should not fall below 75% in a course."
    assert not verify_extractive_claims(claims(fragment, fragment), [SOURCE])["supported"]


def test_old_policy_excerpt_is_rejected_when_revision_follows():
    old = "Students must earn a C grade."
    source = {**SOURCE, "text": "Existing policy: " + old + " Revised Policy: " + TEXT}
    assert not verify_extractive_claims(claims(old, old), [source])["supported"]


def test_missing_final_punctuation_is_rejected():
    fragment = TEXT[:-1]
    assert not verify_extractive_claims(claims(fragment, fragment), [SOURCE])["supported"]


def test_extractive_callback_preserves_quotes_and_skips_model_reviewer(monkeypatch):
    reviewer = AsyncMock(side_effect=AssertionError("No model reviewer in extractive mode"))
    monkeypatch.setattr("src.single_agent.verify_claim_support", reviewer)
    ctx = SimpleNamespace(state={
        ANSWER_MODE_KEY: "extractive", EVIDENCE_KEY: {"rule": SOURCE}},
        user_content=types.Content(role="user", parts=[types.Part(text="Attendance obligation?")]))
    raw = json.dumps({"status": "answer", "claims": claims()})
    result = asyncio.run(validate_final(ctx, LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=raw)]))))
    assert TEXT in result.content.parts[0].text
    checked = ctx.state[RESULT_KEY]
    assert checked["validation"] == "verbatim_excerpt_checks_passed"
    assert checked["claims"][0]["citations"][0]["quote"] == TEXT
    assert checked["warnings"]  # Review and policy-status warnings survive.
    reviewer.assert_not_awaited()


FAILING_PASSAGE = (
    "According to the recent Senate and Board-approved policy, students' "
    "examination performance is no longer linked to their course attendance. "
    "As per the amendment in attendance policy a student is expected to have full "
    "attendance in each course unless the student takes a leave of absence for "
    "valid medical or bonafide reasons."
)


def test_actual_revised_policy_heading_regression():
    source = {**SOURCE, "text":
              "Existing policy: Students must earn a C grade.\n\n"
              "Revised Policy: " + FAILING_PASSAGE + "\n\n" + TEXT}
    result = verify_extractive_claims(
        claims(FAILING_PASSAGE, FAILING_PASSAGE), [source])
    assert result["supported"]


def test_paragraph_start_is_valid():
    source = {**SOURCE, "text": "Attendance requirements\n\n" + TEXT}
    assert verify_extractive_claims(claims(), [source])["supported"]


def test_named_heading_colon_is_valid():
    source = {**SOURCE, "text": "Attendance requirements:\n" + TEXT}
    assert verify_extractive_claims(claims(), [source])["supported"]


def test_parenthesized_list_item_is_valid():
    source = {**SOURCE, "text": "Requirements\n(a) " + TEXT}
    assert verify_extractive_claims(claims(), [source])["supported"]


def test_clipped_condition_after_colon_is_invalid():
    quote = "The instructor may recommend de-registration."
    source = {**SOURCE, "text": "If absence exceeds two weeks: " + quote}
    assert not verify_extractive_claims(claims(quote, quote), [source])["supported"]


def test_clipped_condition_across_pdf_blank_lines_is_invalid():
    quote = "The instructor may recommend de-registration."
    source = {**SOURCE, "text": "If absence exceeds two weeks,\n\n" + quote}
    assert not verify_extractive_claims(claims(quote, quote), [source])["supported"]


def test_historical_rule_at_paragraph_start_is_still_rejected():
    old = "Students must earn a C grade."
    source = {**SOURCE, "text": "Existing Policy:\n\n" + old + "\n\nRevised Policy: " + TEXT}
    assert not verify_extractive_claims(claims(old, old), [source])["supported"]


def test_identical_quote_can_match_revised_occurrence():
    source = {**SOURCE, "text": "Existing Policy: " + TEXT + "\nRevised Policy: " + TEXT}
    assert verify_extractive_claims(claims(), [source])["supported"]


def test_offsets_preserve_unicode_whitespace_and_linebreaks():
    source = {**SOURCE, "text": "Revised\u00a0Policy:\n\n" + FAILING_PASSAGE.replace(" ", "\n", 3)}
    assert verify_extractive_claims(claims(FAILING_PASSAGE, FAILING_PASSAGE), [source])["supported"]
