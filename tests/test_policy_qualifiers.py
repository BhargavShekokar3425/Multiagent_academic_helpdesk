"""Targeted qualifier and per-claim citation regression tests."""
import asyncio
import json

import pytest

from src.validation.citations import validate_policy_output
from src.validation.support import qualifier_errors, verify_claim_support


@pytest.mark.parametrize("claim,quote", [
    ("Four weeks of non-contiguous absence causes semester withdrawal.",
     "Four weeks, not necessarily contiguous, results in withdrawal."),
    ("The instructor automatically de-registers the student.",
     "The instructor may recommend de-registration."),
    ("Below 75% attendance, the instructor can recommend de-registration.",
     "Without any valid reason, the instructor can recommend de-registration."),
    ("After two weeks of absence the instructor may recommend de-registration.",
     "Without sanctioned leave, the instructor may recommend de-registration."),
    ("Four weeks of absence causes semester withdrawal.",
     "Absence from all academic activities causes withdrawal."),
    ("Absence from all academic activities causes semester withdrawal.",
     "Absence from all academic activities, with or without sanctioned leave, causes withdrawal."),
])
def test_observed_qualifier_errors_are_blocked(claim, quote):
    assert qualifier_errors(claim, [{"quote": quote}])


def test_complete_withdrawal_conditions_are_preserved():
    claim = (
        "Absence from all academic activities for four or more weeks, "
        "not necessarily contiguous, with or without sanctioned leave "
        "results in semester withdrawal."
    )
    assert qualifier_errors(claim, [{"quote": claim}]) == []


def test_discretionary_condition_is_accepted():
    text = (
        "If attendance falls below 75% without any valid reason, "
        "the instructor can recommend de-registration."
    )
    assert qualifier_errors(text, [{"quote": text}]) == []


def test_unasked_withdrawal_claim_is_flagged():
    text = "Absence from all academic activities causes semester withdrawal."
    assert qualifier_errors(text, [{"quote": text}],
                            "When can an instructor recommend de-registration?")


def test_two_claims_using_one_chunk_keep_both_quotes():
    first = "The Minor examination lasts 2 hours."
    second = "The Major examination lasts 3 hours."
    source = {"chunk_id": "same", "document_id": "exam_policy",
              "page_start": 1, "text": first + " " + second,
              "needs_review": False, "policy_status": "unknown"}
    payload = {"status": "answer", "claims": [
        {"text": first, "supports": [{"chunk_id": "same", "quote": first}]},
        {"text": second, "supports": [{"chunk_id": "same", "quote": second}]},
    ]}
    result = validate_policy_output(json.dumps(payload), [source])
    assert [c["quote"] for c in result["citations"]] == [first, second]
    assert [c["claim_index"] for c in result["citations"]] == [0, 1]
    assert result["claim_evidence"][0]["citations"][0]["quote"] == first
    assert result["claim_evidence"][1]["citations"][0]["quote"] == second


def test_qualifier_guard_runs_before_model_review(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("No model call should be needed")
    monkeypatch.setattr("litellm.acompletion", forbidden)
    source = {"chunk_id": "same", "document_id": "withdrawal",
              "text": "Absence for four weeks, not necessarily contiguous, causes withdrawal."}
    claims = [{
        "text": "Four weeks of non-contiguous absence causes semester withdrawal.",
        "supports": [{"chunk_id": "same", "quote": source["text"]}],
    }]
    result = asyncio.run(verify_claim_support(claims, [source]))
    assert result["supported"] is False
    assert result["method"] == "qualifier_regression_guard"
