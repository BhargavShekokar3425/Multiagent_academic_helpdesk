"""Regression tests for source-excerpt ID selection."""
import json
import pytest

from src.validation.excerpts import excerpt_catalog, resolve_excerpt_selection
from src.validation.citations import validate_policy_output

MINOR = "The Minor examination will be a written examination for 2 hours."
MAJOR = "The duration of the major examination will be 3 hours."
WEIGHT = "The Major examination will be a written examination with a weightage not less than 40% of the overall evaluation."
SOURCE = {
    "chunk_id": "exam", "document_id": "circular_52_2024", "page_start": 1,
    "text": "Examination policy:\n1. " + MINOR + "\n2. " + WEIGHT + " " + MAJOR,
    "policy_status": "unknown", "needs_review": True,
}


def test_catalog_contains_all_requested_exam_values():
    texts = {e["text"] for e in excerpt_catalog([SOURCE])}
    assert {MINOR, MAJOR, WEIGHT} <= texts


def test_ids_render_original_case_and_citations():
    catalog = excerpt_catalog([SOURCE])
    item = next(e for e in catalog if e["text"] == MAJOR)
    raw = json.dumps({"status": "answer", "claims": [{"excerpt_id": item["excerpt_id"]}]})
    resolved = resolve_excerpt_selection(raw, [SOURCE])
    checked = validate_policy_output(resolved, [SOURCE])
    assert checked["status"] == "answer"
    assert checked["citations"][0]["quote"] == MAJOR
    assert checked["claims"][0]["text"] == MAJOR


def test_unknown_id_is_rejected():
    with pytest.raises(ValueError):
        resolve_excerpt_selection(
            '{"status":"answer","claims":[{"excerpt_id":"invented"}]}', [SOURCE])


def test_selected_id_cannot_override_source_text():
    key = excerpt_catalog([SOURCE])[0]["excerpt_id"]
    with pytest.raises(ValueError):
        resolve_excerpt_selection(json.dumps({"status": "answer", "claims": [
            {"excerpt_id": key, "text": "An invented duration."}]}), [SOURCE])


def test_catalog_omits_existing_policy():
    source = {**SOURCE, "text": "Existing Policy: The duration is 2 hours. Revised Policy: " + MAJOR}
    texts = {e["text"] for e in excerpt_catalog([source])}
    assert "The duration is 2 hours." not in texts
    assert MAJOR in texts


def test_catalog_ids_are_stable():
    assert excerpt_catalog([SOURCE]) == excerpt_catalog([SOURCE])
