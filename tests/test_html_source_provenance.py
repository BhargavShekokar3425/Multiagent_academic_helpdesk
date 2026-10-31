"""HTML source integrity, semester isolation, and accurate citation locations."""
import csv
import hashlib
import json

import pytest

from src.tools._records import records, registered_source, sources
from src.tools.agent_tools import render_result

FIELDS = ("course_code", "semester", "instructor_name", "email",
          "source_doc_id", "source_page", "additional_source_ids")


@pytest.fixture
def html_data(tmp_path, monkeypatch):
    html = tmp_path / "teaching.html"
    html.write_text("<h2>Semester I, TEST</h2><p>TEST401 — Test Instructor</p>")
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps({"sources": {"test_html": {
        "source_url": "https://faculty.example.invalid/teaching",
        "filename": str(html),
        "sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
        "source_location": "Teaching / Semester I, TEST",
        "semester": "TEST-S1",
        "review_status": "source_checked",
    }}}))
    monkeypatch.setenv("STRUCTURED_SOURCES_JSON", str(registry))
    path = tmp_path / "faculty.csv"
    monkeypatch.setenv("FACULTY_CSV", str(path))

    def write(**changes):
        row = dict(course_code="TEST401", semester="TEST-S1",
                   instructor_name="Test Instructor", email="test@example.invalid",
                   source_doc_id="test_html", source_page="", additional_source_ids="")
        row.update(changes)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)

    write()
    return html, registry, write


def load_rows():
    return records("FACULTY_CSV", "unused.csv", FIELDS)


def test_checked_html_has_location_and_no_fake_page(html_data):
    evidence = sources(load_rows())
    assert evidence[0]["page"] == ""
    assert evidence[0]["source_location"] == "Teaching / Semester I, TEST"
    result = render_result("r", {"r": {
        "tool": "instructor", "status": "success",
        "data": {"instructors": load_rows()}, "evidence": evidence}})
    assert "source p." not in result["answer"]
    assert "https://faculty.example.invalid/teaching" in result["answer"]


def test_changed_original_is_rejected(html_data):
    html_data[0].write_text("Changed source after review")
    with pytest.raises(ValueError, match="Saved HTML source changed"):
        load_rows()


def test_html_does_not_carry_over_between_semesters(html_data):
    html_data[2](semester="TEST-S2")
    with pytest.raises(ValueError, match="does not apply"):
        load_rows()


def test_missing_page_does_not_accept_an_unregistered_source(html_data):
    html_data[2](source_doc_id="invented")
    with pytest.raises(ValueError, match="Unregistered"):
        load_rows()


def test_html_cannot_be_labelled_pdf_page_one(html_data):
    html_data[2](source_page="1")
    with pytest.raises(ValueError, match="cannot use PDF page"):
        load_rows()


def test_additional_contact_source_must_also_be_checked(html_data):
    html_data[2](additional_source_ids="invented_contact")
    with pytest.raises(ValueError, match="Unregistered"):
        load_rows()


def test_existing_pdf_reference_still_works(html_data):
    html_data[2](source_doc_id="test_pdf", source_page="3")
    assert sources(load_rows()) == [{"document_id": "test_pdf", "page": "3"}]


def test_missing_html_location_is_rejected(html_data):
    registry = html_data[1]
    value = json.loads(registry.read_text())
    value["sources"]["test_html"].pop("source_location")
    registry.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source_location"):
        registered_source("test_html", "TEST-S1")
