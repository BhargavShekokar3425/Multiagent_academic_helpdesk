"""Deterministic tool tests using temporary, synthetic CSVs only."""
import csv
import sqlite3
import pytest

from src.tools.schedule import get_course_slot, count_department_electives
from src.tools.faculty import lookup_instructor
from src.tools.email import generate_email_draft
from src.tools.logging import log_unanswered


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    def write(variable, filename, fields, rows):
        path = tmp_path / filename
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        monkeypatch.setenv(variable, str(path))
    source = {"source_doc_id": "test_source", "source_page": "1"}
    fields = ["course_code", "course_name", "department", "level", "course_type",
              "semester", "offered", "source_doc_id", "source_page"]
    course = dict(source, course_code="TEST401", course_name="Test Systems",
                  department="Humanities", level="400", course_type="Elective",
                  semester="TEST-S1", offered="true")
    write("COURSES_CSV", "courses.csv", fields, [
        course, course, dict(course, course_code="TEST402"),
        dict(course, course_code="TEST403", offered="false"),
        dict(course, course_code="TEST404", semester="TEST-S2")])
    write("TIMETABLE_CSV", "timetable.csv",
          ["course_code", "semester", "section", "slot", "day", "start_time", "end_time",
           "source_doc_id", "source_page"], [
        dict(source, course_code="TEST401", semester="TEST-S1", section="A", slot="E",
             day="Monday", start_time="10:00", end_time="11:00"),
        dict(source, course_code="TEST401", semester="TEST-S1", section="B", slot="F",
             day="Tuesday", start_time="10:00", end_time="11:00")])
    write("FACULTY_CSV", "faculty.csv",
          ["course_code", "semester", "instructor_name", "email", "source_doc_id", "source_page"], [
        dict(source, course_code="TEST401", semester="TEST-S1",
             instructor_name="Test Instructor", email="instructor@example.invalid")])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "helpdesk.sqlite3"))
    return tmp_path


def test_exact_slot_preserves_sections_and_citations(datasets):
    result = get_course_slot("TEST401", "TEST-S1")
    assert result["status"] == "success"
    assert {r["slot"] for r in result["data"]["meetings"]} == {"E", "F"}
    assert result["evidence"][0]["document_id"] == "test_source"


def test_elective_count_is_distinct_offered_courses(datasets):
    result = count_department_electives("Humanities", 400, "TEST-S1")
    assert result["data"]["count"] == 2


def test_missing_semester_requires_clarification(datasets):
    assert get_course_slot("TEST401", "")["status"] == "ambiguous"


def test_empty_matches_do_not_establish_official_zero(datasets):
    assert count_department_electives("Other", 400, "TEST-S1")["status"] == "not_found"


def test_instructor_does_not_carry_over_between_semesters(datasets):
    assert lookup_instructor("TEST401", "TEST-S1")["status"] == "success"
    assert lookup_instructor("TEST401", "TEST-S2")["status"] == "not_found"


def test_email_uses_verified_lookup_and_missing_detail_placeholders(datasets):
    result = generate_email_draft("TEST401", "TEST-S1", "sick_leave")
    assert result["data"]["to"] == ["instructor@example.invalid"]
    assert "[Student name]" in result["data"]["body"]
    assert "[Start/exam date]" in result["data"]["body"]
    assert result["data"]["sent"] is False


def test_unknown_instructor_email_is_not_invented(datasets):
    result = generate_email_draft("UNKNOWN", "TEST-S1", "makeup_exam")
    assert result["data"]["to"] == ["[Instructor email]"]
    assert result["data"]["requires_student_review"]


def test_unanswered_log_is_parameterized_and_pending(datasets):
    result = log_unanswered("What's unknown?", "insufficient_evidence", "No source")
    assert result["status"] == "success"
    with sqlite3.connect(datasets / "helpdesk.sqlite3") as connection:
        row = connection.execute("SELECT question,status FROM unanswered_questions").fetchone()
    assert row == ("What's unknown?", "pending_review")


def test_duplicate_course_titles_require_code(datasets):
    assert get_course_slot("Test Systems", "TEST-S1")["status"] == "ambiguous"


def test_partial_count_is_labelled_in_tool_and_rendering(datasets):
    from src.tools.agent_tools import render_result
    result = count_department_electives("Humanities", 400, "TEST-S1")
    assert result["data"]["count"] == 2
    assert result["data"]["coverage_verified"] is False
    assert result["data"]["count_scope"] == "recorded_matches_only"
    rendered = render_result("r", {"r": {"tool": "elective_count", **result}})
    assert "recorded" in rendered["answer"]
    assert "official total is not established" in rendered["answer"]


def test_coverage_declaration_cannot_carry_over_semesters(datasets, monkeypatch):
    import json
    registry = datasets / "coverage.json"
    registry.write_text(json.dumps({"sources": {}, "coverage": {
        "semester": "TEST-S2", "offering_list_complete": True}}))
    monkeypatch.setenv("STRUCTURED_SOURCES_JSON", str(registry))
    result = count_department_electives("Humanities", 400, "TEST-S1")
    assert result["data"]["coverage_verified"] is False


def test_checked_complete_list_is_distinguished_from_partial(datasets, monkeypatch):
    import json
    registry = datasets / "coverage.json"
    registry.write_text(json.dumps({"sources": {}, "coverage": {
        "semester": "TEST-S1", "offering_list_complete": True}}))
    monkeypatch.setenv("STRUCTURED_SOURCES_JSON", str(registry))
    result = count_department_electives("Humanities", 400, "TEST-S1")
    assert result["data"]["coverage_verified"] is True
    assert result["data"]["count_scope"] == "complete_offering_list"
