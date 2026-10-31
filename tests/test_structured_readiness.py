"""Dataset relationships and completeness reporting, using synthetic records."""
from ingestion.validate_structured_data import row_issues, build_report


def data():
    return {
        "courses": [{"course_code": "TEST401", "semester": "TEST-S1"}],
        "faculty": [{"course_code": "TEST401", "semester": "TEST-S1",
                     "instructor_name": "Test Instructor", "email": "test@example.invalid"}],
        "timetable": [{"course_code": "TEST401", "semester": "TEST-S1", "section": "A",
                       "slot": "E", "day": "Monday", "start_time": "10:00", "end_time": "11:00"}],
    }


def test_matching_course_semester_and_usable_fields():
    assert row_issues(data()) == []


def test_assignment_cannot_join_to_course_in_another_semester():
    value = data()
    value["faculty"][0]["semester"] = "TEST-S2"
    assert any("same code and semester" in r["reason"] for r in row_issues(value))


def test_missing_slot_and_invalid_time_are_reported():
    value = data()
    value["timetable"][0].update(slot="", start_time="13:00", end_time="12:00")
    issues = row_issues(value)
    assert any("slot" in r.get("fields", []) for r in issues)
    assert any("start before end" in r["reason"] for r in issues)


def test_missing_contact_is_not_reported_as_ready():
    value = data()
    value["faculty"][0]["email"] = ""
    assert any("email" in r["reason"] for r in row_issues(value))


def test_populated_rows_alone_do_not_establish_full_readiness(tmp_path, monkeypatch):
    import ingestion.validate_structured_data as module
    manifest = tmp_path / "data/document_manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("document_id\ntest_source\n")
    value = data()
    for rows in value.values():
        for row in rows:
            row.update(source_doc_id="test_source", source_page="1")
    value["courses"][0].update(department="Test", level="400", course_type="Elective")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "source_registry", lambda: {})
    monkeypatch.setattr(module, "coverage_metadata", lambda semester: {})
    monkeypatch.setattr(module, "course_rows", lambda: value["courses"])
    monkeypatch.setattr(module, "timetable_rows", lambda: value["timetable"])
    monkeypatch.setattr(module, "records", lambda *args: value["faculty"])
    result = build_report("TEST-S1")
    assert result["structurally_valid"] is True
    assert result["ready"] is False
    assert result["coverage"]["offering_list_complete"] is False
