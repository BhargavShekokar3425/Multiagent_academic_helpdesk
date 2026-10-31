"""Exact semester-specific lookups and Python-computed course counts."""
from ._records import records, response, sources, normalized, coverage_metadata
from .routing import semester_label

COURSES = ("course_code", "course_name", "department", "level", "course_type",
           "semester", "offered", "source_doc_id", "source_page")
TIMETABLE = ("course_code", "semester", "section", "slot", "day", "start_time",
             "end_time", "source_doc_id", "source_page")


def course_rows():
    return records("COURSES_CSV", "data/structured/courses.csv", COURSES)


def timetable_rows():
    return records("TIMETABLE_CSV", "data/structured/timetable.csv", TIMETABLE)


def get_course_slot(course_name_or_code: str, semester: str) -> dict:
    """Look up an exact course code or title in an explicitly specified semester."""
    course_name_or_code = course_name_or_code.strip()
    context = {"course": course_name_or_code, "semester": semester.strip()}
    if not semester.strip():
        return response("ambiguous", context, message="Which academic year and semester?")
    missing = f"No verified timetable record for {course_name_or_code} in {semester_label(semester)}."
    try:
        matches = [r for r in course_rows() if normalized(r["semester"]) == normalized(semester)
                   and normalized(course_name_or_code) in {
                       normalized(r["course_code"]), normalized(r["course_name"])}]
        codes = {r["course_code"].upper() for r in matches}
        if not codes:
            return response("not_found", context, message=missing)
        if len(codes) > 1:
            return response("ambiguous", {"matches": matches}, message="Specify a course code.")
        rows = [r for r in timetable_rows() if r["course_code"].upper() in codes
                and normalized(r["semester"]) == normalized(semester)]
        if not rows:
            return response("not_found", context, message=missing)
        return response("success", {"course_code": next(iter(codes)),
                                    "course_name": matches[0]["course_name"],
                                    "semester": semester, "meetings": rows}, sources(rows))
    except (OSError, ValueError) as exc:
        return response("error", message=str(exc))


def get_courses_in_slot(slot: str, semester: str) -> dict:
    """Return all recorded sections/meetings in a semester's exact slot."""
    if not semester.strip() or not slot.strip():
        return response("ambiguous", message="Specify slot and semester.")
    try:
        rows = [r for r in timetable_rows() if normalized(r["semester"]) == normalized(semester)
                and normalized(r["slot"]) == normalized(slot)]
        return response("success" if rows else "not_found",
                        {"semester": semester, "slot": slot, "meetings": rows}, sources(rows))
    except (OSError, ValueError) as exc:
        return response("error", message=str(exc))


def count_department_electives(department: str, level: int, semester: str) -> dict:
    """Count distinct offered course codes, never sections or timetable meetings."""
    if not semester.strip():
        return response("ambiguous", message="Specify the academic year and semester.")
    try:
        rows = course_rows()
        matches = [r for r in rows if normalized(r["semester"]) == normalized(semester)
                   and normalized(r["department"]) == normalized(department)
                   and r["level"] == str(level)
                   and normalized(r["course_type"]) == "elective"
                   and normalized(r["offered"]) in {"true", "yes", "1"}]
        if not matches:
            # An incomplete or empty dataset cannot establish an official zero.
            return response("not_found", message="No matching records; an official zero is not established.")
        codes = sorted({r["course_code"].upper() for r in matches})
        complete = coverage_metadata(semester).get("offering_list_complete") is True
        return response("success", {"count": len(codes), "course_codes": codes,
                                    "semester": semester,
                                    "coverage_verified": complete,
                                    "count_scope": "complete_offering_list" if complete else "recorded_matches_only",
                                    "counting_rule": "distinct offered course codes"}, sources(matches),
                        "" if complete else "Partial dataset: recorded matches do not establish the official total.")
    except (OSError, ValueError) as exc:
        return response("error", message=str(exc))
