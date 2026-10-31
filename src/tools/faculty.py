"""Look up semester-specific instructor assignments without guessing."""
from ._records import records, response, sources, normalized


def lookup_instructor(course_code: str, semester: str) -> dict:
    """Return all instructors recorded for a course and semester."""
    if not course_code.strip() or not semester.strip():
        return response("ambiguous", message="Specify course code and semester.")
    try:
        rows = records("FACULTY_CSV", "data/structured/faculty.csv",
                       ("course_code", "semester", "instructor_name", "email",
                        "source_doc_id", "source_page"))
        matches = [r for r in rows if normalized(r["course_code"]) == normalized(course_code)
                   and normalized(r["semester"]) == normalized(semester)]
        return response("success" if matches else "not_found",
                        {"course_code": course_code.upper(), "semester": semester,
                         "instructors": matches}, sources(matches))
    except (OSError, ValueError) as exc:
        return response("error", message=str(exc))
