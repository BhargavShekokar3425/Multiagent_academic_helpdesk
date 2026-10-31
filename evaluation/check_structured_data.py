"""Offline integration checks for imported course/instructor CSVs; never sends email."""
import argparse
import json
from pathlib import Path

from ingestion.validate_structured_data import build_report
from src.tools.schedule import course_rows, get_course_slot, count_department_electives
from src.tools.faculty import lookup_instructor
from src.tools.email import generate_email_draft
from src.tools._records import normalized


def run_checks(semester):
    readiness = build_report(semester)
    checks = []

    def check(name, condition, details=None):
        checks.append({"check": name, "passed": bool(condition), "details": details})

    check("structured_integrity", readiness.get("structurally_valid"), readiness)
    try:
        courses = [r for r in course_rows()
                   if normalized(r["semester"]) == normalized(semester)]
    except (OSError, ValueError) as exc:
        courses = []
        check("load_courses", False, str(exc))
    check("target_semester_has_courses", bool(courses))
    for row in courses:
        code = row["course_code"]
        result = lookup_instructor(code, semester)
        instructors = result.get("data", {}).get("instructors", [])
        expected = list(dict.fromkeys(r["email"] for r in instructors if r["email"]))
        check(code + ":instructor", result["status"] == "success" and
              bool(expected) and bool(result["evidence"]))
        wrong_semester = lookup_instructor(code, "UNRECORDED-CHECK-SEMESTER")
        check(code + ":semester_isolation", wrong_semester["status"] == "not_found")
        draft = generate_email_draft(code, semester, "sick_leave")
        data = draft.get("data", {})
        check(code + ":email_draft", draft["status"] == "success" and
              data.get("to") == expected and data.get("sent") is False and
              data.get("requires_student_review") is True and
              "[Student name]" in data.get("body", "") and
              "[Start/exam date]" in data.get("body", ""))
    missing = get_course_slot("UNRECORDED-CHECK-COURSE", semester)
    check("unknown_course_has_no_slot", missing["status"] == "not_found")
    missing_semester = get_course_slot("UNRECORDED-CHECK-COURSE", "")
    check("missing_semester_requires_clarification", missing_semester["status"] == "ambiguous")
    wrong = lookup_instructor("UNRECORDED-CHECK-COURSE", "UNRECORDED-CHECK-SEMESTER")
    check("unknown_assignment_has_no_recipient", wrong["status"] == "not_found")
    zero = count_department_electives("UNRECORDED-CHECK-DEPARTMENT", 0, semester)
    check("missing_records_do_not_establish_zero", zero["status"] == "not_found")
    return {"semester": semester, "checks_passed": all(c["passed"] for c in checks),
            "full_dataset_ready": readiness.get("ready", False),
            "live_model_called": False, "email_sent": False, "checks": checks,
            "limitation": "Offline integration checks do not verify original-source semantics or live policy answers."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semester", default="2026-27-S1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks(args.semester)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["checks_passed"] else 1)


if __name__ == "__main__":
    main()
