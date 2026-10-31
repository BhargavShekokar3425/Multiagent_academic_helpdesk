"""Read-only structured-data integrity, coverage and join checks."""
import argparse
import csv
import json
import re
from pathlib import Path

from src.tools.schedule import course_rows, timetable_rows
from src.tools._records import records, ROOT, source_registry, coverage_metadata, normalized


def row_issues(datasets):
    """Check relationships and usable fields without inferring missing facts."""
    courses = {(normalized(r["course_code"]), normalized(r["semester"]))
               for r in datasets.get("courses", [])}
    issues = []
    for name in ("timetable", "faculty"):
        for index, row in enumerate(datasets.get(name, []), 2):
            key = (normalized(row["course_code"]), normalized(row["semester"]))
            if key not in courses:
                issues.append({"dataset": name, "row": index,
                               "reason": "No course with the same code and semester"})
            if name == "faculty":
                if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", row["email"]):
                    issues.append({"dataset": name, "row": index,
                                   "reason": "Missing or invalid instructor email"})
                if not row["instructor_name"]:
                    issues.append({"dataset": name, "row": index,
                                   "reason": "Missing instructor name"})
            else:
                missing = [k for k in ("section", "slot", "day", "start_time", "end_time")
                           if not row.get(k)]
                if missing:
                    issues.append({"dataset": name, "row": index,
                                   "reason": "Missing timetable fields", "fields": missing})
                valid_times = all(re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",
                                              row.get(k, ""))
                                  for k in ("start_time", "end_time"))
                if not valid_times or row["start_time"] >= row["end_time"]:
                    issues.append({"dataset": name, "row": index,
                                   "reason": "Meeting needs valid HH:MM times with start before end"})
                if row.get("day") not in ("Monday", "Tuesday", "Wednesday", "Thursday",
                                          "Friday", "Saturday", "Sunday"):
                    issues.append({"dataset": name, "row": index,
                                   "reason": "Unknown meeting day"})
    return issues


def build_report(semester=None):
    loaders = {
        "courses": course_rows,
        "timetable": timetable_rows,
        "faculty": lambda: records("FACULTY_CSV", "data/structured/faculty.csv",
            ("course_code", "semester", "instructor_name", "email", "source_doc_id", "source_page")),
    }
    report = {"semester": semester, "datasets": {}, "structurally_valid": True}
    try:
        with (ROOT / "data/document_manifest.csv").open(encoding="utf-8-sig") as stream:
            known = {r["document_id"] for r in csv.DictReader(stream)}
        known.update(source_registry())
    except (OSError, ValueError) as exc:
        return {**report, "structurally_valid": False, "ready": False, "error": str(exc)}
    datasets = {}
    for name, loader in loaders.items():
        try:
            all_rows = loader()
            rows = [r for r in all_rows if semester is None or
                    normalized(r["semester"]) == normalized(semester)]
            datasets[name] = rows
            references = {r["source_doc_id"] for r in rows}
            references.update(s for r in rows for s in
                              r.get("additional_source_ids", "").split(";") if s)
            missing = sorted(references - known)
            report["datasets"][name] = {
                "rows": len(rows), "semesters": sorted({r["semester"] for r in rows}),
                "unknown_source_ids": missing, "needs_data": not bool(rows),
            }
            report["structurally_valid"] &= not bool(missing)
        except (OSError, ValueError) as exc:
            datasets[name] = []
            report["datasets"][name] = {"error": str(exc), "needs_data": True}
            report["structurally_valid"] = False
    report["join_and_field_issues"] = row_issues(datasets)
    report["structurally_valid"] &= not bool(report["join_and_field_issues"])
    semesters = {r["semester"] for rows in datasets.values() for r in rows}
    target = semester or (next(iter(semesters)) if len(semesters) == 1 else "")
    try:
        declared = coverage_metadata(target) if target else {}
    except (OSError, ValueError) as exc:
        declared = {}
        report["coverage_error"] = str(exc)
        report["structurally_valid"] = False
    report["coverage"] = {k: declared.get(k) is True for k in (
        "offering_list_complete", "timetable_complete", "instructor_assignments_complete")}
    report["courses_missing_count_metadata"] = [
        {"course_code": r["course_code"], "semester": r["semester"],
         "fields": [k for k in ("department", "level", "course_type") if not r.get(k)]}
        for r in datasets.get("courses", [])
        if any(not r.get(k) for k in ("department", "level", "course_type"))]
    report["ready"] = bool(report["structurally_valid"] and
                           all(datasets.values()) and all(report["coverage"].values()) and
                           not report["courses_missing_count_metadata"])
    report["notice"] = ("Source integrity and valid joins do not prove policy applicability. "
                        "Full readiness also requires explicitly checked dataset coverage.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semester")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.semester)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
