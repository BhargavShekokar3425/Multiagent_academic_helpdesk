# Academic helpdesk to-do

Scope: finish the structured-data integration following the source acquisition work.
Checked against the workspace on 2026-10-31. Unavailable official records remain open.

## Current implementation pass

- [x] Make partial elective counts distinguish recorded matches from official totals.
- [x] Validate course/semester joins, timetable fields, instructor emails, and source IDs.
- [x] Add a repeatable offline smoke-check command for imported structured data.
- [x] Replace placeholder setup/run instructions with working commands.
- [x] Run regression tests and save the readiness/smoke reports.

## Already completed

- [x] Acquire original teaching/contact pages with HTTPS certificate verification.
- [x] Import six semester-specific course records and nine instructor assignments.
- [x] Preserve co-instructor contacts, original files, URL/section citations and SHA-256.
- [x] Reject changed HTML sources, invented page references and wrong source semesters.
- [x] Retain safe missing-data responses and draft-only email behavior.

## External records still required

- [ ] Obtain a verified Semester I, AY 2026–27 timetable with sections and slots.
- [ ] Obtain the complete departmental offering list, including levels and categories.
- [ ] Import and source-check the new records; confirm declared coverage is complete.
- [ ] Verify successful real slot lookups and an exact elective count against that export.

## Later project work (not completed by this structured-data pass)

- [ ] Review remaining OCR/table flags against originals and record applicability.
- [ ] Evaluate live policy answers against the assignment questions and ground truth.
- [ ] Implement/evaluate the specialist multi-agent path against the single-agent baseline.

## Completed pass: verification

- 110 offline tests passed.
- 24 repeatable structured-data checks passed; no model called or email sent.
- Installed dependencies: pip check reported no broken requirements.
- requirements.txt converted from UTF-16 to UTF-8; installed mcp==2.3.0 recorded.
- Current records: six courses, nine instructor assignments, zero timetable meetings.
- Current source IDs and course/semester joins are valid.
- Full dataset readiness remains false; external records above remain open.

Reports:
- reports/structured_readiness.json
- reports/structured_smoke.json

Repeat checks:
    python -m ingestion.validate_structured_data --semester 2026-27-S1 --output reports/structured_readiness.json
    python -m evaluation.check_structured_data --semester 2026-27-S1 --output reports/structured_smoke.json
    python -m pytest tests -q -p no:cacheprovider

Do not mark external records or later project tasks complete based on this test pass.
