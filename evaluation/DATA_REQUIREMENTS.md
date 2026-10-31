# Academic data requirements for benchmark completion

The fixed benchmark can run now. Positive timetable and official elective-total
evaluation still requires authorized academic records. No production CSV, academic
document or manifest was changed for this milestone.

## Current source snapshot

Six course offerings and nine instructor assignments are recorded for 2026-27-S1.
The timetable has zero rows. Offering-list, timetable and instructor completeness
flags remain false. Course-level metadata is missing and most department/category
fields are incomplete.

S01 and S02 are pending_source timetable cases. S08 is a pending_source official
department/level elective count. S03 intentionally omits the semester. Missing
records are never interpreted as an official zero or cancellation.

## Official-source search

The current public-source search did not establish a complete, applicable Semester I
2026-27 course/section timetable or complete department/level offering list. Direct
academic-office fetches also timed out. Search results are discovery, not imported data.

Relevant official entry points:

- https://iitj.ac.in/office-of-academics/en/Academic-Regulations
- https://iitj.ac.in/office-of-academics/en/curriculum
- https://www.iitj.ac.in/office-of-academics/en/curriculum-for-programs-from-2019-to-2025
- https://home.iitj.ac.in/~hardikjain/teaching.html

The curriculum listings explicitly mark many relevant course-content/HSS links as
intranet-only. Faculty teaching pages can substantiate individual offerings; they do
not establish complete institute-wide offering coverage or a timetable.

Existing request draft:
reports/semester_2026_27_s1_records_request.md. It remains unsent. No account access,
restriction bypass or message sending was attempted.

## Authorized files needed

1. Semester I 2026-27 timetable: course codes, sections, slots, days/times, programme/
   department coverage, issue/revision date and supersession status.
2. Complete department offering list: actual offered flag, department, course level,
   elective/core category, semester and explicit completeness/coverage statement.
3. Instructor assignments for each relevant course/section, with institutional emails.
4. Cohort-specific curriculum and credit rules, including amendments and degree track.

Use official files or authorized ERP exports without student registrations, grades,
attendance or other personal data.

## After obtaining records

Keep the existing import/provenance process in data/structured/staging/2026-27-S1.
Validate original-source semantics, saved hashes, semester, course identity and
coverage before promoting rows. Do not set completeness flags based on a handful
of faculty pages. Do not infer elective status or course level merely from a code.

Adding records changes the benchmark snapshot. Preserve v1 experiment artifacts.
Create an explicitly reviewed benchmark/ground-truth revision and update hashes and
pending-source expectations together. Re-run both architectures under the same new
snapshot; never retrofit old run results into the revised experiment.
