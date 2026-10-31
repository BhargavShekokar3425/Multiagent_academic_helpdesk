# Semester I, AY 2026–27 import staging

These header-only templates match the current operational CSV schema.
They are not read by the helpdesk tools. There are no invented example rows.

1. Obtain an authorized official semester offering list, timetable, and instructor
   assignment/contact export using the unsent request in
   reports/semester_2026_27_s1_records_request.md.
2. Keep original public or otherwise authorized source files in
   data/raw_documents/docs/. Do not save credentials or session cookies.
3. Record source URL, title, original semester, revision date and exact location.
   PDF rows require source_doc_id and a positive source_page registered in
   data/document_manifest.csv. Checked HTML rows instead use a blank source_page
   and an entry in data/structured/source_registry.json with source URL, section,
   original file SHA-256, semester applicability and review_status=source_checked.
   additional_source_ids retains separate contact or linked course evidence.
   Never invent PDF pages for HTML or spreadsheet sources. Spreadsheet provenance
   still needs an explicit adapter before operational import.
4. Fill these staging CSVs using semester 2026-27-S1. Leave unestablished fields
   empty and identify them in the source report. Do not infer offered courses
   from historical curriculum, infer slots from faculty names, or treat partial
   teaching lists as the complete departmental elective list.
5. Resolve unknown required fields, verify course-code and semester joins, and
   compare every populated row against its source before promoting it.
   Preserve any existing operational records when merging.
6. After approved records are merged, run:
   python -m ingestion.validate_structured_data

Only after real records are imported, test:
- A known course title and code return the correct semester, section and slot.
- An unknown course and a wrong semester return not_found.
- Instructor lookup returns all recorded instructors and source references.
- An email draft uses only the verified recipient; no email is sent.
- Elective counts use a complete departmental offering list and distinct codes.

The validator checks structure and source IDs. It does not approve source contents,
verify timetable accuracy, or establish completeness of an elective count.
