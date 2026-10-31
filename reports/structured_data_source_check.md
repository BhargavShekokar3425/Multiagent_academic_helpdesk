# Semester I, AY 2026–27: source acquisition and import

Checked: 2026-10-31. Target semester: 2026-27-S1.

## Current result

Six faculty-recorded courses and nine instructor assignments have been imported
into courses.csv and faculty.csv. timetable.csv remains header-only. A complete
institutional/departmental offering list and verified section/slot timetable were
not located in the public sources or connected records checked.

This is partial coverage. Do not describe six as the total number of courses
offered, use it as a complete elective list, or infer slots from these records.

## Acquired sources and records

| Course | Semester teaching source | Contact evidence |
|---|---|---|
| EEL7009 — Applied Linear Algebra | https://home.iitj.ac.in/~rn/teaching.html | https://home.iitj.ac.in/~rn/ |
| CSL7560 — Advanced Data Structures and Algorithms | https://home.iitj.ac.in/~vimalraj/ and its linked https://home.iitj.ac.in/~vimalraj/courses/adsa/csl7560.html | Vimal Raj Sharma's same home page |
| CSL2010 — Introduction to Machine Learning | https://home.iitj.ac.in/~hardikjain/teaching.html | https://www.iitj.ac.in/People/Profile/ec8b858c-7122-4046-8597-5a7871a10258 |
| AIL4080 — Introduction to Cognitive Science | https://sidharthranjan.github.io/teaching.html | Institutional Sidharth Ranjan profile below |
| LAL6220 — Language, Society and Computation | https://sidharthranjan.github.io/teaching.html | Institutional Sidharth Ranjan and Gurujegan Murugesan profiles |
| AIL7260 — Foundations of Vision and Language | https://sidharthranjan.github.io/teaching.html | Institutional Sidharth Ranjan, Deeksha Varshney and Divya Saxena profiles |

Institutional profile linking Sidharth Ranjan's external faculty site:
https://iitj.ac.in/People/Profile/2aa58988-dfd1-4079-8735-6cbfd6676b84?ln=en

Co-instructor institutional profiles:
- https://iitj.ac.in/People/Profile/09d6a999-1ee7-40b5-9205-9fab85f9ae50
- https://iitj.ac.in/People/Profile/abbbb653-e869-4981-8fa0-ba312445b97c?ln=en
- https://iitj.ac.in/People/Profile/d789ff2f-3ed9-4c8f-9510-15fcde1599c0

The external teaching list is faculty-authored and linked by the official profile.
It is not an institutional timetable or complete departmental offering export.
Its source authority is explicitly distinguished in source_registry.json.

Contact obfuscations such as [at] and [dot] were normalized. Every email was
checked against a downloaded institutional or institute-hosted contact page.
All listed co-instructors were retained. Unknown slots, course levels, and elective
classifications were left empty. CSL7560 is described as a core M.Tech CSE course
by the linked course page; other hosting departments were not inferred from codes
or instructor affiliations.

## Original file acquisition

Original HTML files are saved in data/raw_documents/docs/.
reports/official_source_downloads.json records URL, final URL, byte size and SHA-256.
data/structured/source_registry.json registers checked HTML source locations.

Download failures were traced to the university server omitting its intermediate
certificate. The intermediate was obtained from the certificate's issuer URL:
http://crt.sectigo.com/SectigoPublicServerAuthenticationCAOVR36.crt
It was verified successfully against /etc/ssl/certs/ca-certificates.crt.
Only then was it loaded into a temporary client SSL context for HTTPS downloads.
Hostname and certificate verification remained enabled. System trust settings
were not modified. The fetched contact and teaching pages were inspected directly.

HTML sources are not added to the PDF-only ingestion manifest or automatically
embedded. Structured lookups cite the HTML URL and section, without invented pages.

## Additional channels checked

- Office of Academics home/navigation and CSE undergraduate course links.
- Official first-year timetable page and its linked workbook.
- Targeted public searches for current offerings, slots, timetables and course pages.
- Connected Gmail searches for institute timetable/registration messages.
- Google Drive searches for timetables, slot systems and 2026–27 offering records.

The official first-year timetable and the relevant connected message/workbook
refer to 2023–24. They were excluded from target-semester imports. Student
allocation details were not imported.

The third-party Ironman timetable upload could not be traced to a verified
institutional original; it was excluded. Search results for other IITs were excluded.
The legacy academics.iitj.ac.in host failed DNS resolution.

## Validation

- 102 offline tests passed, including eight HTML provenance regression tests.
- Integrity changes, missing source locations, unknown source IDs and wrong source
  semesters are rejected. HTML cannot be labelled with a PDF page number.
- Six real course instructor lookups and six unsent email drafts passed.
- Wrong-semester lookup and known/unknown courses without timetable data returned
  not_found. Missing semester returned ambiguous.
- An exact elective count remains unestablished.
- Results: reports/structured_lookup_smoke_2026_27_s1.json.
- python -m ingestion.validate_structured_data reports 6 courses, 9 faculty rows,
  0 timetable rows, no unknown source IDs, ready=false.

## Remaining acquisition requirement

An authorized 2026–27 Semester I timetable/export with course codes, sections,
slots and meetings is still needed. A complete offering list with department,
level and elective category is needed for exact elective counts.

No public search can establish that inaccessible or unindexed records do not exist.
Successful slot answers and complete elective counts remain unavailable until
those records are obtained. The unsent records request is in
reports/semester_2026_27_s1_records_request.md. No email was sent.
