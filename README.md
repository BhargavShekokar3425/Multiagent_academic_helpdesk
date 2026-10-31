# Multi-Agent Academic Helpdesk

Local IIT Jodhpur academic helpdesk using Google ADK, Ollama, ChromaDB and CSV lookups.
The current implemented path is the single-agent baseline. Specialist-agent evaluation
is tracked separately; see [the working checklist](reports/helpdesk_todo.md).

## Current data status

Semester I, AY 2026–27 has six faculty-recorded courses and nine instructor assignments.
Coverage is partial: the verified timetable and complete departmental offering export
are still missing. Slot queries return a specific missing-record response. Recorded
elective matches are labelled partial; they do not establish the official total.

Source records and limitations: [source report](reports/structured_data_source_check.md).
Original HTML/PDF files live in data/raw_documents/docs/ and scanned/.
HTML citations use URLs and section locations from data/structured/source_registry.json;
PDF citations use the document manifest and page numbers. Source files must retain
their recorded SHA-256. HTML files are not automatically inserted into the PDF pipeline.

## Setup on Linux

From the project directory, using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Ollama must be installed and serving locally for live policy/RAG calls.
The existing defaults are qwen3:8b and nomic-embed-text. If models are not installed:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama list
```

Optional .env settings:

```dotenv
LLM_MODEL=qwen3:8b
EMBED_MODEL=nomic-embed-text
OLLAMA_API_BASE=http://localhost:11434
CHROMA_PATH=data/vector_store
```

The requirements file is UTF-8 and includes the installed MCP dependency required by
the current LiteLLM import path. Setup commands above are documented for reuse;
this maintenance pass did not recreate the environment or download models.

## Offline checks (no model calls or email sending)

```bash
python -m ingestion.validate_structured_data --semester 2026-27-S1 --output reports/structured_readiness.json
python -m evaluation.check_structured_data --semester 2026-27-S1 --output reports/structured_smoke.json
python -m pytest tests -q -p no:cacheprovider
```

A passing smoke check means the imported rows work through the tools. It does not mean
the dataset is complete. The readiness report must still show ready=false while
official records or coverage declarations are missing. Keep partial coverage flags false.

## Live policy and retrieval checks

An existing compatible Chroma collection is needed. The current local collection
includes review-flagged passages, and source warnings are retained.

```bash
python -m src.tools.rag --query "What are the Minor and Major examination durations?"
python -m src.single_agent --query "What are minor and major exam durations and minimum Major examination weightage?" --extractive --debug
python -m src.single_agent --query "What is the slot for Autonomous Systems?" --semester "2026-27-S1" --extractive --debug
```

The last command currently returns no verified timetable record. It must not use
historical teaching lists to invent a slot.

Extractive policy answers select source-owned excerpts, validate citations and
qualifiers, and allow at most one revision before abstaining. Passing quote checks
does not independently establish current policy or cohort applicability.

## Structured tools

- courses.csv: semester-specific course records, including known metadata.
- timetable.csv: course, semester, section, slot and meetings.
- faculty.csv: all recorded course instructors and institutional emails.
- source_registry.json: checked HTML sources, locations, hashes and coverage.
- additional_source_ids: semicolon-separated contact or linked-course evidence.

Instructor and email tools retain co-instructors. Email tools produce editable drafts
only, leave missing student details as placeholders, and never send messages.

To import new official records, follow
[data/structured/staging/2026-27-S1/README.md](data/structured/staging/2026-27-S1/README.md).
The source request in reports/semester_2026_27_s1_records_request.md remains unsent.
