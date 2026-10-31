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


## Single-agent and multi-agent execution

The baseline remains the default. Choose the architecture explicitly for comparable
runs against the same local model, documents, Chroma collection and structured CSVs:

```powershell
python -m src.single_agent --mode single --query "What are minor and major exam durations and minimum Major examination weightage?" --extractive --debug
python -m src.single_agent --mode multi --query "What are minor and major exam durations and minimum Major examination weightage?" --extractive --debug
python -m src.single_agent --mode multi --query "What is the slot for Autonomous Systems?" --semester "2026-27-S1" --debug
python -m src.single_agent --mode multi --program "PG" --query "I missed my Major exam due to illness. Explain the applicable makeup-exam policy and draft an email to the instructor." --extractive --debug
```

For the ADK entry point (`src.agent.root_agent`), set
`HELPDESK_AGENT_MODE=single` or `HELPDESK_AGENT_MODE=multi` before starting ADK.
In PowerShell: `$env:HELPDESK_AGENT_MODE = "multi"`. CLI `--mode` takes precedence.
Restart ADK after changing the environment; root_agent is constructed on import.
No additional dependencies or models are required.

The multi-agent architecture uses an ADK BaseAgent coordinator with three
single-turn LlmAgent sub-agents:

- Policy Agent: existing retrieve_policy tool and all existing citation, support,
  qualifier and bounded-revision checks.
- Schedule Agent: exact course, semester, slot, offering, instructor and
  Python-computed elective-count tools.
- Email Agent: course resolution, verified instructor lookup and existing unsent
  email templates. Missing personal facts and contacts remain placeholders.

The coordinator selects specialists using explicit intent rules, invokes each
selected specialist once, and combines canonical validated results in Python.
It does not ask another LLM to rewrite validated policy claims or drafts.
ADK sub_agents were selected over AgentTool wrappers so sequencing, state handoff
and final rendering stay explicit inside one Runner/session. The installed ADK
also recommends single-turn sub-agents over direct AgentTool wrappers.
The root coordinator is intentionally deterministic; the three specialists use
the shared local model configuration and execute sequentially.

Each policy handoff rechecks quote identity and, for extractive answers, source
boundaries and qualifying conditions. Paraphrased answers must have passed the
baseline support review. Structured results are reconstructed from server-owned
tool records. Unvalidated specialist prose is never the final answer.

Responses contain status, answer, citations and sections. Single-specialist
structured answers also expose tool, lookup_status and data. Mixed success/failure
returns partial_answer and retains the successful section. With --debug, routing,
agent_calls, tool_calls and specialist_diagnostics expose what ran. Errors,
refusals and unresolved requests reuse the existing review log, once per request.

Timetable coverage remains empty. Offering and instructor coverage remains partial.
A recorded elective count is explicitly partial unless complete coverage was
declared for that semester. No missing course, slot, address, medical detail or
examination date is generated to fill a data gap. Drafts are never sent.

Offline tests mock model inference while exercising real ADK Runner sessions,
callbacks, tools, handoff validation and logging. Synthetic academic records stay
inside isolated tests:

```powershell
python -m pytest tests/test_multi_agent.py tests/test_repository_validity.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider
```

The milestone's local Qwen3 smoke checks covered the exam-duration question and
the mixed makeup-policy/email request. These are smoke checks, not an accuracy
evaluation or Windows memory benchmark. Current policy applicability and OCR review
warnings remain visible. Llama comparison and full evaluation are subsequent work.
