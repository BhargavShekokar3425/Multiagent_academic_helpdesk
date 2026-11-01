# Academic Helpdesk full-v2 handover
Existing ADK/Ollama/Chroma architecture retained. Shared canonical structured
rendering and eight-tool parity, trusted student facts, mixed policy/email
composition, abbreviation/table passage handling, exact table-reference retrieval,
academic applicability abstention and source-aware evaluation diagnostics added.

Final selected run: evaluation/results/raw/full-v2/revision-2/.
Qwen live:90responses each,3repetitions. AI-assisted accuracy single54/57,
multi57/57. Both12/12 out-of-scope and9/9 unsupported abstentions.
These are in-sample repair-benchmark results, not independent human grading or
proof that orchestration alone causes gains. Full-v1 files unchanged.
288tests pass; scripts.verify_submission reports code gates passed.
Sources24PDFs,111pages,19review-flagged;6offerings,9instructor records,0timetable.
Llama absent. Actual Windows16GB verification unavailable (host Linux30.97GiB).
Single E05 omits advance-approval context. Both modes return redundant P09
flattened table excerpts rejected by evaluator support; labeled table remains correct.
Do not silently alter grades or old runs to remove these limitations.

README contains PowerShell setup/run/rebuild/evaluation instructions.
reports/final_report.md/.pdf, full_v2/, data_readiness_full_v2.json,
hardware_verification.json and submission_verification.json are deliverables.
Human review packet is under evaluation/results/raw/review_support/.
No automatic commit/push; main had pre-existing document moves and manifest edits.
Review source redistribution separately; never stage raw/private files, secrets,
SQLite/Chroma/model artifacts or broad data directories.
Next: human-review policy relevance/E05/P09, verify actual Windows16GB machine,
install optionalLlama and use a fresh run namespace; acquire authorized timetable,
complete offerings/category coverage and verify current-policy applicability.
