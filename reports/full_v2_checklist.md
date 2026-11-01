# Full-v2 implementation checklist
- Phase 0: inventory inspected; full-v1 file hashes pinned in full_v1_preservation.json.
- Phase 1: canonical structured contracts and parity: single_agent.py, agent_tools.py; regression tests.
- Phase 2: source-owned abbreviation/line/table excerpts: excerpts.py, extractive.py, context_budget.py.
- Phase 3: exact table/section fallback: rag.py; isolated fake collection tests.
- Phases 4–6: shared user facts, mixed composition, scope/applicability and intent: request_context.py, single_agent.py, orchestration.py, agents/common.py.
- Phase 7: content/workflow/source validity and detailed stability: evaluation modules; immutable v1 human-review sample.
- Phase 8: separate full-v2 runs, warmup, manifests, exports and plots; skip absent Llama explicitly.
- Phases 9–10: source inventory and hardware verification scripts; no source promotion or Windows claims without evidence.
- Phases 11–12: report, README, submission checks and full regression.

## Verified delivery gates
- 288 tests passed; syntax/import and both-agent construction checks pass.
- Final live Qwen run: revision-2, 90 unique responses per architecture, three repetitions.
- AI-assisted accuracy single54/57, multi57/57; raw responses unchanged.
- Full-v1 preservation hashes all match; benchmark/source hashes pass.
- Reports, PDF, CSV and charts generated from actual stored metrics.
- Live sequential single/multi smoke checks pass on Linux30.97GiB, not Windows16GB.
- External: Llama absent, independent human review pending, source applicability and complete timetable/offerings unresolved.
- Residual: single E05 missing advance-approval context; redundant P09 flattened excerpts fail evaluator support despite correct labeled table.
- No staging, commits or pushes performed; pre-existing source moves left untouched.
