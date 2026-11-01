# IIT Jodhpur Academic Helpdesk: Final Technical Report

## 1. Abstract
This assignment implements a local, source-grounded academic helpdesk and compares
an eight-tool single Google ADK agent against a coordinator with policy, schedule
and email specialists. The system uses Ollama, ChromaDB, exact source citations,
structured semester records and unsent templates. Full-v1 failures motivated bounded
contract, context, applicability and handoff repairs. This report uses saved real
inference artifacts; AI-assisted grading is not independent human verification.

## 2. Introduction and problem statement
Students need defensible academic answers, course information and professional email
drafts. Similarity retrieval alone does not establish a policy, a timetable or a
student's private records. The objective is correct supported assistance and explicit
refusal, clarification or abstention when the task cannot be established.

## 3. Academic document corpus
The retained source snapshot has 24 PDFs and 111 extracted pages.
All current paths and extraction hashes match. There are 19 flagged pages.
Some official listings are intranet-only; local availability does not establish
redistribution rights or latest applicability. No academic document or source
manifest was changed for full-v2. See data_readiness_full_v2.json for per-source hashes.

## 4. RAG methodology
Existing extraction, cleaning, page-bounded character chunks, Chroma cosine retrieval
and nomic-embed-text embeddings were preserved. The active development index has 215
chunks including review-flagged content. top_k remains 5 with bounded neighbour
context. A query with an explicit table/section reference also searches actual indexed
passages under the same programme filter. Missing references cannot be replaced by
merely similar tables. Both modes use the same retrieval implementation.
Explicit references narrow excerpt selection; abbreviations and table headings/rows
remain attached. Complete verified table spans retain exact source text rather than
reconstructing numerical cells. These are engineering changes, not new academic facts.

## 5. Structured retrieval tools
CSV tools resolve identity independently of timetable availability, list recorded
offerings, preserve co-instructors, count distinct offered course codes in Python and
return source-backed instructor mappings. Six offerings and nine assignments are
recorded; timetable rows are absent and completeness flags remain false. No official
zero is inferred from missing matches. Both architectures expose equivalent tools.

## 6. Single-agent baseline
The baseline uses one ADK agent, current-turn tool stores, the common local model
factory, shared request facts and canonical result rendering. A successful structured
result is not reinterpreted as a policy chunk. Explicit unknown result IDs still fail.
For mixed tasks, a server-owned draft is preserved alongside validated policy output.
This is a repaired v2 baseline; the original v1 implementation/results remain archived.

## 7. Multi-agent architecture
A deterministic ADK BaseAgent coordinator invokes Policy, Schedule and Email
single-turn sub-agents sequentially. Policy owns retrieval/validation, Schedule owns
structured lookup, and Email owns unsent drafts. The coordinator rechecks handoffs
and combines canonical results in Python. Policy subtask projection prevents an
email request from being mistaken for a fact that needs documentary substantiation.
Shared guards continue to use the original user request.

## 8. Validation and refusal mechanisms
Policy answers require current-turn chunk IDs and continuous exact quotations.
Source boundaries, historical sections and qualifying conditions are checked; one
bounded revision remains the limit. Header-preserving table passages are exact and
complete-row verified. Structured outputs use server-owned result IDs. Missing personal
facts stay placeholders; no email is sent. Nonacademic questions are refused.
Private academic records and unsupported future calendar years produce academic
insufficient-evidence responses. Applicability remains unverified where sources lack
current programme/cohort authority. Source fidelity is not semantic completeness.

## 9. Experimental methodology
The fixed 30 questions contain 10 policy, 8 schedule, 5 email, 4 out-of-scope and 3
unsupported cases, with two policy/email tasks. Nineteen answerable cases give a
57-response accuracy denominator per configuration at three repetitions. Other
behaviors have separate denominators. Each request uses an isolated session/log.
Full-v2 uses temperature 0, 8192 context, maximum output 2400 and a six-model-call
Runner limit. Generation models load sequentially, and each configuration is warmed
consistently before measured requests. Model digests, source/index/configuration/
application/evaluator hashes and warmup timings are saved. V1's cold/warm imbalance
prevents interpreting latency changes as an orchestration-only speed result.
The benchmark was also used for development; these are in-sample repair results,
not estimates of generalization to unseen questions.

## 10. Full-v1 versus full-v2 findings
| Experiment | Model | Architecture | Correct/graded | Pending | Graded accuracy |
|---|---|---|---:|---:|---:|
| full-v1 | qwen3:8b | single | 15/57 | 0 | 26.32% |
| full-v1 | qwen3:8b | multi | 36/57 | 0 | 63.16% |
| full-v2 final | qwen3:8b | single | 54/57 | 0 | 94.74% |
| full-v2 final | qwen3:8b | multi | 57/57 | 0 | 100.0% |

The first full-v2 attempt remains at evaluation/results/raw/full-v2. Remaining
source-scoping and specialist-task issues were investigated before the separately
saved final revision at evaluation/results/raw/full-v2/revision-2.
No old run was overwritten. Improvements combine contract handling, shared
safeguards, source context, lookup parity and coordination changes. They cannot be
assigned solely to multi-agent reasoning or any one fix without controlled ablations.

## 11. Qwen versus Llama
Qwen3:8b was executed live. Both Llama 3.1:8b configurations are explicitly skipped
when unavailable. No Qwen/Llama quality conclusion is made. Install separately with
ollama pull llama3.1:8b and run the identical warmup/benchmark workflow in a new
namespace. Mocked unit tests are not benchmark inference.

## 12. Accuracy, groundedness and citation results
full-v1 / single: citation validity 85/91 (pending 0); policy-claim support 82/91; refusal 12/12; academic abstention 0/9.

full-v1 / multi: citation validity 119/128 (pending 0); policy-claim support 71/83; refusal 12/12; academic abstention 3/9.

full-v2 final / single: citation validity 144/144 (pending 0); policy-claim support 66/69; refusal 12/12; academic abstention 9/9.

full-v2 final / multi: citation validity 163/163 (pending 0); policy-claim support 85/88; refusal 12/12; academic abstention 9/9.

The evaluator distinguishes content, workflow, outer status, identity/quote fidelity,
relevance and applicability. Known corpus sources outside the reviewed catalog are
pending, not fabricated. Paraphrases and semantic completeness need independent
review. Final review artifacts explicitly identify AI assistance and human_verified=false.
V1 citation metrics used a smaller source catalog and are not directly comparable as
an isolated citation-quality improvement. No similarity/self-review score substitutes
for ground truth.

## 13. Failure and stability analysis
V1 failures included successful tools discarded through the wrong final schema,
wrong programme/table values, fragmented abbreviations, missing requested tables,
omitted drafts and academic requests called nonacademic. Stored final-revision
error_classification.json is automated triage, not an independently confirmed cause.
Stability captures answers, minutes/percentages, citations, tool arguments, internal
outputs and grading outcomes. Formatting-only variation is separated from material
values/status changes. Repeated errors are not correct simply because they are stable.
Final-revision questions with variation:
single: P03
multi: P02, P03, P08

Residual final-run findings: single E05 omits the required advance-approval context
in all three repetitions and is graded incorrect. Both modes repeat a flattened
P09 table alongside the correctly labeled table; the evaluator rejects that duplicate
claim in each repetition, although the requested17/10 values are available in the
complete source table. Thus task accuracy and claim-support scores differ. This
readability/validation inconsistency remains a known limitation, not a perfect-score
claim. Human review should prioritize E05, P09 and policy relevance.

## 14. Latency and memory observations
full-v1 / single: mean 3.458 s; median 3.259 s; p95 12.288 s.
full-v1 / multi: mean 2.686 s; median 2.644 s; p95 6.021 s.
full-v2 final / single: mean 2.393 s; median 0.007 s; p95 7.508 s.
full-v2 final / multi: mean 1.775 s; median 0.009 s; p95 4.812 s.
Measured host: Linux-7.0.0-38-generic-x86_64-with-glibc2.39; RAM 30.97 GiB.
Windows/16 GB target verification: not established on this host. Hardware script
observations, if present, include process-RSS estimates and separate Ollama GPU
residency. RSS sums can double-count shared pages and do not certify memory capacity.
Run python -m scripts.verify_hardware --live on the actual target laptop.

## 15. Limitations
Independent human grading is pending. Timetable, full offering/category coverage,
broader instructor coverage and latest policy applicability are unresolved. Some
sources require access/redistribution review. Llama and actual Windows/16 GB tests
were unavailable. Routing and qualifier checks are conservative English heuristics.
Exact quotations can still be irrelevant; no universal semantic guarantee is claimed.
Fixed-benchmark optimization may overfit known phrasing. Broader held-out questions
and perturbations should be tested before deployment.

## 16. Conclusions
The existing architecture was retained and verified with new regressions and real
sequential experiments. The recorded results quantify the effect of the implemented
repair bundle on this benchmark. Missing evidence is still handled honestly.
The repository is code-complete with explicitly listed external verification gates;
it is not certified submission-ready on unavailable target hardware or independently
human-verified academic applicability.

## 17. References and reproducibility
- README.md: setup, execution, ingestion and evaluation commands.
- evaluation/questions.json and ground_truth.json: fixed benchmark and source-pinned rubric.
- reports/full_v1_preservation.json: immutable historical file hashes.
- evaluation/results/raw/full-v1/: original raw/graded benchmark.
- evaluation/results/raw/full-v2/revision-2/: final live manifests, responses and grades.
- reports/full_v2/comparison.csv and PNG charts: actual aggregate results.
- https://docs.ollama.com/windows
- https://pymupdf.readthedocs.io/en/latest/installation.html
- https://iitj.ac.in/office-of-academics/en/Academic-Regulations
- evaluation/DATA_REQUIREMENTS.md: authorized official records still required.
- Team contribution declaration: [supply actual member contributions].
