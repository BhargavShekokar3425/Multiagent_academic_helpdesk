# Academic Helpdesk Benchmark

This framework evaluates the existing agents; it does not tune them or change the
retrieval pipeline. It uses local Ollama and the existing answer_policy API.

## Dataset and ground truth

questions.json fixes 30 questions: 10 policy/curriculum, 8 schedule/course/offering,
5 email, 4 out-of-scope and 3 academically relevant unsupported cases. P10 and E05
require policy plus email coordination. Difficulty levels are simple, medium and hard.

ground_truth.json contains acceptable criteria, routes, sources, exact PDF pages/
sections, frozen email templates, structured expectations and file hashes.

Coverage in this version:

- 19 answerable cases with source-backed facts or a verified application-template contract.
- 3 pending_source schedule cases: S01, S02 and S08.
- 1 missing-semester clarification case: S03.
- 4 expected refusals and 3 expected unsupported-question abstentions.
- 14 source identities: three PDFs and eleven checked HTML records.

The PG original PDF text was inspected. Circular 51 pages 1–2 and Circular 52 page 1
were visually inspected for the tested clauses. Relevant extraction text is frozen
with original-PDF hashes. HTML provenance uses the existing checked registry and saved
file hashes. No model answer was used as authoritative ground truth. An independent
human source/criteria sign-off is still recommended before final submission.

Named-document policy questions ask what the cited document says. They do not claim
that it is today's latest applicable policy. Production OCR/review flags are retained.
Missing timetable/coverage records do not establish that a slot/course/count is zero.

## Run on Windows PowerShell

From the repository root with the existing virtual environment and Ollama:

```powershell
.\.venv\Scripts\Activate.ps1
python -m evaluation.evaluate validate
python -m evaluation.compare_models --check
```

The check reports installed model names/digests and prerequisites. It does not pull
models. If desired, install missing models separately:

```powershell
ollama pull llama3.1:8b
```

Use the existing compatible Chroma index and nomic-embed-text model. Do not rebuild
or change retrieval settings between comparison runs.

A pilot on both architectures, seven question IDs, two repetitions:

```powershell
python -m evaluation.compare_models --ids P01,P10,S01,S04,E01,O01,U01 --repetitions 2 --output-dir evaluation/results/raw/pilot-new
```

A full comparison (30 questions x repetitions x each installed configuration):

```powershell
python -m evaluation.compare_models --repetitions 3 --answer-mode extractive --output-dir evaluation/results/raw/full-comparison-new
```

Run individual configurations:

```powershell
python -m evaluation.evaluate run --mode single --model qwen3:8b --repetitions 3 --output-dir evaluation/results/raw/qwen-single-new
python -m evaluation.evaluate run --mode multi --model qwen3:8b --repetitions 3 --output-dir evaluation/results/raw/qwen-multi-new
python -m evaluation.evaluate run --mode single --model llama3.1:8b --repetitions 3 --output-dir evaluation/results/raw/llama-single-new
python -m evaluation.evaluate run --mode multi --model llama3.1:8b --repetitions 3 --output-dir evaluation/results/raw/llama-multi-new
```

Output directories must be new. The runner refuses to overwrite a previous run.
--ids limits pilots only; omission runs all 30. --timeout sets the per-question
deadline. A timeout saves an explicitly incomplete run and stops further requests.
--answer-mode paraphrase is supported, but both compared configurations must use the
same answer mode and paraphrases require independent grounding review.

The comparison runs subprocesses sequentially, unloads other generation models and
retains the embedding model. Missing configurations are marked skipped_missing_model.
It verifies that benchmark, ground truth, source files, application code, grading
code, index report, retrieval/model options, answer mode and repetitions agree.
No automatic model installation or 16 GB memory certification is implied.

Each call uses a fresh application Runner/session. The configured generation model
is built with the existing factory, including the module-level single baseline.
Observations wrap existing search/tool-result APIs without changing their output.
Evaluation review logs go to the run directory, not the application's normal database.

## Saved artifacts

Each run contains:

- manifest.json: configuration, hashes, model digest, installed package versions,
  model residency at start, selected questions and completion status.
- benchmark_snapshot.json and ground_truth_snapshot.json: exact input versions.
- records.jsonl: flushed after every question for interrupted-run inspection.
- records.json and records.csv: answers, errors, timing, routes, evidence and grades.
- metrics.json: automatic metrics and detailed per-question stability differences.
- review_template.json: response-hash-bound manual grading form.
- review.sqlite3: isolated unanswered-question log, if failures/refusals occurred.

The comparison directory also contains comparison.json and captured subprocess logs.
Source candidates returned by search are saved as retrieved_evidence_ids and full
observations. They are not mislabeled as cited, relevant or sufficient evidence.
Tool traces and stored tool outputs remain available even for unsuccessful turns.

Raw answers, original evidence and SQLite logs stay under the already ignored
evaluation/results/raw directory. Do not commit these files without privacy review.
CSV text cells escape spreadsheet formula prefixes. JSON/JSONL/text IO explicitly
uses UTF-8.

## Scoring methodology

No embedding similarity or evaluated-model self-review is used as ground truth.

| Metric | Definition and denominator |
|---|---|
| Answer accuracy | Binary correct/incorrect among currently answerable cases. Known wrong status or structured facts score 0. Supported deterministic facts/frozen drafts can score 1. Policy completeness remains null until manual review. Reports graded denominator, eligible count, pending count and conservative lower bound. |
| Citation correctness | Correct source identity, page/location and continuous quote among returned citations. Missing citations are also exposed through required_source_coverage; zero citations do not mean perfect correctness. Unreviewed valid PDF pages require manual review; impossible pages fail. |
| Claim groundedness | Per-policy-claim support. Complete verbatim text can pass source-pinned quote, boundary and qualifier checks. Paraphrases remain pending manual review. This does not score relevance or completeness. |
| Refusal rate | Correct out-of-scope refusals / only the four out-of-scope cases and repetitions. |
| Unsupported abstention | Correct insufficient-evidence behavior / only the three unsupported academic cases and repetitions. An out-of-scope specialist wrapped in insufficient_evidence fails this check. |
| Missing-data behavior | Safe unsuccessful lookups/clarification for pending-source and missing-semester cases, separate from answer accuracy and unsupported cases. |
| Specialist routing | Actual ordered specialist calls vs expected route; multi-agent only. No invented single-agent specialist score. |
| Tool routing | Required tool names observed in actual traces; answers are not used to infer calls. |
| Latency | Mean, median and nearest-rank p95; includes failed requests and session/model setup during each API call. |
| Stability | Pairwise agreement and exact variants across repetitions for answer, observed numeric/structured values, citations, specialists, tools, status and correctness. One repetition cannot establish stability; ungraded correctness remains unknown. |

A stable error or refusal is not a correct answer. Numeric value extraction is a
text observation for stability, not a semantic correctness or entailment test.
Cold/warm model residency and small sample sizes limit latency interpretation.

**Capability caveat:** the existing baseline exposes six tools. Specialists also
expose course_lookup and semester_offerings. S06/S07 therefore measure this existing
capability difference as well as architecture. Report them separately when discussing
architecture effects; this is not a controlled tool-ablation experiment.

## Manual grading

Open records.json, ground_truth_snapshot.json and review_template.json. Review each
answer independently of the evaluated model. For policy and mixed answers require:

1. All requested facts and sub-tasks are answered.
2. Numeric values map to the correct programme/table column and named policy version.
3. Exceptions, approval requirements and discretionary wording are preserved.
4. No invented facts, contacts, dates, eligibility cutoffs or medical details appear.
5. Cited evidence supports each claim and the answer addresses the actual question.

Set accuracy to 1 only if all applicable criteria hold, otherwise 0. Groundedness and
citation pending fields accept binary scores after source review. Leave unreviewed
fields null. Supply reviewer and explanatory notes. Do not turn source fidelity
alone into answer accuracy.

```powershell
Copy-Item evaluation/results/raw/qwen-multi-new/review_template.json evaluation/results/raw/qwen-multi-new/grades.json
# Edit grades.json after independent review.
python -m evaluation.evaluate grade --run-dir evaluation/results/raw/qwen-multi-new --grades evaluation/results/raw/qwen-multi-new/grades.json
python -m evaluation.compare_models --summarize evaluation/results/raw/full-comparison-new
```

Grading writes records_graded.json/.csv and metrics_graded.json. Original responses
and scores remain unchanged. Response hashes, reviewer identity and notes are checked.
Manual grades cannot silently override automatic failures.

When scoring code changes, regrade the same saved answers without new inference:

```powershell
python -m evaluation.evaluate rescore --run-dir evaluation/results/raw/qwen-multi-new
```

This writes separate records_rescored, metrics_rescored, review_template_rescored and
rescore_manifest artifacts, verifies the saved benchmark/ground truth hashes, and
records the current grading-code hashes. Use the rescored review template for manual
review after rescoring. grade prefers rescored records when available. Comparison
summaries prefer reviewed records, then rescored records, then original records.
Reapply human grades after rescoring rather than mixing grading versions.

## Testing and experiment boundaries

```powershell
python -m pytest tests/test_evaluation.py tests/test_repository_validity.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider
```

Normal tests use injected inference and isolated logs. Real pilot output is labeled
live_application; injected test output is labeled injected_invoker. Raw live failures
are retained, and policy grades are not filled by the evaluated LLM.

The first local pilot used seven cases twice on Qwen3 single/multi (28 responses).
It exposed incomplete/failed supported lookups and mixed requests as well as answer
variation. A scoring false positive for wrapped out-of-scope responses was corrected
and saved pilot responses were explicitly rescored; original artifacts were retained.
This pilot is not a final 30-question evaluation or an assignment conclusion.
