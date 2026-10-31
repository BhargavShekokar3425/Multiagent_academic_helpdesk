# Qwen3 pilot observation report

This is a small real-inference pilot, not the final assignment evaluation.

- Configuration: qwen3:8b; existing local index/embeddings and production validators;
  extractive mode; sequential single and multi execution.
- Cases: P01, P10, S01, S04, E01, O01 and U01, each twice.
- Responses: 14 per architecture, 28 total.
- Both Llama configurations were explicitly skipped because llama3.1:8b is absent.
- No source documents, structured records, agents or validators were tuned to this pilot.

| Architecture | Responses | Automatically graded answer accuracy subset | Mean / median / p95 seconds |
|---|---:|---|---|
| single | 14 | 0/4; 4 pending | 3.16 / 2.66 / 9.26 |
| multi | 14 | 4/6; 2 pending | 2.70 / 1.96 / 7.28 |

Accuracy subsets have different pending counts and are not final comparable overall
accuracy estimates. Policy/mixed answer completeness still needs independent manual
review. Verbatim support scores do not resolve this missing review.

Observed issues retained in raw records:

- The single baseline returned insufficient evidence for S04 and E01 in both repetitions,
  although the source-backed instructor/draft data exists.
- P10 produced partial results in multi mode: the policy section answered, while the email
  specialist failed to produce the draft. Both repetitions remain failures for the full task.
- U01 was incorrectly treated as outside academic scope by the specialist/model. Multi mode
  wrapped this as insufficient_evidence; the final evaluator correctly counts this as an
  incorrect academic abstention. Both architectures scored 0/2 on this pilot's unsupported case.
- O01 was correctly refused and S01 safely reported missing timetable data, each 2/2 per mode.
- P01 varied across the single baseline's two repetitions. Multi answers were stable across
  these two repetitions; stable partial/failing answers are not accurate answers.

All returned pilot citations passed the source-reference/quote checks: 19/19 single and
22/22 multi. Exact policy-claim support was 19/19 single and 14/14 multi. These denominators
count only emitted citations/claims, not unanswered cases or omitted requested facts.
Required-source coverage and manual answer completeness are separate.

An evaluator false positive for wrapped out-of-scope responses was discovered during
inspection. Saved responses were explicitly rescored with the corrected evaluator.
Original records.json, metrics.json and comparison.json were preserved.
Use comparison_reviewed.json, metrics_rescored.json and records_rescored.json for the
current automatic grading. rescore_manifest.json records the grading-code hashes and
confirms no new inference was performed for rescoring.

Raw artifacts are local and ignored:
evaluation/results/raw/milestone3-qwen-pilot/

The initial pilot manifest predates the additional grading-code fingerprint metadata;
its source/application hashes and model digest are recorded. The rescore manifest pins
the final grading implementation. Future runs automatically pin both application and
grading code. The pilot cannot establish Windows/16 GB performance, Llama quality,
whole-benchmark accuracy, or general superiority of either architecture.

Before final conclusions: independently review the source/rubric and policy responses,
resolve or explicitly retain pending-source cases, run all 30 cases with the same fixed
configuration for each installed model/architecture, and report missing grades, data gaps,
capability differences and inconsistent answers.
