"""Evaluation contracts and fake-inference tests; no live Ollama dependency."""
import asyncio
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import evaluate, compare_models
from evaluation.scoring import (ROOT, check_citation, extract_values, fingerprint,
    grade_response, load_benchmark, metrics, stability, verify_sources)


@pytest.fixture
def benchmark():
    dataset, truth = load_benchmark()
    return {q["id"]: q for q in dataset["questions"]}, truth


def grade(qid, result, benchmark, tools=(), mode="multi"):
    questions, truth = benchmark
    return grade_response(questions[qid], truth["questions"][qid], result,
                          truth["sources"], list(tools), mode)


def faculty_result(benchmark):
    _, truth = benchmark
    cites = []
    for id in ("vimalraj_teaching_2026_27", "vimalraj_csl7560"):
        source = truth["sources"][id]
        cites.append({"document_id": id, "page": "", "source_url": source["source_url"],
                      "source_location": source["source_location"], "source_sha256": source["sha256"]})
    return {"status": "answer", "answer": "Vimal Raj Sharma — vimalraj@iitj.ac.in",
            "tool": "instructor", "data": {"course_code": "CSL7560", "semester": "2026-27-S1", "instructors": [
                {"instructor_name": "Vimal Raj Sharma", "email": "vimalraj@iitj.ac.in"}]},
            "citations": cites, "debug": {"agent_calls": [{"agent": "schedule_agent"}],
                                           "tool_calls": [{"tool": "instructor", "lookup_status": "success"}]}}


def policy_result():
    text = "The Minor examination will be a written examination for 2 hours."
    cite = {"chunk_id": "418fc490ec14443523a7a060", "document_id": "circular_52_2024",
            "page": 1, "quote": text}
    return {"status": "answer", "answer": text, "citations": [cite],
            "claims": [{"text": text, "citations": [cite]}],
            "debug": {"agent_calls": [{"agent": "policy_agent"}]}}


def record(qid, result, benchmark, repetition=1, mode="multi"):
    questions, truth = benchmark
    tools = result.get("debug", {}).get("tool_calls", [])
    return {"record_id": qid + ":" + str(repetition), "question_id": qid,
            "mode": mode, "model": "fake", "repetition": repetition,
            "category": questions[qid]["category"], "question": questions[qid]["question"], "latency_seconds": repetition,
            "verification_status": truth["questions"][qid]["verification_status"],
            "result": result, "tool_calls": tools,
            "grading": grade(qid, result, benchmark, tools, mode)}


def test_dataset_distribution_sources_and_multi_intent(benchmark):
    questions, truth = benchmark
    assert len(questions) == 30
    assert sum(q["multi_intent"] for q in questions.values()) == 2
    assert sum(q["currently_answerable"] for q in questions.values()) == 19
    assert {k for k, v in truth["questions"].items() if v["verification_status"] == "pending_source"} == {"S01", "S02", "S08"}
    assert all(q["context"].keys() == {"program", "admission_year", "semester"} for q in questions.values())


def test_source_hash_change_fails_closed(benchmark, tmp_path):
    _, truth = benchmark
    changed = deepcopy(truth)
    source = tmp_path / "source.txt"
    source.write_text("old", encoding="utf-8")
    changed["dataset_files"][str(source)] = "incorrect"
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_sources(changed)


def test_duplicate_question_is_rejected(benchmark, tmp_path):
    dataset = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    dataset["questions"][1]["id"] = dataset["questions"][0]["id"]
    path = tmp_path / "questions.json"
    evaluate.write_json(path, dataset)
    with pytest.raises(ValueError, match="unique"):
        load_benchmark(path)


def test_pdf_quote_identity_and_wrong_page(benchmark):
    sources = benchmark[1]["sources"]
    cite = policy_result()["citations"][0]
    assert check_citation(cite, sources)["score"] == 1
    assert check_citation({**cite, "quote": "The examination lasts 99 hours."}, sources)["score"] == 0
    assert check_citation({**cite, "page": 999}, sources)["score"] == 0
    assert check_citation({**cite, "document_id": "invented"}, sources)["score"] == 0


def test_html_provenance_is_checked(benchmark):
    result = faculty_result(benchmark)
    cite = result["citations"][0]
    assert check_citation(cite, benchmark[1]["sources"])["score"] == 1
    assert check_citation({**cite, "source_location": "Invented"}, benchmark[1]["sources"])["score"] == 0


def test_policy_accuracy_stays_manual_despite_exact_excerpt(benchmark):
    output = grade("P01", policy_result(), benchmark)
    assert output["accuracy"] is None
    assert output["claims"][0]["score"] == 1
    assert output["manual_review_required"]
    # A correct excerpt alone does not answer all three requested facts.


def test_model_support_flag_does_not_grade_paraphrase(benchmark):
    result = policy_result()
    result["claims"][0]["text"] = "Minor exams last two hours."
    result["support_check"] = {"supported": True}
    assert grade("P01", result, benchmark)["claims"][0]["score"] is None


def test_invented_quote_is_ungrounded(benchmark):
    result = policy_result()
    result["citations"][0]["quote"] = "Minor exams last 99 hours."
    assert grade("P01", result, benchmark)["claims"][0]["score"] == 0


def test_structured_instructor_is_automatically_scored(benchmark):
    output = grade("S04", faculty_result(benchmark), benchmark, [{"tool": "instructor"}])
    assert output["accuracy"] == 1
    assert output["required_source_coverage"]["missing"] == []


def test_wrong_instructor_fails_even_with_correct_sources(benchmark):
    result = faculty_result(benchmark)
    result["data"]["instructors"][0]["email"] = "invented@example.invalid"
    assert grade("S04", result, benchmark)["accuracy"] == 0


def test_pending_source_is_not_in_answer_accuracy_denominator(benchmark):
    output = grade("S01", {"status": "insufficient_evidence"}, benchmark)
    assert output["accuracy_eligible"] is False
    assert output["accuracy"] is None
    assert output["expected_behavior_correct"] == 1


def test_unsupported_out_of_scope_is_not_correct_abstention(benchmark):
    assert grade("U01", {"status": "out_of_scope"}, benchmark)["expected_behavior_correct"] == 0


def test_single_mode_has_no_specialist_routing_score(benchmark):
    assert grade("S04", faculty_result(benchmark), benchmark, mode="single")["specialist_routing"] is None


def test_tool_routing_is_observed_not_inferred_from_answer(benchmark):
    result = faculty_result(benchmark)
    assert grade("S04", result, benchmark)["tool_route"] == 0
    assert grade("S04", result, benchmark, [{"tool": "instructor"}])["tool_route"] == 1


def test_metrics_keep_separate_denominators_and_pending_manual(benchmark):
    records = [record("S04", faculty_result(benchmark), benchmark),
               record("P01", policy_result(), benchmark),
               record("O01", {"status": "out_of_scope"}, benchmark),
               record("U01", {"status": "out_of_scope"}, benchmark),
               record("S01", {"status": "insufficient_evidence"}, benchmark)]
    result = metrics(records)
    assert result["answer_accuracy"]["eligible"] == 2
    assert result["answer_accuracy"]["denominator"] == 1
    assert result["answer_accuracy"]["pending"] == 1
    assert result["out_of_scope_refusal"]["denominator"] == 1
    assert result["unsupported_abstention"]["value"] == 0
    assert result["missing_data_behavior"]["value"] == 1


def test_p95_uses_nearest_rank(benchmark):
    records = [record("O01", {"status": "out_of_scope"}, benchmark, n) for n in range(1, 21)]
    result = metrics(records)
    assert result["latency_seconds"]["p95"] == 19
    assert result["latency_seconds"]["median"] == 10.5


def test_stability_exposes_values_citations_routes_and_correctness(benchmark):
    first = record("S04", faculty_result(benchmark), benchmark)
    second = record("S04", faculty_result(benchmark), benchmark, 2)
    second["result"]["answer"] = "Different answer"
    second["result"]["data"]["instructors"][0]["email"] = "different@example.invalid"
    second["grading"]["accuracy"] = 0
    result = stability([first, second])[0]
    assert result["inconsistent"] is True
    assert result["fields"]["answer"]["agreement"] == 0
    assert result["fields"]["factual_values"]["agreement"] == 0
    assert result["fields"]["correctness"]["agreement"] == 0


def test_one_run_cannot_establish_stability(benchmark):
    assert stability([record("P01", policy_result(), benchmark)])[0]["measured"] is False


def test_ungraded_correctness_stability_is_unknown(benchmark):
    records = [record("P01", policy_result(), benchmark, n) for n in (1, 2)]
    result = stability(records)[0]
    assert result["fields"]["correctness"]["agreement"] is None


def test_factual_values_are_observations_not_truth():
    result = extract_values({"answer": "A made-up exam lasts 99 hours.", "data": {}})
    assert "99 hours" in result["numeric_units"]


def test_manual_grade_requires_response_hash_and_notes(benchmark):
    records = [record("P01", policy_result(), benchmark)]
    template = evaluate.review_template("run", records, benchmark[1])
    template["reviewer"] = "Test reviewer"
    template["records"][0]["accuracy"] = 0
    with pytest.raises(ValueError, match="notes"):
        evaluate.apply_manual_grades(records, template, "run")
    template["records"][0]["notes"] = "Only one of three required facts was answered."
    result = evaluate.apply_manual_grades(records, template, "run")
    assert result[0]["grading"]["accuracy"] == 0
    assert records[0]["grading"]["accuracy"] is None
    template["records"][0]["response_sha256"] = "wrong"
    with pytest.raises(ValueError, match="response"):
        evaluate.apply_manual_grades(records, template, "run")


def test_manual_grade_cannot_override_automatic_failure(benchmark):
    records = [record("P01", {"status": "error"}, benchmark)]
    entry = {"record_id": records[0]["record_id"], "response_sha256": fingerprint(records[0]["result"]),
             "accuracy": 1, "notes": "Attempted override"}
    with pytest.raises(ValueError, match="pending"):
        evaluate.apply_manual_grades(records, {"run_id": "run", "reviewer": "Reviewer", "records": [entry]}, "run")


def test_runner_repetitions_and_utf8_outputs(benchmark, tmp_path):
    calls = []
    async def fake(question, mode, model, extractive, observations):
        calls.append((question["id"], mode, model, extractive, os.environ["DATABASE_PATH"]))
        if question["id"] == "S04":
            return faculty_result(benchmark)
        if question["id"] == "P01":
            observations["retrieval_calls"].append({"tool": "retrieve_policy", "lookup_status": "success"})
            observations["retrieved_evidence"].append({"chunk_id": "test", "text": "தமிழ்"})
            return policy_result()
        return {"status": "out_of_scope", "answer": "Academic help only — नमस्ते"}
    folder = tmp_path / "run"
    prior = os.environ.get("DATABASE_PATH")
    report = asyncio.run(evaluate.run_benchmark(
        mode="multi", ids=["P01", "S04", "O01"], repetitions=2,
        output_dir=folder, invoker=fake, preflight=False))
    assert len(calls) == 6
    assert os.environ.get("DATABASE_PATH") == prior
    assert report["manifest"]["complete"]
    assert report["manifest"]["execution"] == "injected_invoker"
    records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    assert records[0]["retrieved_evidence_ids"] == ["test"]
    assert len({r["record_id"] for r in records}) == 6
    assert "नमस्ते" in (folder / "records.csv").read_text(encoding="utf-8")
    with (folder / "records.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with pytest.raises(FileExistsError):
        asyncio.run(evaluate.run_benchmark(output_dir=folder, invoker=fake, preflight=False))


def test_runner_exception_is_a_recorded_failure(tmp_path):
    async def fake(*args):
        raise RuntimeError("Synthetic failure")
    report = asyncio.run(evaluate.run_benchmark(ids=["P01"], output_dir=tmp_path / "run",
                                               invoker=fake, preflight=False))
    assert report["metrics"]["errors"] == 1
    assert report["metrics"]["answer_accuracy"]["value"] == 0


def test_timeout_aborts_without_starting_next_model_request(tmp_path):
    called = []
    async def fake(q, *args):
        called.append(q["id"])
        await asyncio.sleep(10)
    report = asyncio.run(evaluate.run_benchmark(ids=["P01", "P02"], timeout=.01,
                                               output_dir=tmp_path / "run", invoker=fake, preflight=False))
    assert called == ["P01"]
    assert not report["manifest"]["complete"]
    assert report["manifest"]["aborted_after_timeout"]


@pytest.mark.parametrize("kwargs", [{"repetitions": 0}, {"mode": "unknown"}, {"ids": ["INVALID"]}])
def test_invalid_runner_options_do_not_start_model(kwargs, tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(evaluate.run_benchmark(output_dir=tmp_path / "run", preflight=False, **kwargs))


def test_missing_model_does_not_run_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "available_models", lambda: {})
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(evaluate.run_benchmark(output_dir=tmp_path / "run"))


def test_model_adapter_reuses_api_captures_search_and_restores_model(monkeypatch, benchmark):
    from src import single_agent
    original_agent = single_agent.baseline_agent
    original_model = os.environ.get("LLM_MODEL")
    monkeypatch.setattr(single_agent, "search_policy", lambda *args, **kwargs: {
        "status": "success", "evidence": [{"chunk_id": "captured", "text": "Source"}]})
    async def fake(question, **kwargs):
        assert single_agent.baseline_agent.model.model == "ollama_chat/llama3.1:8b"
        assert kwargs["mode"] == "single" and kwargs["debug"] is True
        single_agent.search_policy("query")
        return {"status": "insufficient_evidence", "answer": "No evidence"}
    monkeypatch.setattr(single_agent, "answer_policy", fake)
    observations = {"retrieval_calls": [], "retrieved_evidence": [], "tool_results": []}
    result = asyncio.run(evaluate.invoke_application(
        benchmark[0]["P01"], "single", "llama3.1:8b", True, observations))
    assert result["status"] == "insufficient_evidence"
    assert observations["retrieved_evidence"][0]["chunk_id"] == "captured"
    assert single_agent.baseline_agent is original_agent
    assert os.environ.get("LLM_MODEL") == original_model


def test_csv_escapes_spreadsheet_formulas():
    assert evaluate.csv_cell("=malicious()") == "'=malicious()"
    assert evaluate.csv_cell("Normal answer") == "Normal answer"


def test_compare_missing_llama_is_explicit_and_sequential(tmp_path):
    commands, unloaded = [], []
    def execute(command, **kwargs):
        commands.append(command)
        target = Path(command[command.index("--output-dir") + 1])
        target.mkdir()
        evaluate.write_json(target / "manifest.json", {"comparison_fingerprint": "same", "model_digest": "digest"})
        evaluate.write_json(target / "metrics.json", {})
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    report = compare_models.compare(tmp_path / "comparison", ids=["P01"], inventory={"qwen3:8b": "digest"},
                                     execute=execute, unload=lambda name: unloaded.append(name) or [])
    assert len(commands) == 2 and unloaded == ["qwen3:8b"]
    assert [c["status"] for c in report["configurations"]] == [
        "completed", "completed", "skipped_missing_model", "skipped_missing_model"]
    assert report["all_four_configurations_completed"] is False


def test_comparison_rejects_different_data_fingerprints(tmp_path):
    signatures = iter(["first", "different"])
    def execute(command, **kwargs):
        target = Path(command[command.index("--output-dir") + 1])
        target.mkdir()
        evaluate.write_json(target / "manifest.json", {"comparison_fingerprint": next(signatures), "model_digest": "digest"})
        evaluate.write_json(target / "metrics.json", {})
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError, match="changed between"):
        compare_models.compare(tmp_path / "comparison", inventory={"qwen3:8b": "digest"},
                               execute=execute, unload=lambda name: [])


def test_embedding_is_not_unloaded(monkeypatch):
    calls = []
    def request(endpoint, payload=None, **kwargs):
        if endpoint == "/api/ps":
            return {"models": [{"name": "gemma3:latest"}, {"name": "nomic-embed-text:latest"}]} if not calls else {
                "models": [{"name": "nomic-embed-text:latest"}]}
        calls.append(payload["model"])
        return {}
    monkeypatch.setattr(compare_models, "ollama_request", request)
    assert compare_models.unload_competitors("qwen3:8b") == ["gemma3:latest"]
    assert calls == ["gemma3:latest"]


def test_reviewed_source_does_not_apply_to_another_semester(benchmark):
    cite = faculty_result(benchmark)["citations"][0]
    assert check_citation(cite, benchmark[1]["sources"], "2026-27-S2")["score"] == 0


def test_tool_results_remain_observable_after_runner_error():
    observations = {"retrieval_calls": [], "tool_results": [
        {"tool": "instructor", "result_id": "result_1", "status": "success"}]}
    trace = evaluate.merge_tool_trace({"status": "error"}, observations)
    assert trace[0]["tool"] == "instructor"


def test_wrong_semester_structured_answer_is_incorrect(benchmark):
    result = faculty_result(benchmark)
    result["data"]["semester"] = "2026-27-S2"
    assert grade("S04", result, benchmark)["accuracy"] == 0


def test_frozen_email_template_is_automatically_scored(benchmark):
    gt = benchmark[1]["questions"]["E01"]
    result = faculty_result(benchmark)
    result.update(tool="email_draft", data={
        "to": ["vimalraj@iitj.ac.in"], "subject": "Request for sick leave — CSL7560",
        "body": gt["acceptable_email_bodies"][0], "sent": False, "requires_student_review": True})
    assert grade("E01", result, benchmark)["accuracy"] == 1
    result["data"]["body"] += "\nI was admitted to hospital on an invented date."
    output = grade("E01", result, benchmark)
    assert output["accuracy"] is None
    assert output["manual_review_required"]


def test_manual_paraphrase_groundedness_requires_actual_review(benchmark):
    result = policy_result()
    result["claims"][0]["text"] = "Minor exams last two hours."
    records = [record("P01", result, benchmark)]
    grades = evaluate.review_template("run", records, benchmark[1])
    grades["reviewer"] = "Independent test reviewer"
    entry = grades["records"][0]
    entry.update(accuracy=0, notes="One supported claim, but Major duration and weightage are missing.")
    entry["claims"][0]["score"] = 1
    scored = evaluate.apply_manual_grades(records, grades, "run")
    assert scored[0]["grading"]["claims"][0]["score"] == 1
    assert scored[0]["grading"]["accuracy"] == 0


def test_manual_grades_reject_unknown_record(benchmark):
    with pytest.raises(ValueError, match="Unknown"):
        evaluate.apply_manual_grades([], {
            "run_id": "run", "reviewer": "Reviewer",
            "records": [{"record_id": "nonexistent"}]}, "run")


def test_comparison_summary_retains_raw_results(benchmark, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [record("P01", policy_result(), benchmark)]
    evaluate.write_json(run_dir / "records.json", records)
    graded = deepcopy(records)
    graded[0]["grading"].update(accuracy=0, manual_review_required=False)
    evaluate.write_json(run_dir / "records_graded.json", graded)
    from evaluation.scoring import file_hash
    evaluate.write_json(run_dir / "grading_manifest.json", {
        "input_records_sha256": file_hash(run_dir / "records.json"),
        "grading_implementation": {str(p.relative_to(ROOT)): file_hash(p)
                                  for p in sorted((ROOT / "evaluation").glob("*.py"))}})
    evaluate.write_json(tmp_path / "comparison.json", {"configurations": [
        {"status": "completed", "run_dir": str(run_dir), "model": "fake", "mode": "multi"}]})
    report = compare_models.summarize(tmp_path)
    assert report["configurations"][0]["grades_applied"] is True
    assert report["configurations"][0]["metrics"]["answer_accuracy"]["value"] == 0
    assert json.loads((run_dir / "records.json").read_text(encoding="utf-8"))[0]["grading"]["accuracy"] is None


def test_comparison_timeout_does_not_start_next_configuration(tmp_path):
    import subprocess
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], 1)
    result = compare_models.compare(tmp_path / "comparison", ids=["P01"],
        inventory={"qwen3:8b": "digest", "llama3.1:8b": "digest2"},
        execute=execute, unload=lambda name: [])
    assert len(calls) == 1
    assert result["aborted_after_timeout"]


def test_out_of_scope_specialist_wrapped_as_insufficient_is_not_correct_abstention(benchmark):
    result = {"status": "insufficient_evidence", "answer": "This supports academic questions.",
              "sections": [{"status": "out_of_scope", "answer": "Outside academic scope."}]}
    assert grade("U01", result, benchmark)["expected_behavior_correct"] == 0
    assert grade("S01", result, benchmark)["expected_behavior_correct"] == 0


def test_rescore_preserves_raw_records_without_inference(benchmark, tmp_path):
    folder = tmp_path / "run"
    async def fake(*args):
        return {"status": "insufficient_evidence", "answer": "No evidence",
                "sections": [{"status": "out_of_scope"}]}
    asyncio.run(evaluate.run_benchmark(ids=["U01"], output_dir=folder, invoker=fake, preflight=False))
    before = (folder / "records.json").read_bytes()
    records = evaluate.rescore_run(folder)
    assert records[0]["grading"]["expected_behavior_correct"] == 0
    assert (folder / "records.json").read_bytes() == before
    assert json.loads((folder / "rescore_manifest.json").read_text(encoding="utf-8"))["no_inference_performed"]


def test_review_summary_rejects_unversioned_manual_grades(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    evaluate.write_json(run / "records.json", [])
    evaluate.write_json(run / "records_graded.json", [])
    evaluate.write_json(tmp_path / "comparison.json", {"configurations": [
        {"status": "completed", "run_dir": str(run)}]})
    with pytest.raises(ValueError, match="grading manifest"):
        compare_models.summarize(tmp_path)


def test_refusal_correctness_stability_uses_behavior_scores(benchmark):
    first = record("O01", {"status": "out_of_scope"}, benchmark)
    second = record("O01", {"status": "answer", "answer": "An unsupported forecast."}, benchmark, 2)
    result = stability([first, second])[0]
    assert result["fields"]["correctness"]["values"] == [1, 0]
    assert result["fields"]["correctness"]["agreement"] == 0
