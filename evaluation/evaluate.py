"""Reproducible local benchmark runner and explicit human-grading workflow."""
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid

from .scoring import (ROOT, extract_values, file_hash, fingerprint, grade_response,
                      load_benchmark, metrics, verify_sources)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def environment(values):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def ollama_request(endpoint, payload=None, timeout=10):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
    request = urllib.request.Request(
        base + endpoint, data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as stream:
        return json.load(stream)


def available_models():
    return {m["name"]: m.get("digest", "") for m in ollama_request("/api/tags")["models"]}


def configuration_snapshot(truth):
    """Fingerprint data/settings, excluding mutable review logs and credentials."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    paths = {"COURSES_CSV": "data/structured/courses.csv",
             "FACULTY_CSV": "data/structured/faculty.csv",
             "TIMETABLE_CSV": "data/structured/timetable.csv",
             "STRUCTURED_SOURCES_JSON": "data/structured/source_registry.json"}
    for variable, default in paths.items():
        actual = Path(os.getenv(variable, default))
        actual = actual if actual.is_absolute() else ROOT / actual
        if file_hash(actual) != truth["dataset_files"][default]:
            raise ValueError(variable + " does not match the benchmark source snapshot.")
    chroma = Path(os.getenv("CHROMA_PATH", "data/vector_store"))
    chroma = chroma if chroma.is_absolute() else ROOT / chroma
    report_path = chroma / "index_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return {"source_files": verify_sources(truth),
            "capability_caveat": "The existing baseline has six tools; specialists also expose course_lookup and semester_offerings. S06/S07 measure this capability difference too; this is not a controlled tool-ablation.",
            "embedding_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
            "index_report": report,
            "retrieval_configuration": "Existing application defaults: top_k=5; include-review policy unchanged.",
            "model_configuration": {"temperature": 0, "num_ctx": 8192, "max_tokens": 2400,
                                    "reasoning_effort": "none", "max_llm_calls": 6},
            "grading_implementation_files": {str(p.relative_to(ROOT)): file_hash(p)
                    for p in sorted((ROOT / "evaluation").glob("*.py"))},
            "implementation_files": {str(p.relative_to(ROOT)): file_hash(p)
                                     for p in sorted((ROOT / "src").rglob("*.py"))}}


async def invoke_application(question, mode, model, extractive, observations):
    """Instrument existing APIs without changing their outputs or validators."""
    from unittest.mock import patch
    from src import single_agent
    from src.tools import agent_tools
    original_search, original_remember = single_agent.search_policy, agent_tools.remember

    def search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        evidence = result.get("evidence", [])
        observations["retrieval_calls"].append({
            "tool": "retrieve_policy", "lookup_status": result.get("status"),
            "query": args[0] if args else kwargs.get("query", ""),
            "evidence_ids": [s["chunk_id"] for s in evidence]})
        observations["retrieved_evidence"].extend(deepcopy(evidence))
        return result

    def remember(context, name, result):
        selected = original_remember(context, name, result)
        observations["tool_results"].append({"tool": name, **deepcopy(selected)})
        return selected

    context = question["context"]
    # The pre-existing module-level baseline may have been imported under another
    # model. Build it through the existing factory under this run's model setting.
    with environment({"LLM_MODEL": model}), patch.object(single_agent, "search_policy", search), patch.object(agent_tools, "remember", remember):
        with patch.object(single_agent, "baseline_agent", single_agent.build_agent()):
            return await single_agent.answer_policy(
                question["question"], program=context["program"],
                admission_year=context["admission_year"], semester=context["semester"],
                mode=mode, extractive=extractive, debug=True)


def merge_tool_trace(result, observations):
    debug = result.get("debug", {}).get("tool_calls", [])
    # The baseline does not trace retrieval, so use the observed existing search
    # API for both modes. Other traces remain the application's server-owned ones.
    structured = [t for t in debug if t.get("tool") != "retrieve_policy"]
    seen = {(t.get("tool"), t.get("result_id")) for t in structured}
    for item in observations["tool_results"]:
        if (item["tool"], item["result_id"]) not in seen:
            structured.append({"tool": item["tool"], "result_id": item["result_id"],
                               "lookup_status": item["status"], "trace_source": "observed_tool_result"})
    return observations["retrieval_calls"] + structured


def csv_cell(value):
    value = "" if value is None else str(value)
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def write_csv(path, records):
    fields = ["record_id", "question_id", "model", "mode", "repetition", "category",
              "status", "latency_seconds", "accuracy", "accuracy_method", "behavior_correct",
              "answer", "citations", "specialists", "tools", "factual_values", "error"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            result, grade = r["result"], r["grading"]
            row = {k: r.get(k) for k in fields}
            row.update(status=result.get("status"), answer=result.get("answer", ""),
                       accuracy=grade["accuracy"], accuracy_method=grade["accuracy_method"],
                       behavior_correct=grade["expected_behavior_correct"], error=result.get("error", ""),
                       citations=json.dumps(result.get("citations", []), ensure_ascii=False),
                       specialists=json.dumps(result.get("debug", {}).get("agent_calls", [])),
                       tools=json.dumps(r["tool_calls"]), factual_values=json.dumps(extract_values(result)))
            writer.writerow({k: csv_cell(v) for k, v in row.items()})


def review_template(run_id, records, truth):
    return {"run_id": run_id, "reviewer": "", "instructions":
        "Set binary accuracy/support/citation scores only after checking the pinned sources and every rubric criterion. "
        "Leave unreviewed fields null. Explain each reviewed record. Automatic failures cannot be silently overridden.",
        "records": [{
            "record_id": r["record_id"], "response_sha256": fingerprint(r["result"]),
            "question_id": r["question_id"],
            "question": r["question"],
            "answer": r["result"].get("answer", ""),
            "source_evidence": truth["questions"][r["question_id"]]["evidence"],
            "returned_citations": r["result"].get("citations", []),
            "criteria": truth["questions"][r["question_id"]]["acceptable_answer_criteria"],
            "accuracy": None, "notes": "",
            "claims": [{"index": c["index"], "text": c["text"], "score": None}
                       for c in r["grading"]["claims"] if c["score"] is None],
            "citations": [{"index": c["index"], "score": None}
                          for c in r["grading"]["citation_checks"] if c["score"] is None],
        } for r in records if r["grading"]["manual_review_required"]]}


async def run_benchmark(mode="single", model="qwen3:8b", repetitions=1, ids=None,
                        output_dir=None, timeout=180, answer_mode="extractive",
                        invoker=None, preflight=True):
    if mode not in {"single", "multi"} or answer_mode not in {"extractive", "paraphrase"}:
        raise ValueError("Invalid architecture or answer mode.")
    if type(repetitions) is not int or repetitions < 1 or timeout <= 0:
        raise ValueError("Repetitions and timeout must be positive.")
    dataset, truth = load_benchmark()
    questions = dataset["questions"]
    if ids:
        if set(ids) - {q["id"] for q in questions}:
            raise ValueError("Unknown question IDs.")
        questions = [q for q in questions if q["id"] in ids]
    inventory = available_models() if preflight else {}
    if preflight and model not in inventory:
        raise ValueError("Ollama model unavailable: " + model + ". Install it separately before evaluating.")
    snapshot = configuration_snapshot(truth)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = Path(output_dir) if output_dir else ROOT / "evaluation/results/raw" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
                "benchmark_id": dataset["benchmark_id"], "benchmark_sha256": fingerprint(dataset),
                "ground_truth_sha256": fingerprint(truth), "mode": mode, "model": model,
                "model_digest": inventory.get(model), "answer_mode": answer_mode,
                "ollama_residency_at_start": ollama_request("/api/ps") if preflight else None,
                "repetitions": repetitions, "question_ids": [q["id"] for q in questions],
                "planned_records": len(questions) * repetitions, "timeout_seconds": timeout,
                "execution": "live_application" if invoker is None else "injected_invoker",
                "snapshot": snapshot, "comparison_fingerprint": fingerprint({
                    "benchmark": dataset, "truth": truth, "snapshot": snapshot,
                    "answer_mode": answer_mode, "repetitions": repetitions, "timeout": timeout, "ids": [q["id"] for q in questions]}),
                "package_versions": {name: importlib.metadata.version(name)
                    for name in ("google-adk", "litellm", "chromadb", "ollama")}}
    write_json(folder / "manifest.json", metadata)
    write_json(folder / "benchmark_snapshot.json", dataset)
    write_json(folder / "ground_truth_snapshot.json", truth)
    records = []
    execute = invoker or invoke_application
    aborted = False
    with environment({"DATABASE_PATH": folder.resolve() / "review.sqlite3"}):
        with (folder / "records.jsonl").open("w", encoding="utf-8") as stream:
            for repetition in range(1, repetitions + 1):
                for q in questions:
                    observations = {"retrieval_calls": [], "retrieved_evidence": [], "tool_results": []}
                    start = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(execute(
                            q, mode, model, answer_mode == "extractive", observations), timeout)
                        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                            raise ValueError("Application returned an invalid result object.")
                    except asyncio.TimeoutError:
                        result = {"status": "error", "answer": "", "error": "Benchmark request timed out.",
                                  "failure_type": "benchmark_timeout"}
                        # Do not start another inference after a potentially active
                        # provider thread timed out. Retain the partial run honestly.
                        aborted = True
                    except Exception as exc:
                        result = {"status": "error", "answer": "", "error": type(exc).__name__ + ": " + str(exc),
                                  "failure_type": "benchmark_exception"}
                    elapsed = time.perf_counter() - start
                    tools = merge_tool_trace(result, observations)
                    gt = truth["questions"][q["id"]]
                    record = {"record_id": run_id + ":" + q["id"] + ":" + str(repetition),
                              "question_id": q["id"], "question": q["question"],
                              "category": q["category"], "difficulty": q["difficulty"],
                              "verification_status": gt["verification_status"],
                              "model": model, "mode": mode, "repetition": repetition,
                              "session_isolation": "Fresh answer_policy Runner/session per invocation.",
                              "latency_seconds": elapsed, "result": result, "tool_calls": tools,
                              "retrieved_evidence_ids": list(dict.fromkeys(
                                  s["chunk_id"] for s in observations["retrieved_evidence"])),
                              "observations": observations}
                    record["grading"] = grade_response(q, gt, result, truth["sources"], tools, mode)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    if aborted:
                        break
                if aborted:
                    break
    metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), completed_records=len(records),
                    complete=len(records) == metadata["planned_records"] and not aborted, aborted_after_timeout=aborted)
    write_json(folder / "manifest.json", metadata)
    write_json(folder / "records.json", records)
    write_csv(folder / "records.csv", records)
    write_json(folder / "metrics.json", metrics(records))
    write_json(folder / "review_template.json", review_template(run_id, records, truth))
    return {"run_dir": str(folder), "manifest": metadata, "metrics": metrics(records)}


def apply_manual_grades(records, grades, run_id):
    """Bind independent human grades to exact response hashes."""
    if grades.get("run_id") != run_id or not isinstance(grades.get("records"), list):
        raise ValueError("Grades must identify this run and contain a records list.")
    if not isinstance(grades.get("reviewer"), str) or not grades["reviewer"].strip():
        raise ValueError("A reviewer identity is required.")
    copied = deepcopy(records)
    by_id = {r["record_id"]: r for r in copied}
    seen = set()
    for entry in grades["records"]:
        rid = entry.get("record_id")
        if rid not in by_id or rid in seen:
            raise ValueError("Unknown or duplicate grading record.")
        seen.add(rid)
        record = by_id[rid]
        if entry.get("response_sha256") != fingerprint(record["result"]):
            raise ValueError("Grade does not match the recorded response.")
        updates = entry.get("accuracy") is not None or any(
            v.get("score") is not None for key in ("claims", "citations") for v in entry.get(key, []))
        if not updates:
            continue
        if not isinstance(entry.get("notes"), str) or not entry["notes"].strip():
            raise ValueError("Reviewed records require explanatory notes.")
        grade = record["grading"]
        if entry.get("accuracy") is not None:
            value = entry["accuracy"]
            if type(value) is not int or value not in (0, 1):
                raise ValueError("Manual accuracy must be binary.")
            if not grade["accuracy_eligible"] or grade["accuracy"] is not None:
                raise ValueError("Only pending eligible accuracy may be manually graded.")
            grade["accuracy"] = value
            if grade.get("content_accuracy") is None:
                grade["content_accuracy"] = value
            grade["accuracy_method"] = "independent_manual_review"
        for key, target in (("claims", "claims"), ("citations", "citation_checks")):
            allowed = {v["index"]: v for v in grade[target]}
            indices = set()
            for item in entry.get(key, []):
                value, index = item.get("score"), item.get("index")
                if type(index) is not int or index not in allowed or index in indices:
                    raise ValueError("Unknown or duplicate claim/citation index.")
                indices.add(index)
                if value is None:
                    continue
                if type(value) is not int or value not in (0, 1) or allowed[index]["score"] is not None:
                    raise ValueError("Only pending claim/citation scores may receive binary grades.")
                allowed[index].update(score=value, reason="Independent manual source review.")
        grade["manual_review_required"] = (
            grade["accuracy_eligible"] and grade["accuracy"] is None or
            any(c["score"] is None for c in grade["claims"] + grade["citation_checks"]))
        grade["reviewer"] = grades["reviewer"]
        grade["review_notes"] = entry["notes"]
    return copied



def rescore_run(folder):
    """Regrade saved responses without inference; preserve original experiment files."""
    folder = Path(folder)
    dataset, truth = load_benchmark(folder / "benchmark_snapshot.json", folder / "ground_truth_snapshot.json")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if fingerprint(dataset) != manifest["benchmark_sha256"] or fingerprint(truth) != manifest["ground_truth_sha256"]:
        raise ValueError("Saved benchmark or ground truth changed.")
    records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in dataset["questions"]}
    for record in records:
        qid = record["question_id"]
        record["grading"] = grade_response(by_id[qid], truth["questions"][qid],
            record["result"], truth["sources"], record["tool_calls"], record["mode"])
    write_json(folder / "records_rescored.json", records)
    write_csv(folder / "records_rescored.csv", records)
    write_json(folder / "metrics_rescored.json", metrics(records))
    write_json(folder / "review_template_rescored.json", review_template(manifest["run_id"], records, truth))
    write_json(folder / "rescore_manifest.json", {
        "input_records_sha256": file_hash(folder / "records.json"),
        "run_id": manifest["run_id"], "no_inference_performed": True,
        "grading_implementation": {str(p.relative_to(ROOT)): file_hash(p)
                                  for p in sorted((ROOT / "evaluation").glob("*.py"))}})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Validate all 30 questions, ground truth and source hashes.")
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("single", "multi"), required=True)
    run.add_argument("--model", default="qwen3:8b")
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--ids", help="Comma-separated pilot IDs; omission runs all 30.")
    run.add_argument("--answer-mode", choices=("extractive", "paraphrase"), default="extractive")
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--output-dir", type=Path)
    rescore = commands.add_parser("rescore")
    rescore.add_argument("--run-dir", type=Path, required=True)
    grade = commands.add_parser("grade")
    grade.add_argument("--run-dir", type=Path, required=True)
    grade.add_argument("--grades", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            dataset, truth = load_benchmark()
            print(json.dumps({"benchmark": dataset["benchmark_id"], "questions": len(dataset["questions"]),
                              "source_hashes_verified": True, "pending_source": [
                                  k for k, v in truth["questions"].items() if v["verification_status"] == "pending_source"]}, indent=2))
        elif args.command == "run":
            result = asyncio.run(run_benchmark(
                mode=args.mode, model=args.model, repetitions=args.repetitions,
                ids=args.ids.split(",") if args.ids else None, output_dir=args.output_dir,
                timeout=args.timeout, answer_mode=args.answer_mode))
            print(json.dumps({"run_dir": result["run_dir"], "complete": result["manifest"]["complete"],
                              "metrics": {k: v for k, v in result["metrics"].items() if k != "stability"}}, indent=2))
            return 0 if result["manifest"]["complete"] else 1
        elif args.command == "rescore":
            records = rescore_run(args.run_dir)
            print(json.dumps({"rescored_records": len(records), "no_inference_performed": True}))
        else:
            manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
            records_path = args.run_dir / "records_rescored.json"
            if not records_path.exists():
                records_path = args.run_dir / "records.json"
            records = json.loads(records_path.read_text(encoding="utf-8"))
            grades = json.loads(args.grades.read_text(encoding="utf-8"))
            graded = apply_manual_grades(records, grades, manifest["run_id"])
            write_json(args.run_dir / "records_graded.json", graded)
            write_csv(args.run_dir / "records_graded.csv", graded)
            write_json(args.run_dir / "metrics_graded.json", metrics(graded))
            write_json(args.run_dir / "grading_manifest.json", {
                "input_records_sha256": file_hash(records_path),
                "grades_sha256": file_hash(args.grades),
                "reviewer": grades["reviewer"],
                "grading_implementation": {str(p.relative_to(ROOT)): file_hash(p)
                                          for p in sorted((ROOT / "evaluation").glob("*.py"))}})
            print(json.dumps({"graded_records": len(graded), "raw_records_unchanged": True}))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
