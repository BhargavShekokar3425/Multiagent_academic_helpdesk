"""Sequential single/multi x Qwen3/Llama comparison; never installs models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid
import time

from .evaluate import available_models, ollama_request, write_json
from .scoring import ROOT, metrics, file_hash


MODELS = ("qwen3:8b", "llama3.1:8b")
MODES = ("single", "multi")


def prerequisites(inventory=None):
    inventory = available_models() if inventory is None else inventory
    return {"models": [{"model": name, "installed": name in inventory,
                        "digest": inventory.get(name),
                        "prerequisite": "" if name in inventory else "ollama pull " + name}
                       for name in MODELS],
            "installation_performed": False,
            "memory_policy": "Configurations run sequentially; unload other generation models. This is not a 16 GB memory certification."}


def unload_competitors(target):
    """Leave the shared embedding model available; no model weight files deleted."""
    unloaded = []
    for item in ollama_request("/api/ps").get("models", []):
        name = item["name"]
        if name != target and not name.startswith("nomic-embed-text"):
            ollama_request("/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
            unloaded.append(name)
    remaining = [m["name"] for m in ollama_request("/api/ps").get("models", [])
                 if m["name"] != target and not m["name"].startswith("nomic-embed-text")]
    if remaining:
        raise ValueError("Other generation models remain loaded: " + ", ".join(remaining))
    return unloaded


def compare(output_dir=None, repetitions=1, ids=None, timeout=180, answer_mode="extractive",
            inventory=None, execute=None, unload=None, warmup=False):
    if repetitions < 1 or timeout <= 0:
        raise ValueError("Repetitions and timeout must be positive.")
    inventory = available_models() if inventory is None else inventory
    report = {"schema_version": 1, "prerequisites": prerequisites(inventory),
              "repetitions": repetitions, "ids": ids, "answer_mode": answer_mode,
              "sequential": True, "configurations": []}
    folder = Path(output_dir) if output_dir else ROOT / "evaluation/results/raw" / (
        "comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    run_process = execute or subprocess.run
    unload = unload or unload_competitors
    reference = None
    for model in MODELS:
        if model not in inventory:
            for mode in MODES:
                report["configurations"].append({"model": model, "mode": mode,
                    "status": "skipped_missing_model", "prerequisite": "ollama pull " + model})
            continue
        unloaded = unload(model)
        for mode in MODES:
            target = folder / (re.sub(r"[^a-zA-Z0-9_-]", "_", model) + "-" + mode)
            command = [sys.executable, "-m", "evaluation.evaluate", "run", "--mode", mode,
                       "--model", model, "--repetitions", str(repetitions),
                       "--timeout", str(timeout), "--answer-mode", answer_mode, "--output-dir", str(target.resolve())]
            if ids:
                command += ["--ids", ",".join(ids)]
            warmup_record = None
            if warmup:
                resident = any(m["name"] == model for m in ollama_request("/api/ps").get("models", []))
                started = time.perf_counter()
                response = ollama_request("/api/generate", {
                    "model": model, "prompt": "Reply with OK.", "stream": False,
                    "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 8}}, timeout=180)
                warmup_record = {"performed": True, "cold_before_warmup": not resident,
                                 "elapsed_seconds": time.perf_counter() - started,
                                 "load_duration_ns": response.get("load_duration")}
            entry = {"warmup": warmup_record, "model": model, "mode": mode, "command": command, "unloaded_models": unloaded}
            try:
                # A finished subprocess isolates model configuration and ensures
                # no other architecture/model run executes concurrently.
                result = run_process(command, cwd=ROOT, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace",
                                     timeout=(len(ids) if ids else 30) * repetitions * timeout + 120)
                (folder / (target.name + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
                (folder / (target.name + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
                entry["exit_code"] = result.returncode
                if result.returncode != 0:
                    entry["status"] = "failed"
                else:
                    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                    if manifest.get("model_digest") != inventory[model]:
                        raise ValueError("Installed model digest changed during comparison.")
                    signature = manifest["comparison_fingerprint"]
                    if reference is None:
                        reference = signature
                    if signature != reference:
                        raise ValueError("Benchmark/data/settings changed between configurations; comparison rejected.")
                    entry.update(status="completed", run_dir=str(target),
                                 comparison_fingerprint=signature,
                                 metrics=json.loads((target / "metrics.json").read_text(encoding="utf-8")))
            except subprocess.TimeoutExpired:
                entry.update(status="failed", error="Configuration process timed out and was stopped.")
                report["configurations"].append(entry)
                report["aborted_after_timeout"] = True
                report["all_four_configurations_completed"] = False
                write_json(folder / "comparison.json", report)
                return {"output_dir": str(folder), **report}
            report["configurations"].append(entry)
            write_json(folder / "comparison.json", report)
    report["all_four_configurations_completed"] = sum(
        c["status"] == "completed" for c in report["configurations"]) == 4
    report["comparison_fingerprint"] = reference
    write_json(folder / "comparison.json", report)
    return {"output_dir": str(folder), **report}



def summarize(folder):
    """Recompute comparison metrics after manual review; leave raw results intact."""
    folder = Path(folder)
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    for entry in report["configurations"]:
        if entry["status"] != "completed":
            continue
        run_dir = Path(entry["run_dir"])
        graded = run_dir / "records_graded.json"
        rescored = run_dir / "records_rescored.json"
        input_path = rescored if rescored.exists() else run_dir / "records.json"
        if graded.exists():
            provenance = run_dir / "grading_manifest.json"
            if not provenance.exists():
                raise ValueError("Reviewed records need a grading manifest; reapply manual grades.")
            grade_meta = json.loads(provenance.read_text(encoding="utf-8"))
            current = {str(p.relative_to(ROOT)): file_hash(p)
                       for p in sorted((ROOT / "evaluation").glob("*.py"))}
            if (grade_meta["input_records_sha256"] != file_hash(input_path)
                    or grade_meta["grading_implementation"] != current):
                raise ValueError("Manual grades are stale; reapply them to the current scoring version.")
        records_path = graded if graded.exists() else input_path
        entry["metrics"] = metrics(json.loads(records_path.read_text(encoding="utf-8")))
        entry["grades_applied"] = graded.exists()
        entry["rescored"] = rescored.exists()
    report["manual_review_complete"] = all(
        c.get("metrics", {}).get("manual_review_records", 0) == 0
        for c in report["configurations"] if c["status"] == "completed")
    write_json(folder / "comparison_reviewed.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true", help="Check installation only; no inference or downloads.")
    operation.add_argument("--summarize", type=Path, help="Recompute an existing comparison after manual grading.")
    parser.add_argument("--warmup", action="store_true", help="Warm each configuration identically; exclude warmup timing from benchmark latency.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--ids")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--answer-mode", choices=("extractive", "paraphrase"), default="extractive")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            report = summarize(args.summarize)
            print(json.dumps({"output": str(args.summarize / "comparison_reviewed.json"),
                              "manual_review_complete": report["manual_review_complete"]}))
            return 0
        if args.check:
            print(json.dumps(prerequisites(), indent=2))
            return 0
        report = compare(args.output_dir, args.repetitions, args.ids.split(",") if args.ids else None,
                         args.timeout, args.answer_mode, warmup=args.warmup)
        print(json.dumps({"output_dir": report["output_dir"],
                          "configurations": [{"model": c["model"], "mode": c["mode"], "status": c["status"]}
                                             for c in report["configurations"]],
                          "all_four_configurations_completed": report["all_four_configurations_completed"]}, indent=2))
        return 1 if any(c["status"] == "failed" for c in report["configurations"]) else 0
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
