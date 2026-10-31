"""Conservative page cleaning; preserve source text, layout and review flags.

Run with python -m ingestion.clean_documents.
Cleaning does not approve OCR output or reconstruct tables.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def clean_text(text):
    """Keep horizontal spacing, Unicode joiners, numbers and policy wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.expandtabs(4)
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    cleaned, blanks = [], 0
    for line in lines:
        if not line:
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
        cleaned.append(line)
    return "\n".join(cleaned).strip("\n")


def numeric_tokens(text):
    return re.findall(r"\d+(?:[.,]\d+)*|[%٪]", text)


def process_records(input_path):
    records, keys = [], set()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()),
        "pages_read": 0, "pages_written": 0, "pages_changed": 0,
        "empty_pages": 0, "extraction_failed_pages": 0,
        "pages_needing_review": 0, "characters_before": 0,
        "characters_after": 0, "review_pages": [], "methods": {},
        "notice": (
            "Cleaning preserves review flags; it does not verify sources, "
            "repair OCR wording, reconstruct tables, or approve retrieval."
        ),
    }
    methods = Counter()
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(source, dict):
                raise ValueError(f"Line {line_number}: expected a page object")
            doc_id, page_number = source.get("document_id"), source.get("page_number")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"Line {line_number}: missing document_id")
            if type(page_number) is not int or page_number < 1:
                raise ValueError(f"Line {line_number}: invalid page_number")
            key = (doc_id, page_number)
            if key in keys:
                raise ValueError(f"Line {line_number}: duplicate page {key}")
            keys.add(key)
            original = source.get("text")
            if not isinstance(original, str):
                raise ValueError(f"Line {line_number}: text must be a string")
            reasons = source.get("review_reasons", [])
            if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
                raise ValueError(f"Line {line_number}: invalid review_reasons")
            reasons = list(reasons)
            cleaned = clean_text(original)
            # Guard against accidental changes to academic values.
            if numeric_tokens(original) != numeric_tokens(cleaned):
                raise ValueError(f"Line {line_number}: numeric values changed")
            empty = not cleaned.strip()
            failed = source.get("extraction_method") == "failed"
            if empty and "empty_cleaned_text" not in reasons:
                reasons.append("empty_cleaned_text")
            if failed and "page_extraction_failed" not in reasons:
                reasons.append("page_extraction_failed")
            record = dict(source)
            # Original text and all original extraction evidence remain unchanged.
            record["cleaning_schema_version"] = 1
            record["cleaned_text"] = cleaned
            record["cleaning_changed"] = cleaned != original
            record["cleaning_status"] = (
                "blocked" if failed or empty else
                "review_required" if source.get("needs_review") or reasons else "cleaned"
            )
            record["needs_review"] = bool(source.get("needs_review") or reasons)
            record["review_reasons"] = reasons
            records.append(record)
            methods[source.get("extraction_method", "unknown")] += 1
            report["pages_read"] += 1
            report["pages_changed"] += int(cleaned != original)
            report["empty_pages"] += int(empty)
            report["extraction_failed_pages"] += int(failed)
            report["characters_before"] += len(original)
            report["characters_after"] += len(cleaned)
            report["pages_needing_review"] += int(record["needs_review"])
            if record["needs_review"]:
                report["review_pages"].append({
                    "document_id": doc_id, "page_number": page_number,
                    "cleaning_status": record["cleaning_status"], "reasons": reasons,
                })
    if not records:
        raise ValueError("Input contains no page records")
    report["methods"] = dict(methods)
    report["documents"] = len({r["document_id"] for r in records})
    report["has_failures"] = bool(
        report["empty_pages"] or report["extraction_failed_pages"])
    report["requires_review"] = bool(report["pages_needing_review"])
    report["ready_for_indexing"] = not (
        report["has_failures"] or report["requires_review"])
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/processed/full_run_4/extracted_pages.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/cleaned_run_4")
    args = parser.parse_args()
    pages_path = args.output_dir / "cleaned_pages.jsonl"
    report_path = args.output_dir / "cleaning_report.json"
    if pages_path.exists() or report_path.exists():
        parser.error("Output already exists; choose a new --output-dir to preserve it")
    try:
        records, report = process_records(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # Validate all input records before creating output.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with pages_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report["pages_written"] = len(records)
    report["output_dir"] = str(args.output_dir.resolve())
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["has_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
