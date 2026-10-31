"""Shared CSV loading and consistent tool responses."""
import csv
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]




def coverage_metadata(semester):
    """Return explicitly declared dataset coverage for this semester only."""
    path = Path(os.getenv("STRUCTURED_SOURCES_JSON", "data/structured/source_registry.json"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8")).get("coverage", {})
    if not isinstance(value, dict) or normalized(value.get("semester", "")) != normalized(semester):
        return {}
    return value


def source_registry():
    """Read the separately checked HTML source registry; PDFs keep their manifest."""
    path = Path(os.getenv("STRUCTURED_SOURCES_JSON", "data/structured/source_registry.json"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("sources"), dict):
        raise ValueError("Structured source registry needs a sources object")
    return value["sources"]


def registered_source(source_id, semester=""):
    """Resolve a checked HTML location and verify the saved source has not changed."""
    item = source_registry().get(source_id)
    if not item or item.get("review_status") != "source_checked":
        raise ValueError(f"Unregistered or unchecked HTML source: {source_id}")
    if not item.get("source_url", "").startswith("https://") or not item.get("source_location"):
        raise ValueError(f"HTML source {source_id} needs HTTPS URL and source_location")
    if item.get("semester") and semester and item["semester"] != semester:
        raise ValueError(f"HTML source {source_id} does not apply to semester {semester}")
    filename = Path(item["filename"])
    if not filename.is_absolute():
        filename = ROOT / filename
    digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    if digest != item.get("sha256"):
        raise ValueError(f"Saved HTML source changed: {source_id}")
    return {"document_id": source_id, "page": "", "source_type": "html",
            "source_url": item["source_url"], "source_location": item["source_location"],
            "source_sha256": digest, "authority": item.get("authority", "faculty_record")}


def records(variable, default, required):
    load_dotenv(ROOT / ".env")
    path = Path(os.getenv(variable, default))
    if not path.is_absolute():
        path = ROOT / path
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path.name} malformed row {index}")
            cleaned = {k: v.strip() for k, v in row.items()}
            if "source_doc_id" in required:
                if not cleaned["source_doc_id"]:
                    raise ValueError(f"{path.name} row {index} needs source_doc_id")
                page = cleaned["source_page"]
                if page:
                    if cleaned["source_doc_id"] in source_registry():
                        raise ValueError(f"{path.name} row {index}: HTML sources cannot use PDF page numbers")
                    if not page.isdigit() or int(page) < 1:
                        raise ValueError(f"{path.name} row {index} needs positive source_page")
                else:
                    registered_source(cleaned["source_doc_id"], cleaned.get("semester", ""))
                for source_id in filter(None, cleaned.get("additional_source_ids", "").split(";")):
                    registered_source(source_id, cleaned.get("semester", ""))
            for key in ("course_code", "semester"):
                if key in required and not cleaned[key]:
                    raise ValueError(f"{path.name} row {index} has no {key}")
            rows.append(cleaned)
        return rows


def response(status, data=None, evidence=None, message=""):
    return {"status": status, "data": data or {}, "evidence": evidence or [], "message": message}


def sources(rows):
    evidence = []
    seen = set()
    for row in rows:
        primary = ({"document_id": row["source_doc_id"], "page": row["source_page"]}
                   if row.get("source_page") else
                   registered_source(row["source_doc_id"], row.get("semester", "")))
        items = [primary] + [
            registered_source(source_id, row.get("semester", ""))
            for source_id in filter(None, row.get("additional_source_ids", "").split(";"))]
        for item in items:
            key = (item["document_id"], item["page"])
            if key not in seen:
                seen.add(key)
                evidence.append(item)
    return evidence


def normalized(text):
    return " ".join(text.casefold().split())
