"""Create page-bounded character chunks; character counts are not model tokens."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chunk_page(page, size=1800, overlap=240):
    text = page["cleaned_text"]
    start, index = 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer paragraph or line ends, without creating very small chunks.
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut < 0:
                cut = text.rfind("\n", start + size // 2, end)
            if cut < 0:
                cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        content = text[start:end].strip()
        if content:
            digest = hashlib.sha256(
                f'{page["document_id"]}|{page["page_number"]}|{start}|{end}|{content}'.encode()
            ).hexdigest()[:24]
            metadata = dict(page.get("metadata", {}))
            metadata.update({
                "document_id": page["document_id"], "filename": page["filename"],
                "page_start": page["page_number"], "page_end": page["page_number"],
                "source_sha256": page.get("source_sha256", ""),
                "needs_review": bool(page.get("needs_review")),
                "review_reasons": json.dumps(page.get("review_reasons", [])),
                "extraction_method": page.get("extraction_method", "unknown"),
                "chunk_index": index, "character_start": start, "character_end": end,
                "chunking_method": "page_bounded_characters_v1",
            })
            yield {"chunk_id": digest, "text": content, "metadata": metadata}
            index += 1
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Avoid starting in the middle of a word where possible.
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/processed/cleaned_run_4/cleaned_pages.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/chunked_run_4")
    parser.add_argument("--chunk-size", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=240)
    args = parser.parse_args()
    if not 0 <= args.overlap < args.chunk_size // 2 or args.chunk_size < 100:
        parser.error("Use chunk-size >= 100 and 0 <= overlap < half chunk-size")
    output = args.output_dir / "chunks.jsonl"
    report_path = args.output_dir / "chunking_report.json"
    if output.exists() or report_path.exists():
        parser.error("Output exists; choose a new --output-dir")
    chunks, seen = [], set()
    pages, blocked = 0, []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        key = (page["document_id"], page["page_number"])
        if key in seen:
            parser.error(f"Duplicate page: {key}")
        seen.add(key)
        pages += 1
        if page.get("cleaning_status") == "blocked" or not page["cleaned_text"].strip():
            blocked.append(list(key))
            continue
        chunks.extend(chunk_page(page, args.chunk_size, args.overlap))
    if not chunks:
        parser.error("No usable pages found")
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    report = {
        "pages_read": pages, "blocked_pages": blocked, "chunks": len(chunks),
        "review_flagged_chunks": sum(c["metadata"]["needs_review"] for c in chunks),
        "chunks_by_document": dict(Counter(c["metadata"]["document_id"] for c in chunks)),
        "size_unit": "characters; not embedding-model tokens",
        "chunk_size": args.chunk_size, "overlap": args.overlap,
        "limitations": "Tables may span chunks or pages. Review flags persist; table reconstruction is not performed.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
