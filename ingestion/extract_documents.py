"""Extract page text and layout with selective OCR. No processing runs on import."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import unicodedata

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path, raw_root):
    """Validate every row before creating output."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"document_id", "filename", "title"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs document_id, filename and title")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no document rows")
    ids, paths, errors = set(), set(), []
    raw_root = raw_root.resolve()
    for number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            errors.append(f"Row {number}: malformed CSV")
            continue
        row.update({key: value.strip() for key, value in row.items()})
        doc_id, filename = row["document_id"], row["filename"]
        if not doc_id or doc_id in ids:
            errors.append(f"Row {number}: empty or duplicate document ID")
        ids.add(doc_id)
        path = (raw_root / filename).resolve()
        if not filename or Path(filename).is_absolute() or not path.is_relative_to(raw_root):
            errors.append(f"Row {number}: filename must be inside raw_documents")
        elif path.suffix.lower() != ".pdf" or not path.is_file():
            errors.append(f"Row {number}: missing or invalid PDF: {filename}")
        if path in paths:
            errors.append(f"Row {number}: duplicate PDF path")
        paths.add(path)
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def quality(text):
    """Heuristics, not confidence scores or semantic verification."""
    chars = [char for char in text if not char.isspace()]
    bad = sum(char == "\ufffd" or unicodedata.category(char) in {"Cc", "Co", "Cs"}
              for char in chars)
    return {"characters": len(chars),
            "alphanumeric": sum(char.isalnum() for char in chars),
            "bad_character_ratio": bad / max(len(chars), 1)}


def suspicious(metrics):
    return metrics["bad_character_ratio"] > 0.02 or (
        metrics["characters"] >= 50
        and metrics["alphanumeric"] / metrics["characters"] < 0.25)


def valid_bbox(bbox, width, height):
    """Reject sentinel, inverted, nonfinite and off-page text coordinates."""
    return (len(bbox) == 4 and all(math.isfinite(v) for v in bbox)
            and 0 <= bbox[0] < bbox[2] <= width + 2
            and 0 <= bbox[1] < bbox[3] <= height + 2)


def collect_blocks(page, textpage=None, matrix=None):
    options = {"textpage": textpage} if textpage is not None else {}
    blocks, rejected = [], 0
    for b in page.get_text("blocks", sort=False, **options):
        if b[6] != 0:
            continue
        rect = pymupdf.Rect(b[:4])
        if matrix is not None:
            rect = rect * matrix
        bbox = list(rect)
        if not valid_bbox(bbox, page.rect.width, page.rect.height):
            rejected += 1
            continue
        if b[4].strip():
            blocks.append({"bbox": bbox, "text": b[4].strip(),
                           "block_number": b[5]})
    blocks.sort(key=lambda block: (round(block["bbox"][1], 1), block["bbox"][0]))
    return blocks, rejected


def blocks_text(blocks):
    # Preserve paragraphs instead of reconstructing lines across unrelated columns.
    return "\n\n".join(block["text"] for block in blocks)


def raster_ocr(page, display_clip, args):
    """OCR an upright rendered region in an unsaved, zero-rotation PDF.

    Coordinates returned are in the original page's displayed coordinate space.
    Source PDFs are never modified. No intermediate image/PDF files are saved.
    """
    pix = page.get_pixmap(dpi=args.dpi, clip=display_clip,
                          colorspace=pymupdf.csRGB, alpha=False)
    with pymupdf.open() as temporary:
        upright = temporary.new_page(width=display_clip.width, height=display_clip.height)
        upright.insert_image(upright.rect, stream=pix.tobytes("png"))
        tp = upright.get_textpage_ocr(language=args.language, dpi=args.dpi,
                                      full=True, tessdata=args.tessdata)
        blocks, rejected = collect_blocks(upright, tp)
        for block in blocks:
            box = block["bbox"]
            block["bbox"] = [box[0] + display_clip.x0, box[1] + display_clip.y0,
                             box[2] + display_clip.x0, box[3] + display_clip.y0]
        return blocks, rejected


def poppler_page_text(pdf_path, page_number):
    """Read one page with layout preserved; save no intermediate files."""
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext is not installed")
    result = subprocess.run(
        [executable, "-f", str(page_number), "-l", str(page_number),
         "-layout", "-enc", "UTF-8", str(pdf_path.resolve()), "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, check=True,
    )
    return result.stdout.replace("\f", "").strip()


def extract_page(page, metadata, digest, args):
    native_blocks, native_rejected = collect_blocks(page, matrix=page.rotation_matrix)
    native = blocks_text(native_blocks)
    original_native = native
    words = native.split()
    fragmented = (
        len(words) >= 30
        and sum(len(word) == 1 for word in words) / len(words) > 0.35
    )
    layout_candidate = (
        fragmented or "table" in native.lower()
        or metadata.get("type") == "academic_calendar"
    )
    poppler_text, poppler_error, using_poppler = None, None, False
    if native_rejected or not native.strip() or suspicious(quality(native)) or layout_candidate:
        try:
            poppler_text = poppler_page_text(
                args.raw_root / metadata["filename"], page.number + 1)
            candidate_quality = quality(poppler_text)
            if candidate_quality["alphanumeric"] > 0 and not suspicious(candidate_quality):
                native, using_poppler = poppler_text, True
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            poppler_error = f"{type(exc).__name__}: {exc}"
    metrics = quality(native)
    bounds = page.rect
    image_rects = []
    for info in page.get_image_info():
        rect = (pymupdf.Rect(info["bbox"]) * page.rotation_matrix) & bounds
        if not rect.is_empty and not rect.is_infinite:
            image_rects.append(rect)
    area = max(bounds.get_area(), 1)
    regions = []
    for rect in image_rects:
        # Small graphics and logos do not trigger OCR. Covered image text is
        # already represented by the native layer; do not OCR it again.
        coverage = sum((pymupdf.Rect(b["bbox"]) & rect).get_area()
                       for b in native_blocks) / max(rect.get_area(), 1)
        if rect.get_area() / area >= args.min_image_fraction and coverage < 0.05:
            if not any((rect & old).get_area() / max(rect.get_area(), 1) > 0.8
                       for old in regions):
                regions.append(rect)
    reasons = []
    garbled = suspicious(metrics)
    full = (garbled
            or (not native.strip() and bool(image_rects))
            or (metrics["alphanumeric"] < 40
                and any(rect.get_area() / area >= 0.5 for rect in image_rects))
            or (not native.strip() and bool(page.get_drawings())))
    attempted = full or bool(regions)
    text = native
    blocks = [] if using_poppler else native_blocks
    method = "pdftotext_layout" if using_poppler else "native"
    ocr_blocks, ocr_text, ocr_metrics, ocr_error = [], None, None, None
    invalid_ocr = 0
    if native_rejected:
        reasons.append("invalid_native_block_coordinates")
    if garbled:
        reasons.append("native_text_may_be_garbled")
    if poppler_error:
        reasons.append("layout_fallback_failed")
    if attempted:
        try:
            for clip in ([bounds] if full else regions):
                found, rejected = raster_ocr(page, clip, args)
                ocr_blocks.extend(found)
                invalid_ocr += rejected
            ocr_text = blocks_text(ocr_blocks)
            ocr_metrics = quality(ocr_text)
            if invalid_ocr or not ocr_metrics["alphanumeric"] or suspicious(ocr_metrics):
                reasons.append("ocr_output_rejected")
                method = "native_fallback" if native.strip() else "failed"
            else:
                if full:
                    blocks = ocr_blocks
                    text = blocks_text(blocks)
                    method = "ocr_full_page"
                elif using_poppler:
                    # Poppler text and PyMuPDF blocks have different provenance.
                    # Keep OCR supplements separately until reviewed.
                    blocks = []
                    text = native
                    method = "pdftotext_layout"
                    reasons.append("image_text_requires_manual_merge")
                else:
                    blocks = native_blocks + ocr_blocks
                    blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
                    text = blocks_text(blocks)
                    method = "native_with_selective_ocr"
            reasons.append("ocr_output_requires_visual_review")
        except Exception as exc:
            ocr_error = f"{type(exc).__name__}: {exc}"
            reasons.append("ocr_failed")
            method = "native_fallback" if native.strip() else "failed"
    elif not native.strip():
        method = "blank_candidate"
        reasons.append("possibly_blank_page_requires_review")
    # This is a layout warning, not a language or correctness classifier.
    words = text.split()
    single_ratio = sum(len(word) == 1 for word in words) / max(len(words), 1)
    if len(words) >= 30 and single_ratio > 0.35:
        reasons.append("suspicious_word_fragmentation")
    invalid_final = sum(not valid_bbox(b["bbox"], bounds.width, bounds.height)
                        for b in blocks)
    if invalid_final:
        reasons.append("invalid_final_block_coordinates")
        blocks = [b for b in blocks if valid_bbox(b["bbox"], bounds.width, bounds.height)]
        text = blocks_text(blocks)
    return {
        "schema_version": 2, "document_id": metadata["document_id"],
        "filename": metadata["filename"], "source_sha256": digest,
        "page_number": page.number + 1, "page_label": page.get_label(),
        "page_width": bounds.width, "page_height": bounds.height,
        "rotation": page.rotation,
        "coordinate_system": "displayed page PDF points; rotation applied",
        "text": text, "blocks": blocks, "native_text": original_native,
        "layout_text": poppler_text, "layout_error": poppler_error,
        "text_has_matching_blocks": method in {
            "native", "ocr_full_page", "native_with_selective_ocr"
        },
        "native_blocks": native_blocks, "native_quality": quality(original_native),
        "image_metrics": {"count": len(image_rects),
                          "ocr_candidate_regions": len(regions),
                          "largest_image_fraction": max(
                              (r.get_area() / area for r in image_rects), default=0)},
        "ocr_text": ocr_text, "ocr_blocks": ocr_blocks,
        "ocr_quality": ocr_metrics, "ocr_required": attempted,
        "ocr_attempted": attempted, "ocr_error": ocr_error,
        "invalid_native_blocks": native_rejected,
        "invalid_ocr_blocks": invalid_ocr, "invalid_final_blocks": invalid_final,
        "single_character_word_ratio": single_ratio,
        "extraction_method": method, "needs_review": bool(reasons),
        "review_reasons": reasons, "metadata": dict(metadata),
    }


def run(args):
    rows = load_manifest(args.manifest, args.raw_root)
    if args.document_id:
        selected = set(args.document_id)
        unknown = selected - {row["document_id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown document IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["document_id"] in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "document_ids": [row["document_id"] for row in rows],
        "language": args.language, "dpi": args.dpi,
        "pymupdf_version": pymupdf.__version__,
        "documents_opened": 0, "pages_expected": 0, "pages_written": 0,
        "pages_needing_review": 0, "ocr_failures": 0, "errors": [],
        "review_pages": [], "rotated_pages": 0, "invalid_block_pages": 0,
        "notice": "Extraction is not source verification or approval for retrieval.",
    }
    methods = Counter()
    # Replace this run's output, never append duplicate records.
    with (args.output_dir / "extracted_pages.jsonl").open("w", encoding="utf-8") as stream:
        for metadata in rows:
            try:
                path = args.raw_root / metadata["filename"]
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                with pymupdf.open(path) as document:
                    if document.needs_pass:
                        raise ValueError("PDF requires a password")
                    report["documents_opened"] += 1
                    report["pages_expected"] += len(document)
                    for index in range(len(document)):
                        try:
                            record = extract_page(document[index], metadata, digest, args)
                        except Exception as exc:
                            record = {
                                "schema_version": 2, "document_id": metadata["document_id"],
                                "filename": metadata["filename"], "page_number": index + 1,
                                "source_sha256": digest, "metadata": dict(metadata),
                                "text": "", "blocks": [], "extraction_method": "failed",
                                "needs_review": True, "review_reasons": ["page_extraction_failed"],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            report["errors"].append({
                                "document_id": metadata["document_id"],
                                "page_number": index + 1, "error": record["error"]})
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        methods[record["extraction_method"]] += 1
                        report["pages_written"] += 1
                        report["rotated_pages"] += int(bool(record.get("rotation")))
                        report["invalid_block_pages"] += int(any(
                            record.get(key, 0) for key in
                            ("invalid_native_blocks", "invalid_ocr_blocks", "invalid_final_blocks")))
                        if record["needs_review"]:
                            report["review_pages"].append({
                                "document_id": metadata["document_id"],
                                "page_number": index + 1,
                                "reasons": record["review_reasons"],
                                "ocr_error": record.get("ocr_error")})
                        report["pages_needing_review"] += int(record["needs_review"])
                        report["ocr_failures"] += int(bool(record.get("ocr_error")))
                print(f"Extracted: {metadata['document_id']}")
            except Exception as exc:
                report["errors"].append({"document_id": metadata["document_id"],
                                         "error": f"{type(exc).__name__}: {exc}"})
    report["methods"] = dict(methods)
    report["requires_review"] = bool(report["pages_needing_review"])
    report["ready_for_cleaning"] = not (report["requires_review"] or report["errors"])
    report["quality_notice"] = (
        "No runtime errors does not establish correct reading order or OCR accuracy. "
        "Review OCR pages and tables against the originals.")
    report["output_dir"] = str(args.output_dir.resolve())
    report["has_failures"] = bool(
        report["errors"] or report["ocr_failures"]
        or methods["failed"] or methods["native_fallback"]
        or any("invalid_final_block_coordinates" in page["reasons"]
               for page in report["review_pages"])
    )
    report["ready_for_cleaning"] = not (report["requires_review"] or report["has_failures"])
    (args.output_dir / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["has_failures"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/document_manifest.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw_documents")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/trial_4")
    parser.add_argument("--document-id", action="append", help="Repeat to select documents")
    parser.add_argument("--language", default="eng", help="Tesseract languages; e.g. eng+hin")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tessdata", help="Directory containing Tesseract traineddata")
    parser.add_argument("--min-image-fraction", type=float, default=0.12,
                        help="Minimum page area of an image considered for selective OCR")
    args = parser.parse_args()
    if not 0 < args.min_image_fraction <= 1:
        parser.error("--min-image-fraction must be in (0, 1]")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
