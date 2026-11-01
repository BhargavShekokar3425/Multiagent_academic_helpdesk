"""Validate verbatim excerpts with source-aware structural boundaries."""
import re

from .citations import normalized
from .support import qualifier_errors
from .passages import is_table_passage


def normalized_offsets(text):
    """Map normalized characters back to source offsets; retain paragraph layout."""
    parts, offsets = [], []
    for match in re.finditer(r"\S+", text):
        if parts:
            parts.append(" ")
            offsets.append(match.start())
        parts.append(match.group())
        offsets.extend(range(match.start(), match.end()))
    return "".join(parts), offsets


def historical_position(text, position):
    """Historical and revised sections are independent of sentence boundaries."""
    markers = list(re.finditer(r"\b(existing|revised)\s+policy\s*:", text, re.I))
    active = None
    for marker in markers:
        if marker.start() > position:
            break
        active = marker.group(1).lower()
    # Existing policy remains historical even if its revised portion is in
    # another chunk. Do not label unrelated introductory material historical.
    return active == "existing"


def structural_start(source, position):
    prefix = source[:position]
    trimmed = prefix.rstrip()
    if not trimmed or trimmed.endswith((".", "!", "?")):
        return True
    # Numbered/bulleted list item, including a blockquote marker.
    line_prefix = prefix.rsplit("\n", 1)[-1]
    if re.fullmatch(r"\s*(?:[-*•>]|\d+[.)]|\([a-zA-Z0-9]+\)|[a-zA-Z][.)])\s*", line_prefix):
        return True
    # Named policy labels can share a line with the following sentence.
    if re.search(r"\b(?:revised|existing)\s+policy\s*:\s*$", prefix, re.I):
        return True
    # Other short standalone headings may introduce a paragraph after a colon.
    previous_line = trimmed.rsplit("\n", 1)[-1].strip()
    if (previous_line.endswith(":") and len(previous_line) <= 100
            and not re.search(
                r"\b(if|unless|except|provided|when|subject to|only if)\b",
                previous_line, re.I)):
        return True
    # A real paragraph break is acceptable, but not an unfinished condition
    # separated by PDF layout whitespace.
    gap = prefix[len(trimmed):]
    previous_block = re.split(r"\n\s*\n", trimmed)[-1]
    if gap.count("\n") >= 2 and not trimmed.endswith((",", ";", ":")):
        if not re.search(r"\b(if|unless|provided|subject to|except)\b",
                         previous_block, re.I):
            return True
    return False


def verify_extractive_claims(claims, evidence, question=""):
    by_id = {source["chunk_id"]: source for source in evidence}
    verdicts = []
    for index, claim in enumerate(claims):
        reason = None
        supports = claim.get("supports", [])
        if len(supports) != 1:
            reason = "Use one complete source passage and one citation per extractive claim."
        else:
            quote = normalized(supports[0]["quote"])
            text = normalized(claim["text"])
            source = by_id[supports[0]["chunk_id"]]
            source_text, offsets = normalized_offsets(source["text"])
            positions = [m.start() for m in re.finditer(re.escape(quote), source_text)] if quote else []
            current = [p for p in positions if not historical_position(source_text, p)]
            if text != quote:
                reason = "Extractive claim text must equal its supporting quote; do not paraphrase."
            elif not positions:
                reason = "Quote is not present in the source."
            elif not current:
                reason = "The excerpt belongs to the Existing Policy section and describes a historical rule."
            elif is_table_passage(source["text"], supports[0]["quote"]):
                reason = None
            elif not quote.endswith((".", "!", "?")):
                reason = "Copy the complete passage including its final punctuation."
            elif not any(structural_start(source["text"], offsets[p]) for p in current):
                reason = "The excerpt starts inside a sentence; include its qualifying introduction."
            else:
                issues = qualifier_errors(claim["text"], supports, question)
                if issues:
                    reason = " ".join(issues)
        verdicts.append({"index": index, "supported": reason is None,
                         "reason": reason or "Statement is a verbatim passage at a valid source boundary."})
    return {"supported": bool(verdicts) and all(v["supported"] for v in verdicts),
            "method": "extractive_identity_and_qualifier_checks", "claims": verdicts}
