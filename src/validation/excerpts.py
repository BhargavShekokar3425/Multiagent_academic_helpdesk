"""Server-owned source excerpts so the model selects IDs instead of copying text."""
import hashlib
import json
import re

from .citations import normalized
from .extractive import historical_position, verify_extractive_claims
from .passages import table_passages


def excerpt_catalog(evidence):
    result = []
    for source in evidence:
        text = normalized(source["text"])
        # Decimal points and the source's "illness etc. may be permitted"
        # abbreviation must not truncate a condition before its permission clause.
        protected = {i for m in re.finditer(
            r"\b(?:Ph\.D\.|M\.Tech\.|M\.Sc\.|B\.Tech\.|B\.Sc\.|e\.g\.|i\.e\.|etc\.)",
            text, re.I) for i in range(m.start(), m.end())}
        boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)
                      if m.start() not in protected]
        segments = []
        previous = 0
        for end in boundaries + ([len(text)] if not boundaries or boundaries[-1] != len(text) else []):
            segments.append((previous, text[previous:end]))
            previous = end
        for position, segment in segments:
            start = position
            revised = list(re.finditer(r"\brevised\s+policy\s*:\s*", segment, re.I))
            if revised:
                start += revised[-1].end()
                segment = segment[revised[-1].end():]
            trimmed = re.sub(r"^\s*(?:[>•*-]\s*|\d+[.)]\s*)?", "", segment)
            start += len(segment) - len(trimmed)
            quote = trimmed.strip()
            if len(quote) < 20 or quote[0].islower() or historical_position(text, start):
                continue
            claim = {"text": quote, "supports": [{
                "chunk_id": source["chunk_id"], "quote": quote}]}
            if not verify_extractive_claims([claim], [source])["supported"]:
                continue
            key = hashlib.sha256(
                f'{source["chunk_id"]}|{start}|{quote}'.encode()).hexdigest()[:16]
            result.append({
                "excerpt_id": "s_" + key, "chunk_id": source["chunk_id"],
                "document_id": source["document_id"],
                "page": source["page_start"], "text": quote,
            })
        for passage in table_passages(source["text"]):
            key = hashlib.sha256((source["chunk_id"] + "|table|" + passage).encode()).hexdigest()[:16]
            result.append({"excerpt_id": "s_" + key, "chunk_id": source["chunk_id"],
                           "document_id": source["document_id"], "page": source["page_start"],
                           "text": passage})
    return result


def resolve_excerpt_selection(raw, evidence):
    """Materialize selected IDs with trusted text; never accept invented IDs."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return raw  # The existing JSON validator handles this failure.
    if not isinstance(payload, dict) or payload.get("status") != "answer":
        return raw
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return raw
    # Retain the existing strict verbatim path for older integrations.
    if not any(isinstance(c, dict) and "excerpt_id" in c for c in claims):
        return raw
    catalog = {item["excerpt_id"]: item for item in excerpt_catalog(evidence)}
    resolved, seen = [], set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"excerpt_id"}:
            raise ValueError("Select excerpt_id only; do not supply replacement text or quotes.")
        key = claim["excerpt_id"]
        if not isinstance(key, str) or key not in catalog:
            raise ValueError("Selected excerpt ID was not provided by the retrieval tool.")
        if key in seen:
            continue
        seen.add(key)
        item = catalog[key]
        resolved.append({"text": item["text"], "supports": [{
            "chunk_id": item["chunk_id"], "quote": item["text"]}]})
    return json.dumps({"status": "answer", "claims": resolved}, ensure_ascii=False)
