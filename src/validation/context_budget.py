"""Bounded, query-focused excerpts for the small local model."""
import json
import math
import re
from collections import Counter

from .excerpts import excerpt_catalog


def terms(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "a", "an", "is", "are", "what", "of", "for", "and", "in", "to",
            "under", "according", "please", "tell", "me", "it", "its", "on"}
    result = {word.rstrip("s") for word in words if word not in stop}
    if result & {"exam", "examination"}:
        result |= {"exam", "examination"}
    if result & {"duration", "length"}:
        result |= {"hour", "minute", "duration"}
    return result


def focused_excerpts(evidence, query, max_bytes=9000, max_items=18):
    # Explicit source references narrow selection, not the underlying corpus.
    scoped = [e for e in evidence if e.get("retrieval_role") == "exact_reference"]
    circulars = re.findall(r"\bcircular\s+(\d+)\b", query, re.I)
    if circulars:
        named = [e for e in evidence if any(re.search(
            r"(?:^|_)circular_" + re.escape(n) + r"(?:_|$)", e["document_id"], re.I)
            for n in circulars)]
        scoped = named or scoped
    evidence = scoped or evidence
    catalog = excerpt_catalog(evidence)
    from .extractive import verify_extractive_claims
    by_id = {e["chunk_id"]: e for e in evidence}
    catalog = [item for item in catalog if verify_extractive_claims([{
        "text": item["text"], "supports": [{"chunk_id": item["chunk_id"], "quote": item["text"]}]
    }], [by_id[item["chunk_id"]]], query)["supported"]]
    query_terms = terms(query)
    frequency = Counter(token for item in catalog for token in terms(item["text"]))
    def score(item):
        return sum(math.log(1 + len(catalog) / frequency[t])
                   for t in terms(item["text"]) & query_terms)
    ranked = sorted(enumerate(catalog), key=lambda pair: (-score(pair[1]), pair[0]))
    selected, seen = [], set()
    for _, item in ranked:
        if query_terms and score(item) == 0:
            continue
        # The selected source may overlap another chunk; avoid duplicate prose.
        key = (item["document_id"], item["text"])
        if key in seen:
            continue
        candidate = {k: item[k] for k in ("excerpt_id", "document_id", "page", "text")}
        if len(json.dumps(selected + [candidate], ensure_ascii=False).encode()) > max_bytes:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) >= max_items:
            break
    return selected


def selection_schema(excerpts):
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": [
                "answer", "insufficient_evidence", "requires_clarification", "out_of_scope"]},
            "claims": {"type": "array", "maxItems": 6, "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"excerpt_id": {"type": "string", "enum": [
                    item["excerpt_id"] for item in excerpts]}},
                "required": ["excerpt_id"],
            }},
        },
        "required": ["status", "claims"],
    }
