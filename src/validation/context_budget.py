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
    catalog = excerpt_catalog(evidence)
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
