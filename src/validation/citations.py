"""Validate citation identity and exact quotes, not semantic entailment."""
import json
import re

ABSTENTION = "I don't know from the available academic documents."


def normalized(text):
    return " ".join(text.split())


def validate_policy_output(raw, evidence):
    """Return a structured result; unsupported/malformed model output abstains."""
    rejected = {"status": "insufficient_evidence", "answer": ABSTENTION,
                "citations": [], "validation": "rejected"}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", raw).strip()
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {**rejected, "reason": "Model did not return valid JSON."}
    if not isinstance(payload, dict):
        return {**rejected, "reason": "Invalid response object."}
    status = payload.get("status")
    if status in {"out_of_scope", "insufficient_evidence", "requires_clarification"}:
        # Do not pass arbitrary model-generated factual claims through this route.
        answers = {
            "out_of_scope": "I don't know. This helpdesk supports academic policies, courses, schedules and academic email drafts.",
            "insufficient_evidence": ABSTENTION,
            "requires_clarification": "Please specify your academic program, admission year, and degree track.",
        }
        return {"status": status, "answer": answers[status], "citations": [],
                "validation": "abstained"}
    claims = payload.get("claims")
    if status != "answer" or not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        return {**rejected, "reason": "Missing cited claims."}
    by_id = {item["chunk_id"]: item for item in evidence}
    answers, citations, claim_evidence = [], [], []
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            return {**rejected, "reason": "Invalid claim object."}
        text, supports = claim.get("text"), claim.get("supports")
        if not isinstance(text, str) or not text.strip() or not isinstance(supports, list) or not supports:
            return {**rejected, "reason": "Every claim needs supporting evidence."}
        refs, per_claim_citations, seen_supports = [], [], set()
        for support in supports:
            if not isinstance(support, dict):
                return {**rejected, "reason": "Invalid citation."}
            chunk_id = support.get("chunk_id")
            if not isinstance(chunk_id, str):
                return {**rejected, "reason": "Invalid chunk ID."}
            source = by_id.get(chunk_id)
            quote = support.get("quote")
            if not source or not isinstance(quote, str) or len(normalized(quote)) < 12:
                return {**rejected, "reason": "Unknown source or missing quote."}
            if normalized(quote) not in normalized(source["text"]):
                return {
                    **rejected, "reason": "Supporting quote does not match source.",
                    "diagnostic": {
                        "claim": text, "chunk_id": support.get("chunk_id"),
                        "model_quote": quote, "source_text": source["text"],
                    },
                }
            if not source.get("document_id") or not source.get("page_start"):
                return {**rejected, "reason": "Source lacks page citation metadata."}
            label = f'{source["document_id"]}, PDF p. {source["page_start"]}'
            refs.append(label)
            citation = {
                "claim_index": claim_index,
                "chunk_id": source["chunk_id"], "document_id": source["document_id"],
                "page": source["page_start"], "source_url": source.get("source_url", ""),
                "quote": quote, "needs_review": source.get("needs_review", True),
            }
            support_key = (source["chunk_id"], normalized(quote))
            if support_key not in seen_supports:
                per_claim_citations.append(citation)
                citations.append(citation)
                seen_supports.add(support_key)
        answers.append(text.strip() + " [" + "; ".join(dict.fromkeys(refs)) + "]")
        claim_evidence.append({"text": text.strip(), "citations": per_claim_citations})
    warnings = []
    used = [by_id[c["chunk_id"]] for c in citations]
    if any(s.get("needs_review", True) for s in used):
        warnings.append("Some cited text is flagged for OCR or table review.")
    if any(s.get("policy_status", "unknown") == "unknown" for s in used):
        warnings.append("These sources have not been verified as the latest applicable policy.")
    return {"status": "answer", "answer": "\n\n".join(answers),
             "citations": citations, "warnings": warnings, "claims": claims,
            "claim_evidence": claim_evidence,
            "validation": "citation_and_quote_checks_passed",
            "limitation": "Quote matching does not prove semantic support or cohort applicability."}
