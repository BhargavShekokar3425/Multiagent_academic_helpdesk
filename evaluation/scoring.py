"""Evidence-based scoring. Similarity and model self-review are never ground truth."""
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    return " ".join(str(value).split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(questions_path=None, truth_path=None, check_sources=True):
    questions_path = Path(questions_path or ROOT / "evaluation/questions.json")
    truth_path = Path(truth_path or ROOT / "evaluation/ground_truth.json")
    dataset = json.loads(questions_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != 30 or len(set(ids)) != 30 or set(ids) != set(truth["questions"]):
        raise ValueError("Benchmark needs exactly 30 unique matching question/ground-truth IDs.")
    distribution = {category: sum(q["category"] == category for q in questions)
                    for category in ("policy", "schedule", "email", "out_of_scope", "unsupported")}
    if distribution != {"policy": 10, "schedule": 8, "email": 5, "out_of_scope": 4, "unsupported": 3}:
        raise ValueError("Incorrect benchmark category distribution.")
    if dataset["benchmark_id"] != truth["benchmark_id"]:
        raise ValueError("Benchmark identifiers differ.")
    if sum(bool(q.get("multi_intent")) for q in questions) < 2:
        raise ValueError("At least two multi-intent cases are required.")
    for q in questions:
        gt = truth["questions"][q["id"]]
        if not q.get("question") or q["difficulty"] not in {"simple", "medium", "hard"}:
            raise ValueError("Invalid question text/difficulty.")
        if set(q["required_source_ids"]) - truth["sources"].keys():
            raise ValueError("Unknown required source ID.")
        if q["required_source_ids"] != gt["required_source_ids"] or q["expected_route"] != gt["expected_route"]:
            raise ValueError("Question and ground-truth contracts differ.")
        if gt["verification_status"] == "pending_source" and q["currently_answerable"]:
            raise ValueError("Pending-source facts cannot be answerable.")
    if check_sources:
        verify_sources(truth)
    return dataset, truth


def verify_sources(truth):
    """Fail closed on missing/changed files; never refresh expected hashes silently."""
    expected = dict(truth["dataset_files"])
    expected.update({s["filename"]: s["sha256"] for s in truth["sources"].values()})
    for filename, digest in expected.items():
        path = ROOT / filename
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError("Source snapshot changed or unavailable: " + filename)
    return expected


def leaves(result):
    sections = result.get("sections")
    return sections if isinstance(sections, list) else [result]


def policy_claims(result):
    return [claim for section in leaves(result) for claim in section.get("claims", [])
            if isinstance(claim, dict)]


def check_citation(citation, sources, semester=""):
    source = sources.get(citation.get("document_id"))
    if not source:
        return {"score": 0, "reason": "Unknown source identity."}
    if citation.get("source_url") and citation["source_url"] != source["source_url"]:
        return {"score": 0, "reason": "Source URL mismatch."}
    if source["type"] == "html":
        if semester and source.get("semester") and source["semester"] != semester:
            return {"score": 0, "reason": "HTML source applies to a different semester."}
        if (citation.get("page") not in ("", None) or
                citation.get("source_url") != source["source_url"] or
                citation.get("source_location") != source["source_location"]):
            return {"score": 0, "reason": "HTML source location/URL mismatch."}
        if citation.get("source_sha256") not in (None, source["sha256"]):
            return {"score": 0, "reason": "HTML source hash mismatch."}
        return {"score": 1, "reason": "Checked source identity/location; this is not semantic entailment."}
    page = citation.get("page")
    if not isinstance(page, (str, int)) or isinstance(page, bool):
        return {"score": 0, "reason": "Invalid PDF page."}
    try:
        number = int(page)
        if str(page).strip() != str(number) or not 1 <= number <= source.get("page_count", 0):
            raise ValueError
    except (ValueError, TypeError):
        return {"score": 0, "reason": "PDF page outside source bounds."}
    reference = source.get("pages", {}).get(str(number))
    if not reference:
        return {"score": None, "reason": "PDF page needs independent manual review."}
    quote = citation.get("quote")
    if not isinstance(quote, str) or len(normalized(quote)) < 12:
        return {"score": 0, "reason": "Missing PDF supporting quote."}
    if normalized(quote) not in normalized(reference["reference_text"]):
        return {"score": 0, "reason": "Quote is absent from the source-pinned reference page."}
    return {"score": 1, "reason": "PDF identity, reviewed page and continuous quote match."}


def structured_checks(result, gt):
    checks = gt.get("checks", {})
    if not checks:
        return []
    section = next((s for s in leaves(result) if s.get("tool") == checks.get("tool")), {})
    data = section.get("data", {})
    verdicts = []
    def check(name, passed):
        verdicts.append({"criterion": name, "score": int(bool(passed))})
    if checks.get("tool"):
        check("tool_result_available", bool(section) and section.get("status") == "answer")
    for key in ("course_code", "course_name", "semester", "sent", "requires_student_review", "coverage_verified"):
        if key in checks:
            check(key, data.get(key) == checks[key])
    for key in ("to",):
        if key in checks:
            check(key, sorted(data.get(key, [])) == sorted(checks[key]))
    if "instructor_names" in checks:
        check("instructor_names", sorted(r.get("instructor_name", "") for r in data.get("instructors", []))
              == sorted(checks["instructor_names"]))
    if "emails" in checks:
        check("emails", sorted(r.get("email", "") for r in data.get("instructors", []))
              == sorted(checks["emails"]))
    if "offering_codes" in checks:
        check("offering_codes", sorted({r.get("course_code") for r in data.get("courses", [])})
              == sorted(checks["offering_codes"]))
        check("partial_coverage_disclosed", bool(re.search(
            r"partial|complete offering list is not established", section.get("answer", ""), re.I)))
    if "subject_contains" in checks:
        check("subject", checks["subject_contains"].lower() in data.get("subject", "").lower())
    for token in checks.get("body_contains", []):
        check("placeholder:" + token, token in data.get("body", ""))
    return verdicts


def grade_response(question, gt, result, sources, tool_calls, mode):
    status = result.get("status")
    expected = status in gt["expected_statuses"]
    if question["expected_behavior"] in {"abstain", "missing_record"}:
        # A coordinator may wrap an out-of-scope child response as
        # insufficient_evidence. That is not a correct academic abstention.
        expected = expected and not any(
            s.get("status") in {"out_of_scope", "answer"} for s in leaves(result))
        expected = expected and not result.get("citations") and not policy_claims(result)
    citations = result.get("citations", [])
    citation_checks = [{"index": i, **check_citation(c, sources, question["context"]["semester"])} for i, c in enumerate(citations)]
    cited = {c.get("document_id") for c in citations}
    required = set(gt["required_source_ids"])
    coverage = {"matched": len(required & cited), "required": len(required),
                "missing": sorted(required - cited)}
    claims = []
    for index, claim in enumerate(policy_claims(result)):
        supports = claim.get("citations", [])
        verdicts = [check_citation(c, sources, question["context"]["semester"]) for c in supports]
        score, reason = None, "Paraphrased claim requires independent manual source review."
        if not supports or any(v["score"] == 0 for v in verdicts):
            score, reason = 0, "Missing or incorrect supporting citation."
        elif len(supports) == 1 and normalized(claim.get("text", "")) == normalized(supports[0].get("quote", "")):
            if verdicts[0]["score"] == 1:
                from src.validation.extractive import verify_extractive_claims
                cite = supports[0]
                page = sources[cite["document_id"]]["pages"][str(cite["page"])]
                evidence = [{"chunk_id": cite.get("chunk_id", "reference"), "text": page["reference_text"]}]
                candidate = [{"text": claim["text"], "supports": [{
                    "chunk_id": evidence[0]["chunk_id"], "quote": cite["quote"]}]}]
                verified = verify_extractive_claims(candidate, evidence, question["question"])
                score = int(verified["supported"])
                reason = "Source-pinned verbatim identity/boundary/qualifier checks; relevance is separate."
        claims.append({"index": index, "text": claim.get("text", ""), "score": score, "reason": reason})
    deterministic = structured_checks(result, gt)
    eligible = question["currently_answerable"] and gt["verification_status"] == "source_verified"
    accuracy, method = None, "not_applicable"
    if eligible:
        if not expected:
            accuracy, method = 0, "incorrect_response_status"
        elif gt["grading"] == "structured":
            if any(v["score"] == 0 for v in deterministic) or coverage["missing"] or any(c["score"] == 0 for c in citation_checks):
                accuracy, method = 0, "structured_criteria_failed"
            elif gt.get("checks", {}).get("tool") == "email_draft":
                section = next((s for s in leaves(result) if s.get("tool") == "email_draft"), {})
                body = normalized(section.get("data", {}).get("body", ""))
                if body in [normalized(x) for x in gt.get("acceptable_email_bodies", [])]:
                    accuracy, method = 1, "frozen_template_and_structured_checks"
                else:
                    method = "manual_email_review_required"
            else:
                accuracy, method = 1, "source_backed_structured_checks"
        else:
            method = "manual_policy_and_completeness_review_required"
    agents = [c.get("agent") for c in result.get("debug", {}).get("agent_calls", [])]
    expected_agents = gt["expected_route"]["specialists"]
    routing = int(agents == expected_agents) if mode == "multi" else None
    names = {t.get("tool") for t in tool_calls}
    expected_tools = set(gt["expected_route"]["tools"])
    return {
        "accuracy": accuracy, "accuracy_eligible": eligible, "accuracy_method": method,
        "expected_behavior_correct": int(expected), "citation_checks": citation_checks,
        "required_source_coverage": coverage, "claims": claims,
        "structured_checks": deterministic, "specialist_routing": routing,
        "specialist_routing_reason": "" if mode == "multi" else "Single-agent architecture has no specialists.",
        "tool_route": int(expected_tools <= names) if expected_tools else int(not names),
        "manual_review_required": (eligible and accuracy is None) or any(c["score"] is None for c in citation_checks + claims),
    }


def ratio(values, eligible=None):
    values = list(values)
    scored = [v for v in values if v is not None]
    n = len(values) if eligible is None else eligible
    return {"value": sum(scored) / len(scored) if scored else None,
            "numerator": sum(scored), "denominator": len(scored),
            "eligible": n, "pending": n - len(scored),
            "conservative_lower_bound": sum(scored) / n if n else None}


def extract_values(result):
    return {
        "numeric_units": sorted(set(re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|hours?|days?|weeks?|credits?)",
            result.get("answer", ""), re.I))),
        "structured": [{"tool": s.get("tool"), "data": s.get("data", {})}
                       for s in leaves(result) if s.get("tool")],
    }


def stability(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["mode"], record["question_id"])].append(record)
    output = []
    for (model, mode, qid), runs in sorted(grouped.items()):
        if len(runs) < 2:
            output.append({"model": model, "mode": mode, "question_id": qid, "repetitions": 1,
                           "measured": False, "inconsistent": None})
            continue
        snapshots = {
            "answer": [r["result"].get("answer", "") for r in runs],
            "factual_values": [extract_values(r["result"]) for r in runs],
            "citations": [sorted((fingerprint(c) for c in r["result"].get("citations", []))) for r in runs],
            "specialists": [[c.get("agent") for c in r["result"].get("debug", {}).get("agent_calls", [])] for r in runs],
            "tools": [[{k: t.get(k) for k in ("tool", "lookup_status", "course", "semester")} for t in r["tool_calls"]] for r in runs],
            "correctness": [
                r["grading"]["accuracy"] if r["grading"]["accuracy_eligible"]
                else r["grading"]["expected_behavior_correct"] for r in runs],
            "status": [r["result"].get("status") for r in runs],
        }
        fields = {}
        for field, values in snapshots.items():
            if field == "correctness" and any(v is None for v in values):
                fields[field] = {"agreement": None, "all_equal": None, "values": values}
                continue
            keys = [fingerprint(v) for v in values]
            pairs = list(itertools.combinations(keys, 2))
            fields[field] = {"agreement": sum(a == b for a, b in pairs) / len(pairs),
                             "all_equal": len(set(keys)) == 1, "values": values}
        output.append({"model": model, "mode": mode, "question_id": qid,
                       "repetitions": len(runs), "measured": True,
                       "successful_repetitions": sum(r["result"].get("status") in {"answer", "partial_answer"} for r in runs),
                       "inconsistent": any(v["all_equal"] is False for v in fields.values()), "fields": fields})
    return output


def metrics(records):
    """Separate denominators; ungraded policy answers remain pending."""
    accuracy = [r["grading"]["accuracy"] for r in records if r["grading"]["accuracy_eligible"]]
    latencies = sorted(r["latency_seconds"] for r in records)
    citation_values = [c["score"] for r in records for c in r["grading"]["citation_checks"]]
    claim_values = [c["score"] for r in records for c in r["grading"]["claims"]]
    refusals = [r["grading"]["expected_behavior_correct"] for r in records if r["category"] == "out_of_scope"]
    abstentions = [r["grading"]["expected_behavior_correct"] for r in records if r["category"] == "unsupported"]
    missing = [r["grading"]["expected_behavior_correct"] for r in records
               if r["verification_status"] in {"pending_source", "expected_clarification"}]
    return {
        "records": len(records), "answer_accuracy": ratio(accuracy),
        "citation_correctness": ratio(citation_values), "claim_groundedness": ratio(claim_values),
        "out_of_scope_refusal": ratio(refusals), "unsupported_abstention": ratio(abstentions),
        "missing_data_behavior": ratio(missing),
        "specialist_routing": ratio(r["grading"]["specialist_routing"] for r in records if r["mode"] == "multi"),
        "tool_routing": ratio(r["grading"]["tool_route"] for r in records),
        "structured_field_correctness": ratio(
            c["score"] for r in records for c in r["grading"]["structured_checks"]),
        "required_source_coverage": {
            "matched": sum(r["grading"]["required_source_coverage"]["matched"] for r in records),
            "required": sum(r["grading"]["required_source_coverage"]["required"] for r in records)},
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else None,
            "p95_method": "nearest_rank", "includes_errors": True},
        "errors": sum(r["result"].get("status") == "error" for r in records),
        "manual_review_records": sum(r["grading"]["manual_review_required"] for r in records),
        "stability": stability(records),
        "interpretation": "No semantic similarity scoring. Verbatim fidelity is not answer completeness. Error/refusal repetition is not evidence of accurate answers.",
    }
