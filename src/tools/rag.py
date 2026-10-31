"""Retrieve policy evidence from the current local Chroma snapshot.

This tool returns candidates, not a verified policy answer.
Run: python -m src.tools.rag --query "What are the PG leave rules?" --program PG
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def normalize_program(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def compatible_program(requested, recorded):
    """Conservative compatibility for the program labels in our manifest."""
    key = normalize_program(requested)
    label = normalize_program(recorded)
    if not key or not label or label == "allacademicprograms":
        return True
    ug = {"ug", "btech", "undergraduate"}
    pg = {"pg", "postgraduate"}
    aliases = {
        "mtech": {"mtech", "pg", "ugandpg", "ugandpgandphd"},
        "msc": {"msc", "pg", "ugandpg", "ugandpgandphd"},
        "phd": {"phd", "pg", "ugandpgandphd"},
        "msbyresearch": {"msbyresearch", "pg", "ugandpg", "ugandpgandphd"},
    }
    if key in ug:
        return label in ug or label in {"ugandpg", "ugandpgandphd"}
    if key in pg:
        return label in pg or label in {
            "mtech", "msc", "phd", "msbyresearch", "ugandpg", "ugandpgandphd"}
    return label == key or label in aliases.get(key, set())


def evidence_item(chunk_id, text, metadata, role, distance=None):
    reasons = metadata.get("review_reasons", "[]")
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except (ValueError, TypeError):
            reasons = [reasons]
    return {
        "chunk_id": chunk_id, "document_id": metadata["document_id"],
        "title": metadata.get("title", ""), "text": text,
        "page_start": metadata["page_start"], "page_end": metadata["page_end"],
        "source_url": metadata.get("source_url", ""),
        "filename": metadata.get("filename", ""),
        "citation": f'{metadata["document_id"]}, PDF p. {metadata["page_start"]}',
        "program": metadata.get("program", ""),
        "issued_on": metadata.get("issued_on", ""),
        "effective_from": metadata.get("effective_from", ""),
        "policy_status": metadata.get("status", "unknown"),
        "verification_status": metadata.get("verification_status", "unknown"),
        "needs_review": bool(metadata.get("needs_review", True)),
        "review_reasons": reasons,
        "retrieval_role": role, "cosine_distance": distance,
    }


def search_policy(query: str, academic_program: str = "",
                  admission_year: int = 0, top_k: int = 5,
                  include_review: bool = True) -> dict:
    """Search academic policies and return cited evidence.

    academic_program is optional, e.g. B.Tech., PG, M.Tech. or Ph.D.
    admission_year is context only: the current manifest cannot enforce cohort
    eligibility. top_k is the number of ranked hits, before context expansion.
    include_review permits flagged development content, with warnings retained.
    No similarity score is treated as proof that an answer is supported.
    """
    response = {
        "status": "error",
        "data": {"query": query, "academic_program": academic_program,
                 "admission_year": admission_year},
        "evidence": [], "warnings": [], "message": "",
    }
    if not isinstance(query, str) or not query.strip():
        response["message"] = "Provide a nonempty query."
        return response
    if type(top_k) is not int or not 1 <= top_k <= 10:
        response["message"] = "top_k must be an integer from 1 to 10."
        return response
    if type(admission_year) is not int or admission_year < 0:
        response["message"] = "admission_year must be a nonnegative integer."
        return response
    try:
        from dotenv import load_dotenv
        import chromadb
        import ollama
        from ingestion.build_vector_store import embeddings

        load_dotenv(ROOT / ".env")
        path = Path(os.getenv("CHROMA_PATH", "data/vector_store"))
        if not path.is_absolute():
            path = ROOT / path
        report = json.loads((path / "index_report.json").read_text(encoding="utf-8"))
        model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        if model != report["model"]:
            raise ValueError("Configured embedding model differs from indexed model.")
        client = ollama.Client(
            host=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"), timeout=180)
        info = client.show(model).model_dump(mode="json")
        identity = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
        if identity != report["model_identity"]:
            raise ValueError("Embedding model configuration changed; rebuild the index.")
        db = chromadb.PersistentClient(path=str(path))
        collection = db.get_collection(report["collection"], embedding_function=None)
        if collection.metadata.get("snapshot") != report["snapshot"]:
            raise ValueError("Collection snapshot does not match the index report.")

        # Small assignment corpus: inspect metadata to construct exact compatible
        # program filters before vector search, rather than filtering top hits later.
        catalog = collection.get(include=["metadatas"])
        permitted = []
        for metadata in catalog["metadatas"]:
            if (compatible_program(academic_program, metadata.get("program", ""))
                    and (include_review or not metadata.get("needs_review", True))):
                permitted.append(metadata)
        if not permitted:
            response.update(status="insufficient_evidence",
                            message="No indexed chunks match the requested filters.")
            return response
        labels = sorted({m.get("program", "") for m in permitted})
        clauses = [{"program": {"$in": labels}}]
        if not include_review:
            clauses.append({"needs_review": False})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        vectors = embeddings(client, model, [report["query_prefix"] + query.strip()])
        result = collection.query(
            query_embeddings=vectors, where=where,
            n_results=min(top_k, len(permitted)),
            include=["documents", "metadatas", "distances"],
        )
        seen = set()
        for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0]):
            if text and metadata:
                response["evidence"].append(
                    evidence_item(chunk_id, text, metadata, "semantic_hit", distance))
                seen.add(chunk_id)

        # Expand only the two strongest hits. Short circulars get all indexed
        # pages; longer documents get the same and adjacent pages.
        seeds = list(response["evidence"][:2])
        context = {}
        for seed in seeds:
            doc_id = seed["document_id"]
            doc_catalog = [m for m in permitted if m["document_id"] == doc_id]
            page_numbers = {m["page_start"] for m in doc_catalog}
            short_document = len(page_numbers) <= 4
            lower, upper = max(1, seed["page_start"] - 1), seed["page_end"] + 1
            clauses = [{"document_id": doc_id}]
            if not short_document:
                clauses += [{"page_start": {"$gte": lower}},
                            {"page_start": {"$lte": upper}}]
            if not include_review:
                clauses.append({"needs_review": False})
            context_where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
            nearby = collection.get(where=context_where, include=["documents", "metadatas"])
            for chunk_id, text, metadata in zip(
                    nearby["ids"], nearby["documents"], nearby["metadatas"]):
                if chunk_id not in seen and text:
                    context[chunk_id] = (text, metadata)
        char_count = sum(len(e["text"]) for e in response["evidence"])
        for chunk_id, (text, metadata) in sorted(
                context.items(), key=lambda entry: (
                    entry[1][1]["document_id"], entry[1][1]["page_start"],
                    entry[1][1].get("chunk_index", 0))):
            if len(response["evidence"]) >= top_k + 12 or char_count + len(text) > 24000:
                response["warnings"].append("Context expansion was limited to bound prompt size.")
                break
            response["evidence"].append(
                evidence_item(chunk_id, text, metadata, "neighbor_context"))
            char_count += len(text)
        warnings = response["warnings"]
        warnings.append(
            "Candidates require claim-level checking. Distance is not confidence; "
            "compare amendments with older rules before answering.")
        if not academic_program:
            warnings.append("No program filter supplied; applicability may differ between results.")
        if admission_year:
            warnings.append(
                "Cohort applicability is not automatically verified; inspect source conditions.")
        if any(e["needs_review"] for e in response["evidence"]):
            warnings.append("Evidence includes OCR or table content awaiting review.")
        if any(e["policy_status"] == "unknown" for e in response["evidence"]):
            warnings.append("Some source policy statuses are unknown; current validity is unverified.")
        if not response["evidence"]:
            response.update(status="insufficient_evidence", message="No matching evidence found.")
        else:
            response.update(status="success",
                            message="Retrieved candidate passages; answer sufficiency is not established.")
        response["data"].update(collection=report["collection"],
                                ranked_hits=sum(e["retrieval_role"] == "semantic_hit"
                                                for e in response["evidence"]),
                                evidence_count=len(response["evidence"]))
        return response
    except Exception as exc:
        response["evidence"] = []
        response["message"] = f"Retrieval failed: {type(exc).__name__}: {exc}"
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--program", default="")
    parser.add_argument("--admission-year", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--exclude-review", action="store_true")
    args = parser.parse_args()
    result = search_policy(args.query, args.program, args.admission_year,
                           args.top_k, not args.exclude_review)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
