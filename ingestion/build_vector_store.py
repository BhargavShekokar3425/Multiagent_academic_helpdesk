"""Build a local Chroma snapshot with Ollama embeddings, or inspect its retrieval."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
import ollama

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embeddings(client, model, texts):
    result = client.embed(model=model, input=texts, truncate=False)
    vectors = result["embeddings"]
    if len(vectors) != len(texts) or not vectors:
        raise ValueError("Embedding count mismatch")
    if len({len(v) for v in vectors}) != 1 or not vectors[0]:
        raise ValueError("Invalid embedding dimensions")
    if any(not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise ValueError("Invalid embedding values")
    return vectors


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path,
                        default=ROOT / "data/processed/chunked_run_4/chunks.jsonl")
    parser.add_argument("--chroma-path", type=Path,
                        default=Path(os.getenv("CHROMA_PATH", "data/vector_store")))
    parser.add_argument("--model", default=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--include-review", action="store_true",
                        help="Explicitly include flagged text for development; flags remain")
    parser.add_argument("--query", help="Search the last successfully built collection; do not rebuild")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.model.startswith("nomic-embed-text"):
        parser.error("This builder uses Nomic task prefixes; use nomic-embed-text")
    path = args.chroma_path
    if not path.is_absolute():
        path = ROOT / path
    report_path = path / "index_report.json"
    ollama_client = ollama.Client(
        host=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"), timeout=180)
    if args.query:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["model"] != args.model:
            parser.error("Query embedding model must match the indexed model")
        current = ollama_client.show(args.model).model_dump(mode="json")
        identity = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        if identity != report["model_identity"]:
            parser.error("Model configuration changed; rebuild the index before querying")
        db = chromadb.PersistentClient(path=str(path))
        collection = db.get_collection(report["collection"], embedding_function=None)
        vectors = embeddings(ollama_client, args.model, [QUERY_PREFIX + args.query])
        results = collection.query(query_embeddings=vectors,
                                   n_results=min(5, collection.count()),
                                   include=["documents", "metadatas", "distances"])
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    chunks = [json.loads(line) for line in args.chunks.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        parser.error("Duplicate chunk IDs")
    selected = [c for c in chunks if args.include_review or not c["metadata"]["needs_review"]]
    if not selected:
        parser.error("No eligible chunks. Inspect chunking report and review flags.")
    for chunk in selected:
        if not chunk["text"].strip():
            parser.error("Empty chunk")
        chunk["metadata"] = {k: v for k, v in chunk["metadata"].items()
                             if isinstance(v, (str, int, float, bool))}
    # Pin the collection snapshot to both source content and model configuration.
    model_info = ollama_client.show(args.model).model_dump(mode="json")
    identity = hashlib.sha256(json.dumps(model_info, sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(
        {"chunks": selected, "model": args.model, "identity": identity,
         "prefix": DOCUMENT_PREFIX}, sort_keys=True).encode()).hexdigest()
    name = "academic_helpdesk_" + fingerprint[:20]
    # Probe the model before creating database files.
    first = selected[:args.batch_size]
    vectors = embeddings(ollama_client, args.model,
                         [DOCUMENT_PREFIX + c["text"] for c in first])
    db = chromadb.PersistentClient(path=str(path))
    collection = db.get_or_create_collection(
        name=name, embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model,
                  "snapshot": fingerprint, "model_identity": identity})
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset:offset + args.batch_size]
        if offset:
            vectors = embeddings(ollama_client, args.model,
                                 [DOCUMENT_PREFIX + c["text"] for c in batch])
        collection.upsert(ids=[c["chunk_id"] for c in batch],
                          documents=[c["text"] for c in batch],
                          metadatas=[c["metadata"] for c in batch], embeddings=vectors)
        print(f"Indexed {min(offset + len(batch), len(selected))}/{len(selected)}")
    if collection.count() != len(selected):
        raise ValueError("Stored chunk count does not match selected chunks")
    report = {
        "collection": name, "model": args.model, "model_identity": identity,
        "snapshot": fingerprint, "input_chunks": len(chunks),
        "indexed_chunks": collection.count(), "excluded_review_chunks": len(chunks)-len(selected),
        "includes_review_flagged_content": args.include_review,
        "embedding_dimension": len(vectors[0]), "chroma_path": str(path),
        "document_prefix": DOCUMENT_PREFIX, "query_prefix": QUERY_PREFIX,
        "notice": "Development index. Retrieval relevance is not policy verification.",
    }
    # Publish the current collection pointer only after a complete build.
    temp = report_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
