"""UTF-8 regression checks using temporary files and fake embedding/storage clients."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ingestion import build_vector_store as builder
from src.tools.rag import search_policy

TEXT = "परीक्षा – Major examination duration: 3 hours."
MODEL_INFO = {"family": "offline-test"}
IDENTITY = hashlib.sha256(json.dumps(MODEL_INFO, sort_keys=True).encode()).hexdigest()


class FakeOllama:
    def __init__(self):
        self.inputs = []

    def show(self, model):
        return SimpleNamespace(model_dump=lambda **kwargs: MODEL_INFO)

    def embed(self, model, input, truncate):
        assert truncate is False
        self.inputs.extend(input)
        return {"embeddings": [[1.0, 0.25] for _ in input]}


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.metadata = {"snapshot": "offline-snapshot"}
        self.source_metadata = {
            "document_id": "test_policy", "program": "PG", "page_start": 1,
            "page_end": 1, "needs_review": False, "review_reasons": "[]",
            "status": "unknown", "title": "परीक्षा नियम",
        }

    def count(self):
        return len(self.documents)

    def upsert(self, **kwargs):
        self.documents.extend(kwargs["documents"])

    def query(self, **kwargs):
        return {"ids": [["test_chunk"]], "documents": [[TEXT]],
                "metadatas": [[self.source_metadata]], "distances": [[0.2]]}

    def get(self, **kwargs):
        return {"ids": ["test_chunk"], "documents": [TEXT],
                "metadatas": [self.source_metadata]}


class FakeDB:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        self.collection.metadata = kwargs["metadata"]
        return self.collection

    def get_collection(self, name, **kwargs):
        return self.collection


def use_clients(monkeypatch, collection):
    client = FakeOllama()
    monkeypatch.setattr(builder.ollama, "Client", lambda **kwargs: client)
    monkeypatch.setattr(builder.chromadb, "PersistentClient", lambda **kwargs: FakeDB(collection))
    return client


def emulate_non_utf8_default(monkeypatch, target):
    read = Path.read_text

    def checked_read(path, encoding=None, errors=None):
        if path == target:
            assert encoding == "utf-8", "Persistent Unicode input needs explicit UTF-8"
        # Simulate a Windows ANSI default if the caller omits encoding.
        return read(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", checked_read)


def test_index_builder_preserves_multilingual_chunks_and_report_encoding(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps({
        "chunk_id": "test_chunk", "text": TEXT,
        "metadata": {"document_id": "test_policy", "needs_review": False},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    store = tmp_path / "संग्रह"
    store.mkdir()
    collection = FakeCollection()
    client = use_clients(monkeypatch, collection)
    emulate_non_utf8_default(monkeypatch, chunks)
    write = Path.write_text

    def checked_write(path, data, encoding=None, errors=None, **kwargs):
        if path == store / "index_report.tmp":
            assert encoding == "utf-8"
        return write(path, data, encoding=encoding or "cp1252", errors=errors, **kwargs)

    monkeypatch.setattr(Path, "write_text", checked_write)
    monkeypatch.setattr(sys, "argv", [
        "build_vector_store", "--chunks", str(chunks),
        "--chroma-path", str(store), "--model", "nomic-embed-text"])
    builder.main()
    assert collection.documents == [TEXT]
    assert client.inputs == [builder.DOCUMENT_PREFIX + TEXT]
    report = json.loads((store / "index_report.json").read_text(encoding="utf-8"))
    assert report["indexed_chunks"] == 1
    assert report["chroma_path"] == str(store)


def saved_report(tmp_path):
    report = tmp_path / "index_report.json"
    report.write_text(json.dumps({
        "model": "nomic-embed-text", "model_identity": IDENTITY,
        "collection": "academic_helpdesk_test", "snapshot": "offline-snapshot",
        "query_prefix": builder.QUERY_PREFIX, "note": "परीक्षा",
    }, ensure_ascii=False), encoding="utf-8")
    return report


def test_builder_query_reads_utf8_index_report(tmp_path, monkeypatch, capsys):
    report = saved_report(tmp_path)
    collection = FakeCollection()
    collection.documents = [TEXT]
    client = use_clients(monkeypatch, collection)
    emulate_non_utf8_default(monkeypatch, report)
    monkeypatch.setattr(sys, "argv", [
        "build_vector_store", "--chroma-path", str(tmp_path),
        "--model", "nomic-embed-text", "--query", "परीक्षा"])
    builder.main()
    assert TEXT in capsys.readouterr().out
    assert client.inputs == [builder.QUERY_PREFIX + "परीक्षा"]


def test_rag_reads_utf8_index_report_without_touching_real_chroma(tmp_path, monkeypatch):
    report = saved_report(tmp_path)
    collection = FakeCollection()
    collection.documents = [TEXT]
    use_clients(monkeypatch, collection)
    emulate_non_utf8_default(monkeypatch, report)
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    result = search_policy("परीक्षा", academic_program="PG")
    assert result["status"] == "success", result["message"]
    assert result["evidence"][0]["text"] == TEXT
