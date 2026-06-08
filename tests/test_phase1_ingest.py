from __future__ import annotations

import os
from datetime import date

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from API_RAG_NEW.rag_pipeline import (
    add_records_to_collection,
    sanitize_metadata,
    stable_record_id,
)


class IdentityChunker:
    def split_text(self, text: str) -> list[str]:
        return [text.strip()] if text.strip() else []


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = None
        self.records: dict[str, dict[str, object]] = {}

    def upsert(self, ids, embeddings, metadatas, documents):
        for record_id, embedding, metadata, document in zip(
            ids,
            embeddings,
            metadatas,
            documents,
        ):
            self.records[record_id] = {
                "embedding": embedding,
                "metadata": metadata,
                "document": document,
            }

    def count(self) -> int:
        return len(self.records)


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata=None) -> FakeCollection:
        collection = self.collections.setdefault(name, FakeCollection(name))
        collection.metadata = metadata
        return collection

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


def test_sanitize_metadata_removes_none_and_stringifies_unsupported_types():
    sanitized = sanitize_metadata(
        {
            "source": "demo.txt",
            "chunk_index": 1,
            "score": 0.75,
            "active": True,
            "empty": None,
            "published": date(2026, 6, 8),
            "tags": ["a", "b"],
            "payload": {"key": "value"},
            "id": "internal-record-id",
        }
    )

    assert sanitized == {
        "source": "demo.txt",
        "chunk_index": 1,
        "score": 0.75,
        "active": True,
        "published": "2026-06-08",
        "tags": "['a', 'b']",
        "payload": "{'key': 'value'}",
    }


def test_stable_record_id_is_deterministic_for_same_parts():
    first = stable_record_id("doc", "source.txt", 1, "same chunk")
    second = stable_record_id("doc", "source.txt", 1, "same chunk")
    different = stable_record_id("doc", "source.txt", 2, "same chunk")

    assert first == second
    assert first != different


def test_add_records_upserts_documents_and_keeps_chunk_metadata():
    collection = FakeCollection("phase1")
    records = [
        {
            "id": "stable-id-1",
            "doc_id": "doc_1",
            "source": "demo.txt",
            "source_type": "txt",
            "chunk_index": 1,
            "chunk": "hello world",
            "ignored": None,
        }
    ]

    add_records_to_collection(records, FakeEmbeddingModel(), collection)
    add_records_to_collection(records, FakeEmbeddingModel(), collection)

    assert collection.count() == 1
    stored = collection.records["stable-id-1"]
    assert stored["document"] == "hello world"
    assert stored["metadata"]["chunk"] == "hello world"
    assert "ignored" not in stored["metadata"]


def test_ingesting_same_txt_file_twice_does_not_duplicate_records(monkeypatch):
    from API_RAG_NEW import services

    fake_client = FakeChromaClient()
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "ProtonxSemanticChunker", lambda model: IdentityChunker())

    raw_content = b"repeatable ingest chunk"

    first = services.ingest_file_content("demo.txt", raw_content, "phase1-demo")
    second = services.ingest_file_content("demo.txt", raw_content, "phase1-demo")

    collection = fake_client.get_collection("phase1-demo")
    assert first.chunks == second.chunks == 1
    assert collection.count() == 1
    stored = next(iter(collection.records.values()))
    assert stored["document"] == "repeatable ingest chunk"
    assert stored["metadata"]["chunk"] == "repeatable ingest chunk"


def test_pdf_record_builder_includes_page_metadata():
    from API_RAG_NEW import services

    pages = [(1, "first page"), (2, "second page")]
    records = list(
        services._iter_pdf_chunk_records(
            pages,
            "demo.pdf",
            ".pdf",
            "a" * 64,
            IdentityChunker(),
        )
    )

    assert [record["page_number"] for record in records] == [1, 2]
    assert [record["page_chunk_index"] for record in records] == [1, 1]
    assert [record["chunk_index"] for record in records] == [1, 2]
    assert records[0]["doc_id"] == records[1]["doc_id"]
    assert records[0]["id"] != records[1]["id"]
