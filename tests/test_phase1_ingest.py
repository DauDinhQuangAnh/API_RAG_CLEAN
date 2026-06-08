from __future__ import annotations

import os
from datetime import date

import pandas as pd

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


class PipeChunker:
    def split_text(self, text: str) -> list[str]:
        return [chunk.strip() for chunk in text.split("||") if chunk.strip()]


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = None
        self.records: dict[str, dict[str, object]] = {}
        self.last_get_include = None

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

    def get(self, limit: int, offset: int, include):
        self.last_get_include = include
        records = list(self.records.items())[offset : offset + limit]
        payload = {"ids": [record_id for record_id, _ in records]}
        if "metadatas" in include:
            payload["metadatas"] = [record["metadata"] for _, record in records]
        if "documents" in include:
            payload["documents"] = [record["document"] for _, record in records]
        return payload


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata=None) -> FakeCollection:
        collection = self.collections.setdefault(name, FakeCollection(name))
        collection.metadata = metadata
        return collection

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


class StringableObject:
    def __str__(self) -> str:
        return "custom-object"


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
            "custom": StringableObject(),
            "id": "internal-record-id",
            "_id": "internal-chroma-id",
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
        "custom": "custom-object",
    }


def test_stable_record_id_is_deterministic_and_changes_with_inputs():
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
            "_id": "internal-id",
        }
    ]

    add_records_to_collection(records, FakeEmbeddingModel(), collection)
    add_records_to_collection(records, FakeEmbeddingModel(), collection)

    assert collection.count() == 1
    stored = collection.records["stable-id-1"]
    assert stored["document"] == "hello world"
    assert stored["metadata"]["chunk"] == "hello world"
    assert "id" not in stored["metadata"]
    assert "_id" not in stored["metadata"]
    assert "ignored" not in stored["metadata"]


def test_ingesting_same_txt_file_twice_does_not_duplicate_records(monkeypatch):
    from API_RAG_NEW import services

    fake_client = FakeChromaClient()
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "ProtonxSemanticChunker", lambda model: IdentityChunker())
    monkeypatch.setattr(services, "RAG_CHUNKING_PROFILE", "semantic")

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


def test_tabular_record_builder_uses_one_based_row_and_chunk_metadata():
    from API_RAG_NEW import services

    dataframe = pd.DataFrame(
        {
            "content": ["alpha || beta", "gamma"],
            "note": ["first row", "second row"],
            "id": ["row-id-should-not-win", "another-row-id"],
            "_id": ["row-internal-id", "another-row-internal-id"],
        }
    )
    file_hash = "b" * 64

    records = list(
        services._iter_tabular_chunk_records(
            dataframe,
            "content",
            "data.csv",
            ".csv",
            file_hash,
            PipeChunker(),
        )
    )

    assert [record["row_index"] for record in records] == [1, 1, 2]
    assert [record["row_chunk_index"] for record in records] == [1, 2, 1]
    assert [record["chunk_index"] for record in records] == [1, 2, 3]
    assert records[0]["doc_id"] == stable_record_id(
        "row", file_hash, 1, prefix="doc"
    )
    assert records[2]["doc_id"] == stable_record_id(
        "row", file_hash, 2, prefix="doc"
    )
    assert records[0]["id"] == stable_record_id(
        records[0]["doc_id"],
        "data.csv",
        "csv",
        1,
        1,
        "alpha",
    )
    assert records[0]["note"] == "first row"
    assert "_id" not in records[0]


def test_collection_records_response_includes_metadatas_and_documents(monkeypatch):
    from API_RAG_NEW import services

    fake_client = FakeChromaClient()
    collection = fake_client.get_or_create_collection("phase1-records")
    add_records_to_collection(
        [
            {
                "id": "stable-id-1",
                "doc_id": "doc_1",
                "source": "demo.txt",
                "source_type": "txt",
                "chunk_index": 1,
                "chunk": "visible chunk",
            }
        ],
        FakeEmbeddingModel(),
        collection,
    )
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)

    response = services.get_collection_records("phase1-records", limit=10, offset=0)

    assert collection.last_get_include == ["metadatas", "documents"]
    assert response.ids == ["stable-id-1"]
    assert response.metadatas[0]["chunk"] == "visible chunk"
    assert response.documents == ["visible chunk"]
