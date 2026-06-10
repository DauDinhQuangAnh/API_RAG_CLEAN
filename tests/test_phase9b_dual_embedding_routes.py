from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class FakeCollection:
    def __init__(self, name="demo", metadata=None):
        self.name = name
        self.metadata = metadata
        self.upserts = []
        self.modify_calls = []

    def count(self):
        return 0

    def modify(self, name=None, metadata=None):
        self.modify_calls.append({"name": name, "metadata": metadata})
        if name:
            self.name = name
        if metadata is not None:
            self.metadata = metadata

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class FakeChromaClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []

    def list_collections(self):
        return list(self.collections.values())

    def get_or_create_collection(self, *, name, metadata=None):
        collection = self.collections.get(name)
        if collection is None:
            collection = FakeCollection(name=name, metadata=metadata)
            self.collections[name] = collection
        return collection

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)


class FakeRuntime:
    def __init__(self, provider, chroma_client=None):
        self.provider = provider
        self.model_name = (
            "gemini-embedding-2"
            if provider == "gemini"
            else "keepitreal/vietnamese-sbert"
        )
        self.dimension = 768
        self.embedding_model = object()
        self.chroma_client = chroma_client or FakeChromaClient()
        self.chroma_db_path = "db"


def _client():
    from API_RAG_NEW.main import app
    from API_RAG_NEW.security import require_internal_api_key

    app.dependency_overrides[require_internal_api_key] = lambda: None
    return TestClient(app)


def test_root_and_local_ingest_use_local_provider(monkeypatch):
    from API_RAG_NEW import main
    from API_RAG_NEW.schemas import IngestResponse

    calls = []

    def fake_ingest(
        file_name,
        raw_content,
        collection_name,
        provider="local_sbert",
        chunking_profile=None,
    ):
        calls.append((provider, chunking_profile))
        return IngestResponse(collection_name="demo", rows=1, chunks=1)

    monkeypatch.setattr(main.services, "ingest_file_content", fake_ingest)
    client = _client()

    for path in ("/ingest", "/local/ingest"):
        response = client.post(
            path,
            files={"file": ("demo.txt", b"hello", "text/plain")},
            data={"collection_name": "demo"},
        )
        assert response.status_code == 200

    assert calls == [("local_sbert", "hybrid"), ("local_sbert", "hybrid")]


def test_root_and_local_query_use_local_provider(monkeypatch):
    from API_RAG_NEW import main
    from API_RAG_NEW.schemas import QueryResponse

    calls = []

    def fake_query(collection_name, req, provider="local_sbert"):
        calls.append(provider)
        return QueryResponse(
            metadatas=[],
            retrieved_data="",
            answer="ok",
            full_prompt="prompt",
        )

    monkeypatch.setattr(main.services, "query_collection", fake_query)
    client = _client()

    for path in (
        "/collections/demo/query",
        "/local/collections/demo/query",
    ):
        response = client.post(path, json={"query": "hello"})
        assert response.status_code == 200

    assert calls == ["local_sbert", "local_sbert"]


def test_gemini_ingest_and_query_use_gemini_provider(monkeypatch):
    from API_RAG_NEW import main
    from API_RAG_NEW.schemas import IngestResponse, QueryResponse

    ingest_calls = []
    query_calls = []

    def fake_ingest(
        file_name,
        raw_content,
        collection_name,
        provider="local_sbert",
        chunking_profile=None,
    ):
        ingest_calls.append((provider, chunking_profile))
        return IngestResponse(collection_name="demo", rows=1, chunks=1)

    def fake_query(collection_name, req, provider="local_sbert"):
        query_calls.append(provider)
        return QueryResponse(
            metadatas=[],
            retrieved_data="",
            answer="ok",
            full_prompt="prompt",
        )

    monkeypatch.setattr(main.services, "ingest_file_content", fake_ingest)
    monkeypatch.setattr(main.services, "query_collection", fake_query)
    client = _client()

    ingest_response = client.post(
        "/gemini/ingest",
        files={"file": ("demo.txt", b"hello", "text/plain")},
    )
    query_response = client.post("/gemini/collections/demo/query", json={"query": "hi"})

    assert ingest_response.status_code == 200
    assert query_response.status_code == 200
    assert ingest_calls == [("gemini", "hybrid")]
    assert query_calls == ["gemini"]


def test_local_and_gemini_ingest_forward_chunking_profile(monkeypatch):
    from API_RAG_NEW import main
    from API_RAG_NEW.schemas import IngestResponse

    calls = []

    def fake_ingest(
        file_name,
        raw_content,
        collection_name,
        provider="local_sbert",
        chunking_profile=None,
    ):
        calls.append((provider, chunking_profile))
        return IngestResponse(collection_name="demo", rows=1, chunks=1)

    monkeypatch.setattr(main.services, "ingest_file_content", fake_ingest)
    client = _client()

    local_response = client.post(
        "/local/ingest",
        files={"file": ("demo.txt", b"hello", "text/plain")},
        data={"collection_name": "demo", "chunking_profile": "semantic"},
    )
    gemini_response = client.post(
        "/gemini/ingest",
        files={"file": ("demo.txt", b"hello", "text/plain")},
        data={"collection_name": "demo", "chunking_profile": "hybrid"},
    )

    assert local_response.status_code == 200
    assert gemini_response.status_code == 200
    assert calls == [("local_sbert", "semantic"), ("gemini", "hybrid")]


def test_invalid_chunking_profile_returns_http_400(monkeypatch):
    from API_RAG_NEW import main

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ingest service should not be called")

    monkeypatch.setattr(main.services, "ingest_file_content", fail_if_called)
    client = _client()

    response = client.post(
        "/local/ingest",
        files={"file": ("demo.txt", b"hello", "text/plain")},
        data={"collection_name": "demo", "chunking_profile": "bad"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid chunking_profile. Allowed values: hybrid, semantic."
    )


def test_gemini_routes_fail_without_key_but_local_routes_work(monkeypatch):
    from API_RAG_NEW import services

    local_runtime = FakeRuntime("local_sbert")

    def fake_runtime(provider):
        if provider == "gemini":
            raise RuntimeError("GEMINI_API_KEY must be configured for Gemini embedding routes.")
        return local_runtime

    monkeypatch.setattr(services, "get_embedding_runtime", fake_runtime)
    client = _client()

    local_response = client.get("/collections")
    gemini_response = client.get("/gemini/collections")

    assert local_response.status_code == 200
    assert gemini_response.status_code == 400
    assert "GEMINI_API_KEY" in gemini_response.json()["detail"]


def test_same_collection_name_is_unique_in_shared_db(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionCreateRequest

    shared_client = FakeChromaClient()
    runtimes = {
        "local_sbert": FakeRuntime("local_sbert", chroma_client=shared_client),
        "gemini": FakeRuntime("gemini", chroma_client=shared_client),
    }
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtimes[provider])

    local_info = services.create_collection(
        CollectionCreateRequest(name="shared"),
        provider="local_sbert",
    )

    assert local_info.name == "shared"
    assert (
        shared_client.get_collection("shared").metadata[
            "embedding_provider"
        ]
        == "local_sbert"
    )
    try:
        services.create_collection(
            CollectionCreateRequest(name="shared"),
            provider="gemini",
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Collection already exists."
    else:
        raise AssertionError("Expected duplicate collection HTTPException")


def test_shared_db_collection_list_filters_by_provider(monkeypatch):
    from API_RAG_NEW import services

    shared_client = FakeChromaClient()
    shared_client.collections["local_docs"] = FakeCollection(
        name="local_docs",
        metadata={
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
        },
    )
    shared_client.collections["gemini_docs"] = FakeCollection(
        name="gemini_docs",
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    shared_client.collections["legacy_docs"] = FakeCollection(
        name="legacy_docs",
        metadata=None,
    )
    runtimes = {
        "local_sbert": FakeRuntime("local_sbert", chroma_client=shared_client),
        "gemini": FakeRuntime("gemini", chroma_client=shared_client),
    }
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtimes[provider])

    assert services.list_collections(provider="local_sbert") == {
        "collections": ["local_docs", "legacy_docs"]
    }
    assert services.list_collections(provider="gemini") == {
        "collections": ["gemini_docs"]
    }


def test_collection_metadata_mismatch_returns_http_400():
    from API_RAG_NEW import services

    local_runtime = FakeRuntime("local_sbert")
    collection = FakeCollection(
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        }
    )

    try:
        services._validate_collection_embedding_metadata(local_runtime, collection)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "different embedding provider/model" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_create_collection_metadata_does_not_include_chunking_profile(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionCreateRequest

    runtime = FakeRuntime("local_sbert")
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    info = services.create_collection(
        CollectionCreateRequest(name="chunked"),
        provider="local_sbert",
    )

    assert "chunking_profile" not in info.metadata


def test_first_ingest_into_empty_collection_sets_chunking_profile(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["empty"] = FakeCollection(
        name="empty",
        metadata={
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (1, [{"id": "r1", "chunk": "hello"}], {}),
    )
    monkeypatch.setattr(
        services,
        "add_records_to_collection",
        lambda records, model, collection: len(records),
    )

    response = services.ingest_file_content(
        "demo.txt",
        b"hello",
        "empty",
        provider="local_sbert",
        chunking_profile="semantic",
    )

    assert response.chunking_profile == "semantic"
    assert runtime.chroma_client.collections["empty"].metadata["chunking_profile"] == (
        "semantic"
    )


def test_appending_with_different_chunking_profile_returns_http_400(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["demo"] = FakeCollection(
        name="demo",
        metadata={
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
            "chunking_profile": "hybrid",
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (1, [{"id": "r1", "chunk": "hello"}], {}),
    )

    try:
        services.ingest_file_content(
            "demo.txt",
            b"hello",
            "demo",
            provider="local_sbert",
            chunking_profile="semantic",
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "different chunking_profile" in exc.detail
    else:
        raise AssertionError("Expected HTTPException")


def test_hybrid_fallback_sets_collection_metadata_to_semantic(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    def fake_build_records(extension, raw_content, file_name, file_hash, chunker, profile):
        if profile == "hybrid":
            raise RuntimeError("hybrid failed")
        return 1, [{"id": "r1", "chunk": "hello"}], {}

    monkeypatch.setattr(services, "_build_ingest_records", fake_build_records)
    monkeypatch.setattr(
        services,
        "add_records_to_collection",
        lambda records, model, collection: len(records),
    )

    response = services.ingest_file_content(
        "demo.txt",
        b"hello",
        "fallback",
        provider="local_sbert",
        chunking_profile="hybrid",
    )

    collection = runtime.chroma_client.collections["fallback"]
    assert response.chunking_profile == "semantic"
    assert collection.metadata["chunking_profile"] == "semantic"


def test_legacy_local_collection_allows_ingest_without_chunking_metadata(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["legacy"] = FakeCollection(
        name="legacy",
        metadata=None,
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (1, [{"id": "r1", "chunk": "hello"}], {}),
    )
    monkeypatch.setattr(
        services,
        "add_records_to_collection",
        lambda records, model, collection: len(records),
    )

    response = services.ingest_file_content(
        "demo.txt",
        b"hello",
        "legacy",
        provider="local_sbert",
        chunking_profile="semantic",
    )

    assert response.chunking_profile == "semantic"
    assert runtime.chroma_client.collections["legacy"].metadata["chunking_profile"] == (
        "semantic"
    )


def test_query_endpoints_do_not_forward_chunking_profile(monkeypatch):
    from API_RAG_NEW import main
    from API_RAG_NEW.schemas import QueryResponse

    calls = []

    def fake_query(collection_name, req, provider="local_sbert"):
        calls.append((provider, req.query))
        return QueryResponse(
            metadatas=[],
            retrieved_data="",
            answer="ok",
            full_prompt="prompt",
        )

    monkeypatch.setattr(main.services, "query_collection", fake_query)
    client = _client()

    response = client.post(
        "/local/collections/demo/query",
        json={"query": "hello", "chunking_profile": "semantic"},
    )

    assert response.status_code == 200
    assert calls == [("local_sbert", "hello")]


def test_runtime_payloads_expose_dual_provider_info_safely(monkeypatch):
    from API_RAG_NEW import services

    monkeypatch.setattr(services, "get_gemini_api_key", lambda: None)

    config_payload = services.runtime_config_payload()
    status_payload = services.runtime_status_payload()
    combined = f"{config_payload} {status_payload}"

    assert config_payload["embedding_routes"] == {
        "root": "local_sbert",
        "local": "local_sbert",
        "gemini": "gemini",
    }
    assert config_payload["local_embedding"]["provider"] == "local_sbert"
    assert config_payload["gemini_embedding"]["provider"] == "gemini"
    assert config_payload["gemini_embedding"]["configured"] is False
    assert config_payload["available_chunking_profiles"] == ["hybrid", "semantic"]
    assert config_payload["default_chunking_profile"] in {"hybrid", "semantic"}
    assert status_payload["gemini_embedding"]["configured"] is False
    assert status_payload["available_chunking_profiles"] == ["hybrid", "semantic"]
    assert "GEMINI_API_KEY" not in combined
    assert "RAG_INTERNAL_API_KEY" not in combined
