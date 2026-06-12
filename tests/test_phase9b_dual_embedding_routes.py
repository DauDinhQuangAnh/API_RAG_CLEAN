from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class FakeCollection:
    def __init__(self, name="demo", metadata=None, owner=None, count_value=0):
        self.name = name
        self.metadata = metadata
        self.upserts = []
        self.modify_calls = []
        self._owner = owner
        self.count_value = count_value

    def count(self):
        return self.count_value

    def modify(self, name=None, metadata=None):
        self.modify_calls.append({"name": name, "metadata": metadata})
        if name:
            if self._owner is not None:
                self._owner.collections.pop(self.name, None)
                self._owner.collections[name] = self
            self.name = name
        if metadata is not None:
            self.metadata = metadata

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self.count_value += len(kwargs.get("ids") or [])


class FakeChromaClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []

    def list_collections(self):
        return list(self.collections.values())

    def get_or_create_collection(self, *, name, metadata=None):
        collection = self.collections.get(name)
        if collection is None:
            collection = FakeCollection(name=name, metadata=metadata, owner=self)
            self.collections[name] = collection
        return collection

    def get_collection(self, name):
        if name in self.collections:
            return self.collections[name]
        for collection in self.collections.values():
            if collection.name == name:
                return collection
        raise KeyError(name)

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


def test_same_logical_name_is_provider_scoped_in_shared_db(monkeypatch):
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
    gemini_info = services.create_collection(
        CollectionCreateRequest(name="shared"),
        provider="gemini",
    )

    assert gemini_info.name == "shared"
    assert "shared" in shared_client.collections
    assert "gemini.shared" in shared_client.collections
    assert shared_client.get_collection("gemini.shared").metadata == {
        "logical_collection_name": "shared",
        "storage_collection_name": "gemini.shared",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-2",
        "embedding_dimension": 768,
    }


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
    shared_client.collections["gemini.docs"] = FakeCollection(
        name="gemini.docs",
        metadata={
            "logical_collection_name": "docs",
            "storage_collection_name": "gemini.docs",
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
        "collections": ["docs"]
    }


def test_query_uses_provider_scoped_storage_name(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import QueryRequest

    shared_client = FakeChromaClient()
    local_collection = shared_client.get_or_create_collection(
        name="bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "bv_yhct",
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
        },
    )
    gemini_collection = shared_client.get_or_create_collection(
        name="gemini.bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "gemini.bv_yhct",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    calls = []
    runtimes = {
        "local_sbert": FakeRuntime("local_sbert", chroma_client=shared_client),
        "gemini": FakeRuntime("gemini", chroma_client=shared_client),
    }
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtimes[provider])
    monkeypatch.setattr(services, "_build_optional_rerank_llm", lambda: None)
    monkeypatch.setattr(services, "_build_llm", lambda: type("FakeLLM", (), {"generate_content": lambda self, prompt: "ok"})())

    def fake_vector_search(model, query, collection, final_n, **kwargs):
        calls.append(collection.name)
        return [[]], ""

    monkeypatch.setattr(services, "vector_search", fake_vector_search)

    services.query_collection(
        "bv_yhct",
        QueryRequest(query="hello"),
        provider="local_sbert",
    )
    services.query_collection(
        "bv_yhct",
        QueryRequest(query="hello"),
        provider="gemini",
    )

    assert calls == [local_collection.name, gemini_collection.name]


def test_delete_gemini_does_not_delete_local(monkeypatch):
    from API_RAG_NEW import services

    shared_client = FakeChromaClient()
    shared_client.get_or_create_collection(
        name="bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "bv_yhct",
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
        },
    )
    shared_client.get_or_create_collection(
        name="gemini.bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "gemini.bv_yhct",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    runtimes = {
        "local_sbert": FakeRuntime("local_sbert", chroma_client=shared_client),
        "gemini": FakeRuntime("gemini", chroma_client=shared_client),
    }
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtimes[provider])

    services.delete_collection("bv_yhct", provider="gemini")

    assert "bv_yhct" in shared_client.collections
    assert "gemini.bv_yhct" not in shared_client.collections
    assert shared_client.deleted == ["gemini.bv_yhct"]


def test_rename_gemini_updates_storage_name_and_metadata(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionUpdateRequest

    shared_client = FakeChromaClient()
    collection = shared_client.get_or_create_collection(
        name="gemini.old_name",
        metadata={
            "logical_collection_name": "old_name",
            "storage_collection_name": "gemini.old_name",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
            "chunking_profile": "semantic",
        },
    )
    runtime = FakeRuntime("gemini", chroma_client=shared_client)
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    info = services.update_collection(
        "old_name",
        CollectionUpdateRequest(new_name="new_name"),
        provider="gemini",
    )

    assert info.name == "new_name"
    assert "gemini.old_name" not in shared_client.collections
    assert "gemini.new_name" in shared_client.collections
    assert collection.metadata["logical_collection_name"] == "new_name"
    assert collection.metadata["storage_collection_name"] == "gemini.new_name"
    assert collection.metadata["chunking_profile"] == "semantic"


def test_long_gemini_storage_name_uses_valid_hash_fallback():
    from API_RAG_NEW import services

    logical_name = "a" * 60
    storage_name = services.storage_collection_name("gemini", logical_name)

    assert storage_name.startswith("gemini.")
    assert storage_name != f"gemini.{logical_name}"
    assert len(storage_name) <= 63
    assert storage_name[0].isalnum()
    assert storage_name[-1].isalnum()
    assert all(char.isalnum() or char in {"_", "-", "."} for char in storage_name)


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


def test_gemini_collections_exclude_legacy_raw_name_gemini(monkeypatch):
    from API_RAG_NEW import services

    shared_client = FakeChromaClient()
    shared_client.collections["bv_yhct"] = FakeCollection(
        name="bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "bv_yhct",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    shared_client.collections["gemini.bv_yhct"] = FakeCollection(
        name="gemini.bv_yhct",
        metadata={
            "logical_collection_name": "bv_yhct",
            "storage_collection_name": "gemini.bv_yhct",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    runtime = FakeRuntime("gemini", chroma_client=shared_client)
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    assert services.list_collections(provider="gemini") == {
        "collections": ["bv_yhct"]
    }


def test_local_collections_exclude_gemini_storage_prefix(monkeypatch):
    from API_RAG_NEW import services

    shared_client = FakeChromaClient()
    shared_client.collections["gemini.bv_yhct"] = FakeCollection(
        name="gemini.bv_yhct",
        metadata=None,
    )
    shared_client.collections["local_docs"] = FakeCollection(
        name="local_docs",
        metadata=None,
    )
    runtime = FakeRuntime("local_sbert", chroma_client=shared_client)
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    assert services.list_collections(provider="local_sbert") == {
        "collections": ["local_docs"]
    }


@pytest.mark.parametrize("provider", ["local_sbert", "gemini"])
def test_create_collection_rejects_reserved_public_prefix(monkeypatch, provider):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionCreateRequest

    runtime = FakeRuntime(provider)
    monkeypatch.setattr(services, "get_embedding_runtime", lambda requested: runtime)

    with pytest.raises(HTTPException) as exc_info:
        services.create_collection(
            CollectionCreateRequest(name="gemini.bv_yhct"),
            provider=provider,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == services.RESERVED_COLLECTION_PREFIX_ERROR


@pytest.mark.parametrize("provider", ["local_sbert", "gemini"])
def test_rename_collection_rejects_reserved_public_prefix(monkeypatch, provider):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionUpdateRequest

    storage_name = "demo" if provider == "local_sbert" else "gemini.demo"
    runtime = FakeRuntime(provider)
    runtime.chroma_client.collections[storage_name] = FakeCollection(
        name=storage_name,
        metadata={
            "logical_collection_name": "demo",
            "storage_collection_name": storage_name,
            "embedding_provider": provider,
            "embedding_model": runtime.model_name,
            "embedding_dimension": 768,
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda requested: runtime)

    with pytest.raises(HTTPException) as exc_info:
        services.update_collection(
            "demo",
            CollectionUpdateRequest(new_name="gemini.bv_yhct"),
            provider=provider,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == services.RESERVED_COLLECTION_PREFIX_ERROR


@pytest.mark.parametrize(
    "path",
    ["/ingest", "/local/ingest", "/gemini/ingest"],
)
def test_ingest_routes_reject_reserved_public_prefix(monkeypatch, path):
    from API_RAG_NEW import services

    runtimes = {
        "local_sbert": FakeRuntime("local_sbert"),
        "gemini": FakeRuntime("gemini"),
    }
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtimes[provider])
    client = _client()

    response = client.post(
        path,
        files={"file": ("demo.txt", b"hello", "text/plain")},
        data={"collection_name": "gemini.bv_yhct"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == services.RESERVED_COLLECTION_PREFIX_ERROR


def test_create_gemini_logical_name_uses_gemini_storage(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import CollectionCreateRequest

    runtime = FakeRuntime("gemini")
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    info = services.create_collection(
        CollectionCreateRequest(name="bv_yhct"),
        provider="gemini",
    )

    assert info.name == "bv_yhct"
    assert "gemini.bv_yhct" in runtime.chroma_client.collections


def test_ingest_failure_rolls_back_new_empty_collection(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTTPException(400, "bad file")),
    )

    with pytest.raises(HTTPException):
        services.ingest_file_content(
            "demo.txt",
            b"hello",
            "newdocs",
            provider="local_sbert",
            chunking_profile="semantic",
        )

    assert "newdocs" not in runtime.chroma_client.collections
    assert runtime.chroma_client.deleted == ["newdocs"]


def test_ingest_failure_preserves_existing_collection(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["existing"] = FakeCollection(
        name="existing",
        metadata={
            "logical_collection_name": "existing",
            "storage_collection_name": "existing",
            "embedding_provider": "local_sbert",
            "embedding_model": runtime.model_name,
            "embedding_dimension": 768,
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(HTTPException(400, "bad file")),
    )

    with pytest.raises(HTTPException):
        services.ingest_file_content(
            "demo.txt",
            b"hello",
            "existing",
            provider="local_sbert",
            chunking_profile="semantic",
        )

    assert "existing" in runtime.chroma_client.collections
    assert runtime.chroma_client.deleted == []


def test_successful_ingest_keeps_new_collection(monkeypatch):
    from API_RAG_NEW import services

    runtime = FakeRuntime("local_sbert")
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "_build_ingest_records",
        lambda *args, **kwargs: (1, [{"id": "r1", "chunk": "hello"}], {}),
    )
    monkeypatch.setattr(
        services,
        "add_records_to_collection",
        lambda records, model, collection: collection.upsert(
            ids=[record["id"] for record in records]
        )
        or len(records),
    )

    response = services.ingest_file_content(
        "demo.txt",
        b"hello",
        "newdocs",
        provider="local_sbert",
        chunking_profile="semantic",
    )

    assert response.chunks == 1
    assert "newdocs" in runtime.chroma_client.collections
    assert runtime.chroma_client.deleted == []


def test_query_no_context_does_not_build_answer_llm(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import QueryRequest

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["demo"] = FakeCollection(
        name="demo",
        metadata={
            "logical_collection_name": "demo",
            "storage_collection_name": "demo",
            "embedding_provider": "local_sbert",
            "embedding_model": runtime.model_name,
            "embedding_dimension": 768,
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(services, "vector_search", lambda *args, **kwargs: ([[]], ""))
    monkeypatch.setattr(
        services,
        "_build_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )

    response = services.query_collection(
        "demo",
        QueryRequest(query="hello"),
        provider="local_sbert",
    )

    assert response.answer == services.NO_CONTEXT_ANSWER_MESSAGE
    assert response.citations == []


def test_query_usable_text_with_missing_metadata_calls_answer_llm(monkeypatch):
    from API_RAG_NEW import services
    from API_RAG_NEW.schemas import QueryRequest

    runtime = FakeRuntime("local_sbert")
    runtime.chroma_client.collections["demo"] = FakeCollection(
        name="demo",
        metadata={
            "logical_collection_name": "demo",
            "storage_collection_name": "demo",
            "embedding_provider": "local_sbert",
            "embedding_model": runtime.model_name,
            "embedding_dimension": 768,
        },
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)
    monkeypatch.setattr(
        services,
        "vector_search",
        lambda *args, **kwargs: ([[]], "chunk: usable text"),
    )

    class FakeLLM:
        def generate_content(self, prompt):
            return "answer"

    monkeypatch.setattr(services, "_build_llm", lambda *args, **kwargs: FakeLLM())

    response = services.query_collection(
        "demo",
        QueryRequest(query="hello"),
        provider="local_sbert",
    )

    assert response.answer == "answer"
    assert response.retrieved_data == "chunk: usable text"


def test_runtime_payloads_include_ingest_and_embedding_concurrency():
    from API_RAG_NEW import services

    config_payload = services.runtime_config_payload()
    status_payload = services.runtime_status_payload()

    for key in ("query", "llm", "ingest", "embedding"):
        assert key in config_payload["concurrency"]
        assert key in status_payload["concurrency"]
        for field in ("limit", "active", "available", "rejected"):
            assert field in config_payload["concurrency"][key]
            assert field in status_payload["concurrency"][key]

    assert "rag_max_concurrent_ingests" in config_payload
    assert "rag_max_concurrent_embedding_calls" in config_payload


def test_slot_limiter_queues_then_returns_configured_error_on_timeout():
    from API_RAG_NEW.concurrency import SlotLimiter

    limiter = SlotLimiter(1)
    with limiter.acquire(
        timeout=0,
        error_factory=lambda: HTTPException(status_code=503, detail="busy"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            with limiter.acquire(
                timeout=0,
                error_factory=lambda: HTTPException(status_code=503, detail="busy"),
            ):
                pass

    assert exc_info.value.status_code == 503
    assert limiter.snapshot()["rejected"] == 1


def test_internal_api_key_dev_mode_without_key_allows_access(monkeypatch):
    from API_RAG_NEW import security
    from API_RAG_NEW.main import app

    app.dependency_overrides = {}
    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", None)
    monkeypatch.setattr(security, "RAG_REQUIRE_INTERNAL_API_KEY", False)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    protected_response = client.get("/collections")
    assert protected_response.status_code in {200, 400}


def test_internal_api_key_require_mode_without_key_blocks(monkeypatch):
    from API_RAG_NEW import security
    from API_RAG_NEW.main import app

    app.dependency_overrides = {}
    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", None)
    monkeypatch.setattr(security, "RAG_REQUIRE_INTERNAL_API_KEY", True)
    client = TestClient(app)

    response = client.get("/collections")

    assert response.status_code == 500
    assert "RAG_INTERNAL_API_KEY must be configured" in response.json()["detail"]


def test_internal_api_key_require_mode_valid_and_wrong_key(monkeypatch):
    from API_RAG_NEW import security
    from API_RAG_NEW.main import app

    app.dependency_overrides = {}
    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", "secret")
    monkeypatch.setattr(security, "RAG_REQUIRE_INTERNAL_API_KEY", True)
    client = TestClient(app)

    wrong_response = client.get("/collections", headers={"X-Internal-API-Key": "bad"})
    valid_response = client.get(
        "/collections",
        headers={"X-Internal-API-Key": "secret"},
    )

    assert wrong_response.status_code == 401
    assert valid_response.status_code in {200, 400}
