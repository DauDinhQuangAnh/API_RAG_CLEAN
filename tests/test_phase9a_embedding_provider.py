from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class FakeLocalModel:
    def __init__(self, dimension=3):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts):
        return [[index + 0.1 for index in range(self.dimension)] for _ in texts]


class FakeLocalModelWithoutDimension:
    def encode(self, texts):
        return [[1, 2, 3, 4] for _ in texts]


def test_local_wrapper_detects_dimension_and_returns_float_vectors():
    from API_RAG_NEW.embeddings import LocalSentenceTransformerEmbeddings

    provider = LocalSentenceTransformerEmbeddings(FakeLocalModel(), "fake-local")

    assert provider.provider == "local_sbert"
    assert provider.model_name == "fake-local"
    assert provider.dimension == 3
    assert provider.encode(["abc"]) == [[0.1, 1.1, 2.1]]


def test_local_wrapper_falls_back_to_probe_for_dimension():
    from API_RAG_NEW.embeddings import LocalSentenceTransformerEmbeddings

    provider = LocalSentenceTransformerEmbeddings(
        FakeLocalModelWithoutDimension(),
        "fake-local",
    )

    assert provider.dimension == 4


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeEmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        if isinstance(contents, list):
            return FakeEmbedResponse([FakeEmbedding([999.0, 999.0])])
        return FakeEmbedResponse(
            [FakeEmbedding([float(len(self.calls)), float(len(str(contents)))])]
        )


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()


def test_gemini_provider_formats_documents_and_queries():
    from API_RAG_NEW.embeddings import GeminiTextEmbeddings

    client = FakeGeminiClient()
    provider = GeminiTextEmbeddings(
        api_key="fake",
        model_name="gemini-embedding-2",
        output_dimensionality=768,
        task="question answering",
        batch_size=32,
        client=client,
    )

    assert provider.encode_documents(["abc"], titles=["doc"]) == [[1.0, 22.0]]
    assert client.models.calls[-1]["contents"] == "title: doc | text: abc"

    assert provider.encode_queries(["abc"]) == [[2.0, 37.0]]
    assert client.models.calls[-1]["contents"] == (
        "task: question answering | query: abc"
    )


def test_gemini_provider_calls_embedding_2_once_per_document_text():
    from API_RAG_NEW.embeddings import GeminiTextEmbeddings

    client = FakeGeminiClient()
    provider = GeminiTextEmbeddings(
        api_key="fake",
        model_name="gemini-embedding-2",
        output_dimensionality=768,
        task="question answering",
        batch_size=2,
        client=client,
    )

    vectors = provider.encode_documents(["a", "b"], titles=["one", "two"])

    assert vectors == [[1.0, 20.0], [2.0, 20.0]]
    assert [call["contents"] for call in client.models.calls] == [
        "title: one | text: a",
        "title: two | text: b",
    ]
    assert all(isinstance(call["contents"], str) for call in client.models.calls)


def test_gemini_provider_calls_embedding_2_once_per_query_text():
    from API_RAG_NEW.embeddings import GeminiTextEmbeddings

    client = FakeGeminiClient()
    provider = GeminiTextEmbeddings(
        api_key="fake",
        model_name="gemini-embedding-2",
        output_dimensionality=768,
        task="question answering",
        batch_size=2,
        client=client,
    )

    vectors = provider.encode_queries(["q1", "q2"])

    assert vectors == [[1.0, 36.0], [2.0, 36.0]]
    assert [call["contents"] for call in client.models.calls] == [
        "task: question answering | query: q1",
        "task: question answering | query: q2",
    ]
    assert all(isinstance(call["contents"], str) for call in client.models.calls)


class FailingGeminiModels:
    def embed_content(self, *, model, contents, config):
        raise ValueError("sdk failure with internal details")


class FailingGeminiClient:
    def __init__(self):
        self.models = FailingGeminiModels()


def test_gemini_provider_wraps_sdk_errors_without_secret():
    from API_RAG_NEW.embeddings import GeminiTextEmbeddings

    provider = GeminiTextEmbeddings(
        api_key="secret-key",
        model_name="gemini-embedding-2",
        output_dimensionality=768,
        task="question answering",
        batch_size=32,
        client=FailingGeminiClient(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        provider.encode_documents(["abc"], titles=["doc"])

    assert str(exc_info.value) == "Gemini embedding request failed."
    assert "secret-key" not in str(exc_info.value)


class FakeDocumentEmbeddingModel:
    def __init__(self):
        self.document_calls = []

    def encode_documents(self, texts, titles=None):
        self.document_calls.append((list(texts), list(titles or [])))
        return [[1.0, 0.0] for _ in texts]

    def encode(self, texts):
        raise AssertionError("raw encode should not be used for documents")


class FakeCollection:
    def __init__(self, metadata=None, name="fake_collection"):
        self.name = name
        self.metadata = metadata
        self.upserts = []
        self.queries = []
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

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {
            "ids": [["id1"]],
            "metadatas": [[{"chunk": "abc", "doc_id": "doc1"}]],
            "documents": [["abc"]],
            "distances": [[0.1]],
        }


class FakeRuntime:
    def __init__(
        self,
        provider="local_sbert",
        model_name="fake-local",
        dimension=384,
        chroma_client=None,
        embedding_model=None,
        chroma_db_path="fake_db",
    ):
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
        self.chroma_db_path = chroma_db_path


def test_add_records_to_collection_uses_document_embedding_path():
    from API_RAG_NEW.rag_pipeline import add_records_to_collection

    model = FakeDocumentEmbeddingModel()
    collection = FakeCollection()

    count = add_records_to_collection(
        [
            {
                "id": "id1",
                "chunk": "abc",
                "section_title": "Section A",
                "source": "source.txt",
            }
        ],
        model,
        collection,
    )

    assert count == 1
    assert model.document_calls == [(["abc"], ["Section A"])]
    assert collection.upserts[0]["embeddings"] == [[1.0, 0.0]]


class FakeQueryEmbeddingModel:
    def __init__(self):
        self.query_calls = []

    def encode_queries(self, texts):
        self.query_calls.append(list(texts))
        return [[0.0, 1.0] for _ in texts]

    def encode(self, texts):
        raise AssertionError("raw encode should not be used for queries")


def test_vector_search_uses_query_embedding_path():
    from API_RAG_NEW.rag_pipeline import vector_search

    model = FakeQueryEmbeddingModel()
    collection = FakeCollection()

    metadatas, retrieved_data = vector_search(
        model,
        "question",
        collection,
        1,
        reranker_type="none",
    )

    assert model.query_calls == [["question"]]
    assert collection.queries[0]["query_embeddings"] == [[0.0, 1.0]]
    assert metadatas[0][0]["chunk"] == "abc"
    assert "chunk: abc" in retrieved_data


def test_gemini_dimension_env_validation(monkeypatch):
    monkeypatch.setenv("RAG_GEMINI_EMBEDDING_DIMENSION", "1536")
    from API_RAG_NEW.config import get_gemini_embedding_dimension_env

    assert get_gemini_embedding_dimension_env("RAG_GEMINI_EMBEDDING_DIMENSION", 768) == 1536

    monkeypatch.setenv("RAG_GEMINI_EMBEDDING_DIMENSION", "999")
    assert get_gemini_embedding_dimension_env("RAG_GEMINI_EMBEDDING_DIMENSION", 768) == 768


def test_default_provider_env_resolves_to_local_sbert(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    from API_RAG_NEW.config import get_choice_env

    assert get_choice_env(
        "RAG_EMBEDDING_PROVIDER",
        "local_sbert",
        {"local_sbert", "gemini"},
    ) == "local_sbert"


def test_gemini_mode_requires_api_key(monkeypatch):
    config = importlib.import_module("API_RAG_NEW.config")
    monkeypatch.setattr(config, "get_gemini_api_key", lambda: None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.build_embedding_provider("gemini")


def test_collection_metadata_validation(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    runtime = FakeRuntime(
        provider="gemini",
        model_name="gemini-embedding-2",
        dimension=768,
    )

    matching = FakeCollection(
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        }
    )
    services._validate_collection_embedding_metadata(runtime, matching)

    mismatched = FakeCollection(
        metadata={
            "embedding_provider": "local_sbert",
            "embedding_model": "keepitreal/vietnamese-sbert",
            "embedding_dimension": 768,
        }
    )
    with pytest.raises(HTTPException) as exc_info:
        services._validate_collection_embedding_metadata(runtime, mismatched)
    assert exc_info.value.status_code == 400
    assert "different embedding provider/model" in exc_info.value.detail


def test_legacy_collection_rules(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")

    legacy = FakeCollection(metadata=None)
    local_runtime = FakeRuntime(provider="local_sbert")
    services._validate_collection_embedding_metadata(local_runtime, legacy)

    gemini_runtime = FakeRuntime(provider="gemini")
    with pytest.raises(HTTPException) as exc_info:
        services._validate_collection_embedding_metadata(gemini_runtime, legacy)
    assert exc_info.value.status_code == 400
    assert "no embedding metadata" in exc_info.value.detail


def test_embedding_collection_metadata_contains_integer_dimension(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    runtime = FakeRuntime(
        provider="local_sbert",
        model_name="fake-local",
        dimension=384,
    )

    metadata = services._embedding_collection_metadata(runtime, "desc")

    assert metadata == {
        "description": "desc",
        "embedding_provider": "local_sbert",
        "embedding_model": "fake-local",
        "embedding_dimension": 384,
    }
    assert isinstance(metadata["embedding_dimension"], int)


class FakeChromaClient:
    def __init__(self, collections=None):
        self.collections = {collection.name: collection for collection in collections or []}
        self.created = []

    def list_collections(self):
        return list(self.collections.values())

    def get_or_create_collection(self, *, name, metadata=None):
        collection = self.collections.get(name)
        if collection is None:
            collection = FakeCollection(metadata=metadata, name=name)
            self.collections[name] = collection
        self.created.append({"name": name, "metadata": metadata})
        return collection

    def get_collection(self, name):
        return self.collections[name]


def test_create_collection_writes_embedding_metadata(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    schemas = importlib.import_module("API_RAG_NEW.schemas")
    client = FakeChromaClient()
    runtime = FakeRuntime(
        provider="local_sbert",
        model_name="fake-local",
        dimension=384,
        chroma_client=client,
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    info = services.create_collection(
        schemas.CollectionCreateRequest(name="demo", description="desc")
    )

    assert info.name == "demo"
    assert client.created[0]["metadata"] == {
        "description": "desc",
        "embedding_provider": "local_sbert",
        "embedding_model": "fake-local",
        "embedding_dimension": 384,
    }


def test_update_collection_preserves_embedding_metadata(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    schemas = importlib.import_module("API_RAG_NEW.schemas")
    collection = FakeCollection(
        name="demo",
        metadata={
            "description": "old",
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    runtime = FakeRuntime(
        provider="gemini",
        model_name="gemini-embedding-2",
        dimension=768,
        chroma_client=FakeChromaClient([collection]),
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    services.update_collection(
        "demo",
        schemas.CollectionUpdateRequest(metadata={"description": "new", "owner": "qa"}),
    )

    assert collection.modify_calls[0]["metadata"] == {
        "description": "new",
        "owner": "qa",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-2",
        "embedding_dimension": 768,
    }


def test_update_collection_rejects_embedding_metadata_change(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    schemas = importlib.import_module("API_RAG_NEW.schemas")
    collection = FakeCollection(
        name="demo",
        metadata={
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
        },
    )
    runtime = FakeRuntime(
        provider="gemini",
        model_name="gemini-embedding-2",
        dimension=768,
        chroma_client=FakeChromaClient([collection]),
    )
    monkeypatch.setattr(services, "get_embedding_runtime", lambda provider: runtime)

    with pytest.raises(HTTPException) as exc_info:
        services.update_collection(
            "demo",
            schemas.CollectionUpdateRequest(
                metadata={
                    "embedding_provider": "local_sbert",
                    "embedding_model": "gemini-embedding-2",
                    "embedding_dimension": 768,
                }
            ),
        )

    assert exc_info.value.status_code == 400
    assert "Embedding metadata cannot be changed" in exc_info.value.detail


def test_runtime_payloads_include_safe_embedding_fields(monkeypatch):
    services = importlib.import_module("API_RAG_NEW.services")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "internal-secret")
    monkeypatch.setattr(services, "get_gemini_api_key", lambda: "secret-value")

    config_payload = services.runtime_config_payload()
    status_payload = services.runtime_status_payload()
    combined = f"{config_payload} {status_payload}"

    assert config_payload["embedding_routes"]["root"] == "local_sbert"
    assert config_payload["gemini_embedding"]["configured"] is True
    assert status_payload["gemini_embedding"]["dimension"] in {768, 1536, 3072}
    assert (
        status_payload["gemini_embedding"]["dimension"]
        == config_payload["gemini_embedding"]["dimension"]
    )
    assert "secret-value" not in combined
    assert "internal-secret" not in combined
