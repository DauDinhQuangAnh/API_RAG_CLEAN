from __future__ import annotations

import importlib
from typing import Any

from API_RAG_NEW.schemas import Citation, QueryRequest, QueryResponse


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeCollection:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def query(self, **kwargs):
        selected = self.records[: kwargs["n_results"]]
        return {
            "ids": [[record["id"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "documents": [[record["document"] for record in selected]],
            "distances": [[record.get("distance", 0.0) for record in selected]],
        }


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def response_fields(model_type: type[Any]) -> set[str]:
    fields = getattr(model_type, "model_fields", None)
    if fields is not None:
        return set(fields)
    return set(getattr(model_type, "__fields__", {}))


def test_gemini_reranker_model_falls_back_to_gemini_model_when_unset(monkeypatch):
    import chromadb
    import dotenv
    import download_model
    import API_RAG_NEW.config as config

    try:
        with monkeypatch.context() as isolated:
            isolated.setenv("GEMINI_MODEL", "main-model-for-test")
            isolated.delenv("GEMINI_RERANKER_MODEL", raising=False)
            isolated.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
            isolated.setattr(chromadb, "PersistentClient", lambda path: object())
            isolated.setattr(
                download_model,
                "ensure_embedding_model",
                lambda preferred_model: (object(), "fake-embedding", None),
            )
            reloaded = importlib.reload(config)
            assert reloaded.GEMINI_MODEL == "main-model-for-test"
            assert reloaded.GEMINI_RERANKER_MODEL == "main-model-for-test"
    finally:
        importlib.reload(config)


def test_runtime_config_includes_main_and_reranker_models_without_secrets(monkeypatch):
    from API_RAG_NEW import services

    monkeypatch.setattr(services, "GEMINI_MODEL", "main-model")
    monkeypatch.setattr(services, "GEMINI_RERANKER_MODEL", "reranker-model")

    payload = services.runtime_config_payload()

    assert payload["gemini_model"] == "main-model"
    assert payload["gemini_reranker_model"] == "reranker-model"
    assert "GEMINI_API_KEY" not in str(payload)
    assert "gemini_api_key" not in payload


def test_build_llm_passes_model_override_to_online_llms(monkeypatch):
    from API_RAG_NEW import services

    calls: list[dict[str, Any]] = []

    class FakeOnlineLLMs:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(services, "get_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(services, "OnlineLLMs", FakeOnlineLLMs)

    llm = services._build_llm(model_version="custom-reranker-model")

    assert isinstance(llm, FakeOnlineLLMs)
    assert calls == [
        {
            "name": services.GEMINI_PROVIDER,
            "api_key": "fake-key",
            "model_version": "custom-reranker-model",
        }
    ]


def test_optional_rerank_llm_uses_reranker_model_and_preserves_fallback(monkeypatch):
    from API_RAG_NEW import services

    calls: list[str | None] = []

    def fake_build_llm(*, model_version=None, api_key=None):
        calls.append(model_version)
        return FakeLLM('{"ranked_ids": ["a"]}')

    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "llm")
    monkeypatch.setattr(services, "GEMINI_RERANKER_MODEL", "reranker-model")
    monkeypatch.setattr(services, "_build_llm", fake_build_llm)

    assert isinstance(services._build_optional_rerank_llm(), FakeLLM)
    assert calls == ["reranker-model"]

    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "none")
    assert services._build_optional_rerank_llm() is None

    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "llm")
    monkeypatch.setattr(
        services,
        "_build_llm",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    assert services._build_optional_rerank_llm() is None


def test_query_collection_uses_main_model_for_final_answer_not_reranker(monkeypatch):
    from API_RAG_NEW import services

    collection = FakeCollection(
        [
            {
                "id": "a",
                "document": "answer evidence",
                "metadata": {
                    "doc_id": "doc_1",
                    "source": "demo.txt",
                    "source_type": "txt",
                    "chunk_index": 1,
                },
            }
        ]
    )
    rerank_llm = FakeLLM('{"ranked_ids": ["a"]}')
    answer_llm = FakeLLM("final answer from main model")
    model_versions: list[str | None] = []

    def fake_build_llm(api_key=None, model_version=None):
        model_versions.append(model_version)
        if model_version == "reranker-model":
            return rerank_llm
        if model_version is None:
            return answer_llm
        raise AssertionError(f"unexpected model_version: {model_version}")

    monkeypatch.setattr(services, "_get_collection_or_404", lambda name: collection)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "RAG_FINAL_TOP_N", 1)
    monkeypatch.setattr(services, "RAG_INITIAL_TOP_K", 1)
    monkeypatch.setattr(services, "RAG_INCLUDE_NEIGHBORS", False)
    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "llm")
    monkeypatch.setattr(services, "GEMINI_RERANKER_MODEL", "reranker-model")
    monkeypatch.setattr(services, "_build_llm", fake_build_llm)

    response = services.query_collection("demo", QueryRequest(query="question"))

    assert response.answer == "final answer from main model"
    assert model_versions == ["reranker-model", None]
    assert len(rerank_llm.prompts) == 1
    assert len(answer_llm.prompts) == 1
    assert "Candidates:" in rerank_llm.prompts[0]
    assert "Reference data:" in answer_llm.prompts[0]


def test_query_request_response_and_citation_schemas_remain_compatible():
    assert response_fields(QueryRequest) == {
        "query",
        "columns_to_answer",
        "number_docs_retrieval",
    }
    assert response_fields(QueryResponse) == {
        "metadatas",
        "retrieved_data",
        "answer",
        "full_prompt",
        "citations",
    }
    assert "gemini_reranker_model" not in response_fields(Citation)
