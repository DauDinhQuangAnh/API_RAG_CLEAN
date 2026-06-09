from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from API_RAG_NEW.main import app
from API_RAG_NEW.schemas import QueryRequest, QueryResponse


INTERNAL_SECRET = "phase8-internal-secret"


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
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture(autouse=True)
def reset_limiters():
    from API_RAG_NEW.concurrency import reset_limiters_for_tests

    reset_limiters_for_tests()
    yield
    reset_limiters_for_tests()


def make_record(record_id: str = "a") -> dict[str, Any]:
    return {
        "id": record_id,
        "document": "answer evidence",
        "metadata": {
            "doc_id": "doc_1",
            "source": "demo.txt",
            "source_type": "txt",
            "chunk_index": 1,
        },
    }


def configure_query_service(monkeypatch, llm: FakeLLM) -> None:
    from API_RAG_NEW import services

    monkeypatch.setattr(
        services,
        "_get_collection_or_404",
        lambda name: FakeCollection([make_record()]),
    )
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "RAG_FINAL_TOP_N", 1)
    monkeypatch.setattr(services, "RAG_INITIAL_TOP_K", 1)
    monkeypatch.setattr(services, "RAG_INCLUDE_NEIGHBORS", False)
    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "none")
    monkeypatch.setattr(services, "_build_llm", lambda: llm)


def test_config_defaults_are_sane():
    from API_RAG_NEW import config

    assert config.RAG_MAX_CONCURRENT_QUERIES >= 1
    assert config.RAG_MAX_CONCURRENT_LLM_CALLS >= 1
    assert config.RAG_QUERY_QUEUE_TIMEOUT_SECONDS >= 0
    assert config.RAG_LLM_QUEUE_TIMEOUT_SECONDS >= 0
    assert isinstance(config.RAG_ENABLE_FINAL_ANSWER_FALLBACK, bool)


def test_timeout_config_parser_falls_back_for_empty_invalid_and_negative_values(
    monkeypatch,
):
    from API_RAG_NEW.config import get_float_env

    monkeypatch.setenv("EMPTY_TIMEOUT", "")
    monkeypatch.setenv("INVALID_TIMEOUT", "not-a-number")
    monkeypatch.setenv("NEGATIVE_TIMEOUT", "-1")
    monkeypatch.setenv("VALID_TIMEOUT", "0.25")

    assert get_float_env("EMPTY_TIMEOUT", 2.0) == 2.0
    assert get_float_env("INVALID_TIMEOUT", 2.0) == 2.0
    assert get_float_env("NEGATIVE_TIMEOUT", 2.0) == 2.0
    assert get_float_env("MISSING_TIMEOUT", 2.0) == 2.0
    assert get_float_env("VALID_TIMEOUT", 2.0) == 0.25


def test_query_limiter_rejects_overload_and_releases_slots():
    from API_RAG_NEW.concurrency import (
        acquire_query_slot,
        concurrency_status_payload,
        reset_limiters_for_tests,
    )

    reset_limiters_for_tests(max_concurrent_queries=1)
    with acquire_query_slot(timeout=0):
        status = concurrency_status_payload()["query"]
        assert status["active"] == 1
        assert status["available"] == 0
        with pytest.raises(HTTPException) as exc_info:
            with acquire_query_slot(timeout=0):
                pass

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "RAG server is busy. Please try again later."
    status = concurrency_status_payload()["query"]
    assert status["active"] == 0
    assert status["available"] == 1
    assert status["rejected"] == 1


def test_llm_limiter_rejects_overload_and_releases_slots():
    from API_RAG_NEW.concurrency import (
        LLMOverloadedError,
        acquire_llm_slot,
        concurrency_status_payload,
        reset_limiters_for_tests,
    )

    reset_limiters_for_tests(max_concurrent_llm_calls=1)
    with acquire_llm_slot(timeout=0):
        assert concurrency_status_payload()["llm"]["active"] == 1
        with pytest.raises(LLMOverloadedError):
            with acquire_llm_slot(timeout=0):
                pass

    status = concurrency_status_payload()["llm"]
    assert status["active"] == 0
    assert status["available"] == 1
    assert status["rejected"] == 1


def test_query_endpoint_returns_503_when_busy_and_succeeds_when_available(monkeypatch):
    from API_RAG_NEW import security, services
    from API_RAG_NEW import concurrency
    from API_RAG_NEW.concurrency import acquire_query_slot, reset_limiters_for_tests

    calls: list[str] = []

    def fake_query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
        calls.append(collection_name)
        return QueryResponse(
            metadatas=[[]],
            retrieved_data="",
            answer="answer",
            full_prompt="prompt",
            citations=[],
        )

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", None)
    monkeypatch.setattr(services, "query_collection", fake_query_collection)
    monkeypatch.setattr(concurrency, "RAG_QUERY_QUEUE_TIMEOUT_SECONDS", 0.01)
    reset_limiters_for_tests(max_concurrent_queries=1)
    client = TestClient(app)

    with acquire_query_slot(timeout=0):
        busy = client.post("/collections/demo/query", json={"query": "question"})

    available = client.post("/collections/demo/query", json={"query": "question"})

    assert busy.status_code == 503
    assert busy.json()["detail"] == "RAG server is busy. Please try again later."
    assert available.status_code == 200
    assert available.json()["answer"] == "answer"
    assert calls == ["demo"]


def test_online_llm_generate_content_uses_llm_limiter():
    from API_RAG_NEW.concurrency import concurrency_status_payload, reset_limiters_for_tests
    from llms.onlinellms import GEMINI_PROVIDER, OnlineLLMs

    class FakeResponse:
        text = "ok"

    class FakeModels:
        def generate_content(self, **kwargs):
            assert concurrency_status_payload()["llm"]["active"] == 1
            assert kwargs["model"] == "test-model"
            assert kwargs["contents"] == "prompt"
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    reset_limiters_for_tests(max_concurrent_llm_calls=1)
    llm = OnlineLLMs.__new__(OnlineLLMs)
    llm.name = GEMINI_PROVIDER
    llm.client = FakeClient()
    llm.model_version = "test-model"

    assert llm.generate_content("prompt") == "ok"
    assert concurrency_status_payload()["llm"]["active"] == 0


def test_final_answer_fallback_catches_only_generate_content_failure(monkeypatch):
    from API_RAG_NEW import services

    configure_query_service(monkeypatch, FakeLLM(RuntimeError("gemini overloaded")))
    monkeypatch.setattr(services, "RAG_ENABLE_FINAL_ANSWER_FALLBACK", True)

    response = services.query_collection("demo", QueryRequest(query="question"))

    assert response.answer == services.FINAL_ANSWER_FALLBACK_MESSAGE
    assert response.metadatas
    assert "answer evidence" in response.retrieved_data
    assert "Reference data:" in response.full_prompt
    assert response.citations


def test_disabled_final_answer_fallback_re_raises(monkeypatch):
    from API_RAG_NEW import services

    configure_query_service(monkeypatch, FakeLLM(RuntimeError("gemini overloaded")))
    monkeypatch.setattr(services, "RAG_ENABLE_FINAL_ANSWER_FALLBACK", False)

    with pytest.raises(RuntimeError, match="gemini overloaded"):
        services.query_collection("demo", QueryRequest(query="question"))


def test_build_llm_configuration_errors_are_not_swallowed_by_fallback(monkeypatch):
    from API_RAG_NEW import services

    configure_query_service(monkeypatch, FakeLLM("unused"))
    monkeypatch.setattr(services, "RAG_ENABLE_FINAL_ANSWER_FALLBACK", True)

    def fail_build_llm():
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not configured")

    monkeypatch.setattr(services, "_build_llm", fail_build_llm)

    with pytest.raises(HTTPException) as exc_info:
        services.query_collection("demo", QueryRequest(query="question"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "GEMINI_API_KEY not configured"


def test_runtime_config_includes_concurrency_settings_status_and_no_raw_secrets(
    monkeypatch,
):
    from API_RAG_NEW import security, services

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    monkeypatch.setattr(services, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)

    response = TestClient(app).get(
        "/runtime-config",
        headers={"X-Internal-API-Key": INTERNAL_SECRET},
    )

    assert response.status_code == 200
    payload = response.json()
    for key in (
        "rag_max_concurrent_queries",
        "rag_max_concurrent_llm_calls",
        "rag_query_queue_timeout_seconds",
        "rag_llm_queue_timeout_seconds",
        "rag_enable_final_answer_fallback",
        "concurrency",
    ):
        assert key in payload
    assert payload["rag_internal_api_key_enabled"] is True
    for forbidden_key in (
        "GEMINI_API_KEY",
        "RAG_INTERNAL_API_KEY",
        "rag_internal_api_key",
        "internal_api_key",
        "api_key",
    ):
        assert forbidden_key not in payload
    assert INTERNAL_SECRET not in response.text


def test_runtime_status_is_protected_and_safe(monkeypatch):
    from API_RAG_NEW import security

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    client = TestClient(app)

    blocked = client.get("/runtime-status")
    allowed = client.get(
        "/runtime-status",
        headers={"X-Internal-API-Key": INTERNAL_SECRET},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 200

    payload = allowed.json()
    assert set(payload) == {
        "health",
        "concurrency",
        "embedding_model_name",
        "gemini_model",
        "gemini_reranker_model",
    }
    safe_text = allowed.text
    for forbidden in (
        INTERNAL_SECRET,
        "GEMINI_API_KEY",
        "RAG_INTERNAL_API_KEY",
        "X-Internal-API-Key",
        "prompt",
        "retrieved_data",
        "user_query",
        "query_text",
    ):
        assert forbidden not in safe_text
