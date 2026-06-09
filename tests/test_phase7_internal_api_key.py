from __future__ import annotations

from fastapi.testclient import TestClient

from API_RAG_NEW.main import app
from API_RAG_NEW.schemas import QueryRequest, QueryResponse


INTERNAL_SECRET = "phase7-test-secret"


def test_auth_disabled_allows_protected_endpoint_without_header(monkeypatch):
    from API_RAG_NEW import security, services

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", None)
    monkeypatch.setattr(services, "RAG_INTERNAL_API_KEY", None)

    response = TestClient(app).get("/runtime-config")

    assert response.status_code == 200
    assert response.json()["rag_internal_api_key_enabled"] is False


def test_auth_enabled_rejects_missing_and_wrong_key(monkeypatch):
    from API_RAG_NEW import security

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    client = TestClient(app)

    missing = client.get("/collections")
    wrong = client.get(
        "/collections",
        headers={"X-Internal-API-Key": "wrong-secret"},
    )

    assert missing.status_code == 401
    assert missing.json()["detail"] == "Invalid internal API key."
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid internal API key."


def test_auth_enabled_allows_correct_key(monkeypatch):
    from API_RAG_NEW import security, services

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    monkeypatch.setattr(
        services,
        "list_collections",
        lambda: {"collections": ["demo"]},
    )

    response = TestClient(app).get(
        "/collections",
        headers={"X-Internal-API-Key": INTERNAL_SECRET},
    )

    assert response.status_code == 200
    assert response.json() == {"collections": ["demo"]}


def test_health_remains_public_when_auth_enabled(monkeypatch):
    from API_RAG_NEW import security

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_requires_key_and_reaches_pipeline_with_correct_key(monkeypatch):
    from API_RAG_NEW import security, services

    calls: list[tuple[str, QueryRequest]] = []

    def fake_query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
        calls.append((collection_name, req))
        return QueryResponse(
            metadatas=[[]],
            retrieved_data="",
            answer="answer",
            full_prompt="prompt",
            citations=[],
        )

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    monkeypatch.setattr(services, "query_collection", fake_query_collection)
    client = TestClient(app)

    blocked = client.post(
        "/collections/demo/query",
        json={"query": "question", "number_docs_retrieval": 1},
    )
    allowed = client.post(
        "/collections/demo/query",
        headers={"X-Internal-API-Key": INTERNAL_SECRET},
        json={"query": "question", "number_docs_retrieval": 1},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["answer"] == "answer"
    assert len(calls) == 1
    assert calls[0][0] == "demo"
    assert calls[0][1].query == "question"


def test_runtime_config_requires_key_and_never_exposes_secret(monkeypatch):
    from API_RAG_NEW import security, services

    monkeypatch.setattr(security, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    monkeypatch.setattr(services, "RAG_INTERNAL_API_KEY", INTERNAL_SECRET)
    client = TestClient(app)

    blocked = client.get("/runtime-config")
    allowed = client.get(
        "/runtime-config",
        headers={"X-Internal-API-Key": INTERNAL_SECRET},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 200

    payload = allowed.json()
    assert payload["rag_internal_api_key_enabled"] is True
    for forbidden_key in (
        "RAG_INTERNAL_API_KEY",
        "rag_internal_api_key",
        "internal_api_key",
        "api_key",
    ):
        assert forbidden_key not in payload
    assert INTERNAL_SECRET not in allowed.text
