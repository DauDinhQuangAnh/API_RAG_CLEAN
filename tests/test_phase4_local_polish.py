from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from API_RAG_NEW.main import app


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeLLM:
    def generate_content(self, prompt: str) -> str:
        return "Cau tra loi co trich dan [1]."


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


def test_health_still_returns_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runtime_config_returns_safe_non_secret_config():
    response = TestClient(app).get("/runtime-config")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "rag_initial_top_k",
        "rag_final_top_n",
        "rag_include_neighbors",
        "rag_reranker_type",
        "gemini_model",
        "embedding_model_name",
        "chroma_db_path",
        "cors_origins",
    }.issubset(payload)
    assert "GEMINI_API_KEY" not in response.text
    assert "gemini_api_key" not in payload


def test_blank_query_returns_clear_400():
    response = TestClient(app).post(
        "/collections/demo/query",
        json={"query": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Query must not be empty."


def test_empty_upload_returns_clear_400():
    response = TestClient(app).post(
        "/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_ui_serves_standalone_html():
    response = TestClient(app).get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG API Test UI" in response.text


def test_query_response_preserves_phase3_fields(monkeypatch):
    from API_RAG_NEW import services

    collection = FakeCollection(
        [
            {
                "id": "chunk-1",
                "document": "final chunk",
                "metadata": {
                    "source": "demo.txt",
                    "source_type": "txt",
                    "chunk_index": 1,
                    "doc_id": "doc_1",
                },
            }
        ]
    )

    monkeypatch.setattr(services, "_get_collection_or_404", lambda name: collection)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "RAG_INITIAL_TOP_K", 1)
    monkeypatch.setattr(services, "RAG_INCLUDE_NEIGHBORS", False)
    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "none")
    monkeypatch.setattr(services, "_build_llm", lambda: FakeLLM())

    response = TestClient(app).post(
        "/collections/demo/query",
        json={"query": "question", "number_docs_retrieval": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert {
        "metadatas",
        "retrieved_data",
        "answer",
        "full_prompt",
        "citations",
    }.issubset(payload)
    assert payload["citations"][0]["id"] == 1
    assert payload["citations"][0]["snippet"] == "final chunk"
