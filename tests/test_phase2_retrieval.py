from __future__ import annotations

from typing import Any

from API_RAG_NEW.rag_pipeline import vector_search
from API_RAG_NEW.reranker import rerank_candidate_ids
from API_RAG_NEW.schemas import QueryRequest


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeLLM:
    def __init__(self, response: str = "answer") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingLLM:
    def generate_content(self, prompt: str) -> str:
        raise RuntimeError("reranker failed")


class FakeCollection:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        fail_filtered_get: bool = False,
    ) -> None:
        self.records = records
        self.fail_filtered_get = fail_filtered_get
        self.query_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        n_results = kwargs["n_results"]
        selected = self.records[:n_results]
        return {
            "ids": [[record["id"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "documents": [[record["document"] for record in selected]],
            "distances": [[record.get("distance", 0.0) for record in selected]],
        }

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        if "where" in kwargs and self.fail_filtered_get:
            raise TypeError("where is not supported")

        records = self.records
        where = kwargs.get("where")
        if where and "doc_id" in where:
            records = [
                record
                for record in records
                if record["metadata"].get("doc_id") == where["doc_id"]
            ]

        return {
            "ids": [record["id"] for record in records],
            "metadatas": [record["metadata"] for record in records],
            "documents": [record["document"] for record in records],
        }


def make_record(record_id: str, chunk_index: int, doc_id: str = "doc_1"):
    return {
        "id": record_id,
        "document": f"chunk {chunk_index}",
        "metadata": {
            "doc_id": doc_id,
            "source": "demo.txt",
            "source_type": "txt",
            "chunk_index": chunk_index,
            "chunk": f"stale chunk {chunk_index}",
        },
    }


def response_payload(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_vector_search_uses_expanded_top_k_and_documents():
    collection = FakeCollection([make_record("a", 1), make_record("b", 2)])

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        1,
        initial_top_k=20,
    )

    assert collection.query_calls[0]["n_results"] == 20
    assert collection.query_calls[0]["include"] == [
        "metadatas",
        "documents",
        "distances",
    ]
    assert metadatas == [[collection.records[0]["metadata"] | {"chunk": "chunk 1"}]]
    assert "chunk: chunk 1" in retrieved_data


def test_final_count_uses_env_default_unless_request_explicit(monkeypatch):
    from API_RAG_NEW import services

    monkeypatch.setattr(services, "RAG_FINAL_TOP_N", 6)

    assert services._resolve_final_docs_retrieval(QueryRequest(query="q")) == 6
    assert (
        services._resolve_final_docs_retrieval(
            QueryRequest(query="q", number_docs_retrieval=2)
        )
        == 2
    )

    class PydanticV1Request:
        number_docs_retrieval = 4
        __fields_set__ = {"number_docs_retrieval"}

    assert services._resolve_final_docs_retrieval(PydanticV1Request()) == 4


def test_neighbor_expansion_uses_filtered_doc_id_get():
    collection = FakeCollection(
        [
            make_record("center", 2),
            make_record("previous", 1),
            make_record("next", 3),
        ]
    )

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        3,
        initial_top_k=1,
        include_neighbors=True,
    )

    assert collection.get_calls[0]["where"] == {"doc_id": "doc_1"}
    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [2, 1, 3]
    assert "chunk: chunk 2" in retrieved_data
    assert "chunk: chunk 1" in retrieved_data
    assert "chunk: chunk 3" in retrieved_data


def test_neighbor_expansion_fallback_filters_records_in_python():
    collection = FakeCollection(
        [
            make_record("center", 2),
            make_record("previous", 1),
            make_record("next", 3),
            make_record("other-doc", 1, doc_id="doc_2"),
        ],
        fail_filtered_get=True,
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        3,
        initial_top_k=1,
        include_neighbors=True,
    )

    assert len(collection.get_calls) == 2
    assert collection.get_calls[1]["include"] == ["metadatas", "documents"]
    assert [metadata["doc_id"] for metadata in metadatas[0]] == [
        "doc_1",
        "doc_1",
        "doc_1",
    ]


def test_gemini_reranker_orders_candidate_ids_only_and_fills_missing():
    candidates = [
        type("Candidate", (), {"id": "a", "document": "alpha", "metadata": {}})(),
        type("Candidate", (), {"id": "b", "document": "beta", "metadata": {}})(),
        type("Candidate", (), {"id": "c", "document": "gamma", "metadata": {}})(),
    ]
    llm = FakeLLM('{"ranked_ids": ["c", "invalid"]}')

    ranked_ids = rerank_candidate_ids("question", candidates, 3, llm)

    assert ranked_ids == ["c", "a", "b"]
    assert "Do not answer the question" in llm.prompts[0]
    assert "Rank candidate IDs" in llm.prompts[0]


def test_reranker_malformed_invalid_or_failing_output_falls_back():
    candidates = [
        type("Candidate", (), {"id": "a", "document": "alpha", "metadata": {}})(),
        type("Candidate", (), {"id": "b", "document": "beta", "metadata": {}})(),
    ]

    assert rerank_candidate_ids("q", candidates, 2, FakeLLM("not json")) == ["a", "b"]
    assert rerank_candidate_ids("q", candidates, 2, FakeLLM('{"ranked_ids": ["x"]}')) == [
        "a",
        "b",
    ]
    assert rerank_candidate_ids("q", candidates, 2, FailingLLM()) == ["a", "b"]


def test_vector_search_uses_reranked_ids_and_fills_from_vector_order():
    collection = FakeCollection(
        [make_record("a", 1), make_record("b", 2), make_record("c", 3)]
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        2,
        initial_top_k=3,
        reranker_type="llm",
        rerank_llm=FakeLLM('{"ranked_ids": ["c", "invalid"]}'),
    )

    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [3, 1]


def test_query_response_keeps_existing_fields_and_uses_final_chunks(monkeypatch):
    from API_RAG_NEW import services

    collection = FakeCollection(
        [make_record("a", 1), make_record("b", 2), make_record("c", 3)]
    )
    answer_llm = FakeLLM("final answer")

    monkeypatch.setattr(services, "_get_collection_or_404", lambda name: collection)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "RAG_FINAL_TOP_N", 2)
    monkeypatch.setattr(services, "RAG_INITIAL_TOP_K", 3)
    monkeypatch.setattr(services, "RAG_INCLUDE_NEIGHBORS", False)
    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "none")
    monkeypatch.setattr(services, "_build_llm", lambda: answer_llm)

    response = services.query_collection("demo", QueryRequest(query="question"))
    payload = response_payload(response)

    assert set(payload) == {"metadatas", "retrieved_data", "answer", "full_prompt"}
    assert len(response.metadatas[0]) == 2
    assert "chunk: chunk 1" in response.full_prompt
    assert "chunk: chunk 2" in response.full_prompt
    assert "chunk: chunk 3" not in response.full_prompt
    assert response.answer == "final answer"
