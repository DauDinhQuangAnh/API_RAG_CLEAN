from __future__ import annotations

from typing import Any

from API_RAG_NEW.citations import (
    MAX_CITATION_SNIPPET_CHARS,
    build_citations_from_metadatas,
)
from API_RAG_NEW.rag_pipeline import vector_search
from API_RAG_NEW.schemas import Citation, QueryRequest


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeLLM:
    def __init__(self, response: str = "Câu trả lời có trích dẫn [1].") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeCollection:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        selected = self.records[: kwargs["n_results"]]
        return {
            "ids": [[record["id"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "documents": [[record["document"] for record in selected]],
            "distances": [[record.get("distance", 0.0) for record in selected]],
        }


def response_payload(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_citation_schema_requires_only_id_and_snippet():
    citation = Citation(id=1, snippet="preview")

    assert citation.source is None
    assert citation.source_type is None
    assert citation.page_number is None
    assert citation.chunk_index is None
    assert citation.page_chunk_index is None
    assert citation.row_index is None
    assert citation.row_chunk_index is None
    assert citation.doc_id is None


def test_citation_builder_preserves_metadata_and_uses_short_preview():
    long_chunk = "first line\n\nsecond line " + ("x" * 400)

    citations = build_citations_from_metadatas(
        [
            {
                "source": "document.pdf",
                "source_type": "pdf",
                "page_number": 2,
                "chunk_index": 5,
                "page_chunk_index": 1,
                "doc_id": "doc_pdf",
                "chunk": long_chunk,
            },
            {
                "source": "legacy-row-document.txt",
                "source_type": "txt",
                "chunk_index": "12",
                "row_index": "4",
                "row_chunk_index": "2",
                "doc_id": "doc_row",
                "chunk": "row text",
            },
        ]
    )

    assert citations[0].id == 1
    assert citations[0].source == "document.pdf"
    assert citations[0].source_type == "pdf"
    assert citations[0].page_number == 2
    assert citations[0].chunk_index == 5
    assert citations[0].page_chunk_index == 1
    assert citations[0].row_index is None
    assert citations[0].row_chunk_index is None
    assert citations[0].doc_id == "doc_pdf"
    assert "\n" not in citations[0].snippet
    assert citations[0].snippet.startswith("first line second line")
    assert citations[0].snippet.endswith("...")
    assert len(citations[0].snippet) <= MAX_CITATION_SNIPPET_CHARS + 3
    assert citations[0].snippet != long_chunk

    assert citations[1].id == 2
    assert citations[1].row_index == 4
    assert citations[1].row_chunk_index == 2
    assert citations[1].page_number is None
    assert citations[1].snippet == "row text"


def test_retrieved_data_uses_markers_for_pdf_txt_and_legacy_row_chunks():
    records = [
        {
            "id": "pdf",
            "document": "pdf chunk",
            "metadata": {
                "source": "document.pdf",
                "source_type": "pdf",
                "page_number": 2,
                "chunk_index": 5,
                "page_chunk_index": 1,
                "doc_id": "doc_pdf",
            },
        },
        {
            "id": "txt",
            "document": "txt chunk",
            "metadata": {
                "source": "document.txt",
                "source_type": "txt",
                "chunk_index": 1,
                "doc_id": "doc_txt",
            },
        },
        {
            "id": "legacy-row",
            "document": "row chunk",
            "metadata": {
                "source": "legacy-row-document.txt",
                "source_type": "txt",
                "row_index": 4,
                "row_chunk_index": 1,
                "chunk_index": 12,
                "doc_id": "doc_row",
            },
        },
    ]

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "question",
        FakeCollection(records),
        3,
        initial_top_k=3,
    )
    citations = build_citations_from_metadatas(metadatas[0])

    assert "[1] source=document.pdf | page=2 | chunk_index=5" in retrieved_data
    assert "[2] source=document.txt | page=N/A | chunk_index=1" in retrieved_data
    assert "[3] source=legacy-row-document.txt | row=4 | chunk_index=12" in retrieved_data
    assert "chunk: pdf chunk" in retrieved_data
    assert "chunk: txt chunk" in retrieved_data
    assert "chunk: row chunk" in retrieved_data
    assert [citation.id for citation in citations] == [1, 2, 3]


def test_query_response_includes_backend_citations_and_prompt_instructions(
    monkeypatch,
):
    from API_RAG_NEW import services

    collection = FakeCollection(
        [
            {
                "id": "a",
                "document": "final chunk one",
                "metadata": {
                    "source": "document.pdf",
                    "source_type": "pdf",
                    "page_number": 2,
                    "chunk_index": 5,
                    "page_chunk_index": 1,
                    "doc_id": "doc_pdf",
                    "chunk": "stale chunk",
                },
            },
            {
                "id": "b",
                "document": "final chunk two",
                "metadata": {
                    "source": "document.txt",
                    "source_type": "txt",
                    "chunk_index": 1,
                    "doc_id": "doc_txt",
                },
            },
        ]
    )
    answer_llm = FakeLLM()

    monkeypatch.setattr(services, "_get_collection_or_404", lambda name: collection)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "RAG_FINAL_TOP_N", 2)
    monkeypatch.setattr(services, "RAG_INITIAL_TOP_K", 2)
    monkeypatch.setattr(services, "RAG_INCLUDE_NEIGHBORS", False)
    monkeypatch.setattr(services, "RAG_RERANKER_TYPE", "none")
    monkeypatch.setattr(services, "_build_llm", lambda: answer_llm)

    response = services.query_collection("demo", QueryRequest(query="question"))
    payload = response_payload(response)

    assert set(payload) == {
        "metadatas",
        "retrieved_data",
        "answer",
        "full_prompt",
        "citations",
    }
    assert response.answer == "Câu trả lời có trích dẫn [1]."
    assert len(response.metadatas[0]) == 2
    assert len(response.citations) == 2
    assert response.citations[0].id == 1
    assert response.citations[0].snippet == "final chunk one"
    assert response.citations[1].id == 2
    assert response.citations[1].snippet == "final chunk two"
    assert "[1] source=document.pdf | page=2 | chunk_index=5" in response.retrieved_data
    assert "[2] source=document.txt | page=N/A | chunk_index=1" in response.retrieved_data
    assert "marker dạng [1], [2], [3]" in response.full_prompt
    assert "ví dụ [1] hoặc [1][2]" in response.full_prompt
    assert answer_llm.prompts == [response.full_prompt]
