from __future__ import annotations

from typing import Any

from API_RAG_NEW.rag_pipeline import detect_query_hints, vector_search
from API_RAG_NEW.reranker import _build_rerank_prompt, _compact_metadata
from API_RAG_NEW.schemas import Citation, QueryRequest, QueryResponse


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeCollection:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.query_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        selected = self.records[: kwargs["n_results"]]
        return {
            "ids": [[record["id"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "documents": [[record["document"] for record in selected]],
            "distances": [[record.get("distance", 0.0) for record in selected]],
        }

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
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


def make_record(
    record_id: str,
    chunk_index: int,
    *,
    doc_id: str = "doc_1",
    parent_id: str | None = None,
    section_path: str | None = None,
    chunk_type: str | None = None,
    table_index: int | None = None,
    page_number: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "source": "demo.pdf",
        "source_type": "pdf",
        "chunk_index": chunk_index,
        "chunk": f"stale chunk {chunk_index}",
    }
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    if section_path is not None:
        metadata["section_path"] = section_path
        metadata["section_title"] = section_path.split(">")[-1].strip()
    if chunk_type is not None:
        metadata["chunk_type"] = chunk_type
    if table_index is not None:
        metadata["table_index"] = table_index
        metadata["table_title"] = "Table 1. Pricing"
        metadata["table_row_index"] = chunk_index
    if page_number is not None:
        metadata["page_number"] = page_number

    return {
        "id": record_id,
        "document": f"chunk {chunk_index}",
        "metadata": metadata,
    }


def response_fields(model_type: type[Any]) -> set[str]:
    fields = getattr(model_type, "model_fields", None)
    if fields is not None:
        return set(fields)
    return set(getattr(model_type, "__fields__", {}))


def test_reranker_compact_metadata_includes_phase5_metadata_without_chunk_text():
    metadata = {
        "source": "demo.pdf",
        "source_type": "pdf",
        "chunk_index": 7,
        "page_number": 2,
        "page_chunk_index": 1,
        "row_index": 4,
        "row_chunk_index": 1,
        "chunk_type": "table_row",
        "section_title": "Pricing",
        "section_path": "Overview > Pricing",
        "block_index": 3,
        "parent_id": "parent_1",
        "table_index": 1,
        "table_title": "Table 1. Pricing",
        "table_row_index": 2,
        "table_row_part_index": 1,
        "chunk": "large text should not be metadata",
    }

    compact = _compact_metadata(metadata)

    for key in metadata:
        if key != "chunk":
            assert compact[key] == metadata[key]
    assert "chunk" not in compact


def test_reranker_prompt_mentions_table_rows_structured_questions_and_json_only():
    candidate = type(
        "Candidate",
        (),
        {
            "id": "a",
            "document": "pricing row",
            "metadata": {"chunk_type": "table_row", "table_title": "Pricing"},
        },
    )()

    prompt = _build_rerank_prompt(
        "What is the price?",
        [candidate],
        1,
        query_hints={"asks_number_or_money": True},
    )

    assert "Rank candidate IDs by usefulness" in prompt
    assert "Prefer table_row chunks" in prompt
    assert "structured fields" in prompt
    assert "section_path or table_title" in prompt
    assert "Do not answer the question" in prompt
    assert '{"ranked_ids": ["id1", "id2"]}' in prompt
    assert '"asks_number_or_money": true' in prompt


def test_context_expansion_still_includes_chunk_index_neighbors():
    collection = FakeCollection(
        [
            make_record("center", 2),
            make_record("previous", 1),
            make_record("next", 3),
        ]
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        3,
        initial_top_k=1,
        include_neighbors=True,
        max_total_candidates=10,
    )

    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [2, 1, 3]


def test_context_expansion_can_include_same_parent_id_chunks():
    collection = FakeCollection(
        [
            make_record("center", 2, parent_id="parent_a"),
            make_record("sibling", 4, parent_id="parent_a"),
        ]
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        2,
        initial_top_k=1,
        include_neighbors=True,
        max_context_expansion_per_candidate=3,
    )

    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [2, 4]


def test_context_expansion_can_include_same_section_path_chunks():
    collection = FakeCollection(
        [
            make_record("center", 2, section_path="I. Overview > Pricing"),
            make_record("section-sibling", 5, section_path="I. Overview > Pricing"),
        ]
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        2,
        initial_top_k=1,
        include_neighbors=True,
        max_context_expansion_per_candidate=3,
    )

    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [2, 5]


def test_table_query_hint_prioritizes_table_rows_within_expansion_cap():
    collection = FakeCollection(
        [
            make_record("center", 1, section_path="I. Overview > Pricing"),
            make_record("semantic", 2, section_path="I. Overview > Pricing"),
            make_record(
                "table",
                3,
                section_path="I. Overview > Pricing",
                chunk_type="table_row",
                table_index=1,
                page_number=4,
            ),
        ]
    )

    llm = FakeLLM('{"ranked_ids": ["table"]}')

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "gia trong bang la bao nhieu",
        collection,
        [],
        1,
        initial_top_k=1,
        include_neighbors=True,
        reranker_type="llm",
        rerank_llm=llm,
        max_context_expansion_per_candidate=3,
        max_total_candidates=2,
    )

    assert [metadata["chunk_index"] for metadata in metadatas[0]] == [3]
    assert '"id": "table"' in llm.prompts[0]
    assert '"id": "semantic"' not in llm.prompts[0]
    assert "chunk_type=table_row" in retrieved_data
    assert "table_title=Table 1. Pricing" in retrieved_data
    assert "parent_id" not in retrieved_data


def test_total_candidates_are_capped_before_final_selection():
    collection = FakeCollection(
        [
            make_record("center", 1, section_path="I. Overview"),
            make_record("two", 2, section_path="I. Overview"),
            make_record("three", 3, section_path="I. Overview"),
            make_record("four", 4, section_path="I. Overview"),
        ]
    )

    metadatas, _ = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        10,
        initial_top_k=1,
        include_neighbors=True,
        max_context_expansion_per_candidate=10,
        max_total_candidates=3,
    )

    assert len(metadatas[0]) == 3


def test_distance_guard_is_disabled_by_default_even_with_weak_distances():
    collection = FakeCollection(
        [make_record("weak", 1) | {"distance": 999.0}]
    )

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        1,
        initial_top_k=1,
        max_distance=1.0,
    )

    assert len(metadatas[0]) == 1
    assert "chunk: chunk 1" in retrieved_data


def test_distance_guard_can_return_no_chunks_when_all_distances_are_weak():
    collection = FakeCollection(
        [
            make_record("weak-a", 1) | {"distance": 3.0},
            make_record("weak-b", 2) | {"distance": 4.0},
        ]
    )

    metadatas, retrieved_data = vector_search(
        FakeEmbeddingModel(),
        "question",
        collection,
        [],
        2,
        initial_top_k=2,
        enable_distance_guard=True,
        max_distance=1.0,
    )

    assert metadatas == [[]]
    assert retrieved_data == ""


def test_query_hints_detect_vietnamese_and_english_keywords():
    hints = detect_query_hints("Bang so sanh chi phi tren trang nao?")

    assert hints["asks_table_or_structured_info"] is True
    assert hints["asks_number_or_money"] is True
    assert hints["asks_source_or_page"] is True


def test_query_request_response_and_citation_fields_remain_compatible():
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
    assert "parent_id" not in response_fields(Citation)
