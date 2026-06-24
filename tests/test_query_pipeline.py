"""Tests cho rag_pipeline.py — vector search, context expansion, ranking."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from API_RAG_NEW.rag_pipeline import (
    RetrievedChunk,
    detect_query_hints,
    format_retrieved_data_with_markers,
    stable_record_id,
    vector_search,
)


class TestStableRecordId:
    def test_deterministic(self):
        id1 = stable_record_id("doc1", "file.pdf", "pdf", 1)
        id2 = stable_record_id("doc1", "file.pdf", "pdf", 1)
        assert id1 == id2

    def test_different_inputs_give_different_ids(self):
        id1 = stable_record_id("doc1", "file.pdf", "pdf", 1)
        id2 = stable_record_id("doc1", "file.pdf", "pdf", 2)
        assert id1 != id2

    def test_has_prefix(self):
        record_id = stable_record_id("a", "b")
        assert record_id.startswith("chunk_")


class TestDetectQueryHints:
    def test_detects_table_query(self):
        hints = detect_query_hints("bảng giá dịch vụ")
        assert hints["asks_table_or_structured_info"]

    def test_detects_money_query(self):
        hints = detect_query_hints("chi phí khám bệnh là bao nhiêu")
        assert hints["asks_number_or_money"]

    def test_detects_person_query(self):
        hints = detect_query_hints("ai là trưởng nhóm")
        assert hints["asks_person_or_entity"]

    def test_detects_source_query(self):
        hints = detect_query_hints("trang số mấy nói về vấn đề này")
        assert hints["asks_source_or_page"]

    def test_no_hints_for_generic_query(self):
        hints = detect_query_hints("quy trình đăng ký là gì")
        assert not any(hints.values())


class TestFormatRetrievedData:
    def test_formats_basic_metadata(self):
        metadatas = [
            {
                "source": "test.pdf",
                "chunk_index": 1,
                "chunk": "Nội dung thử nghiệm.",
                "page_number": 1,
            }
        ]
        result = format_retrieved_data_with_markers(metadatas)
        assert "[1]" in result
        assert "test.pdf" in result
        assert "Nội dung thử nghiệm." in result

    def test_empty_metadatas_returns_empty(self):
        result = format_retrieved_data_with_markers([])
        assert result == ""

    def test_multiple_chunks_indexed(self):
        metadatas = [
            {"source": "a.pdf", "chunk_index": 1, "chunk": "A"},
            {"source": "b.pdf", "chunk_index": 2, "chunk": "B"},
        ]
        result = format_retrieved_data_with_markers(metadatas)
        assert "[1]" in result
        assert "[2]" in result


class TestVectorSearch:
    def _make_search_results(self, count: int = 2) -> dict:
        ids = [[f"chunk_{i}" for i in range(count)]]
        metadatas = [[{"source": "test.pdf", "chunk_index": i, "doc_id": "doc1"} for i in range(count)]]
        documents = [[f"Nội dung chunk {i}" for i in range(count)]]
        distances = [[0.1 * (i + 1) for i in range(count)]]
        return {
            "ids": ids,
            "metadatas": metadatas,
            "documents": documents,
            "distances": distances,
        }

    def test_basic_vector_search(self, mock_embedding_model):
        collection = MagicMock()
        collection.query.return_value = self._make_search_results(2)

        with patch("API_RAG_NEW.rag_pipeline.acquire_embedding_slot"):
            metadatas, retrieved_data = vector_search(
                mock_embedding_model,
                "câu hỏi thử nghiệm",
                collection,
                2,
                reranker_type="none",
            )

        assert len(metadatas) == 1
        assert isinstance(retrieved_data, str)

    def test_returns_empty_on_empty_collection(self, mock_embedding_model):
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]
        }

        with patch("API_RAG_NEW.rag_pipeline.acquire_embedding_slot"):
            metadatas, retrieved_data = vector_search(
                mock_embedding_model,
                "query",
                collection,
                3,
                reranker_type="none",
            )

        assert metadatas == [[]]
        assert retrieved_data == ""

    def test_distance_guard_blocks_low_similarity(self, mock_embedding_model):
        collection = MagicMock()
        collection.query.return_value = self._make_search_results(2)

        with patch("API_RAG_NEW.rag_pipeline.acquire_embedding_slot"):
            metadatas, retrieved_data = vector_search(
                mock_embedding_model,
                "query",
                collection,
                2,
                reranker_type="none",
                enable_distance_guard=True,
                max_distance=0.001,  # rất nhỏ → guard kích hoạt
            )

        assert metadatas == [[]]
