"""Tests cho cấu trúc document — block detection, section path, metadata."""
from __future__ import annotations

import pytest


class TestDocumentMetadataSchema:
    def test_query_request_schema(self):
        from API_RAG_NEW.schemas import QueryRequest
        req = QueryRequest(query="câu hỏi tiếng Việt", number_docs_retrieval=3)
        assert req.query == "câu hỏi tiếng Việt"
        assert req.number_docs_retrieval == 3
        assert req.include_debug_info is False

    def test_query_request_debug_flag(self):
        from API_RAG_NEW.schemas import QueryRequest
        req = QueryRequest(query="test", include_debug_info=True)
        assert req.include_debug_info is True

    def test_query_request_default_docs_retrieval(self):
        from API_RAG_NEW.schemas import QueryRequest
        req = QueryRequest(query="q")
        assert req.number_docs_retrieval == 3

    def test_query_request_validates_range(self):
        from pydantic import ValidationError
        from API_RAG_NEW.schemas import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="q", number_docs_retrieval=0)  # ge=1
        with pytest.raises(ValidationError):
            QueryRequest(query="q", number_docs_retrieval=51)  # le=50

    def test_query_response_full_prompt_optional(self):
        from API_RAG_NEW.schemas import QueryResponse
        resp = QueryResponse(
            metadatas=[],
            retrieved_data="",
            answer="câu trả lời",
        )
        assert resp.full_prompt is None
        assert resp.citations == []

    def test_collection_create_request(self):
        from API_RAG_NEW.schemas import CollectionCreateRequest
        req = CollectionCreateRequest(name="test_col", description="mô tả test")
        assert req.name == "test_col"
        assert req.description == "mô tả test"

    def test_document_info_schema(self):
        from API_RAG_NEW.schemas import DocumentInfo
        doc = DocumentInfo(source="file.pdf", chunk_count=5, source_type="pdf")
        assert doc.source == "file.pdf"
        assert doc.chunk_count == 5
        assert doc.source_type == "pdf"
        assert doc.doc_id is None


class TestChunkMetadataIntegrity:
    def test_stable_record_id_matches_expectations(self):
        from API_RAG_NEW.rag_pipeline import stable_record_id
        rid = stable_record_id("doc_id_1", "file.pdf", "pdf", 0)
        assert rid.startswith("chunk_")
        assert len(rid) > 10

    def test_stable_record_id_collision_resistant(self):
        from API_RAG_NEW.rag_pipeline import stable_record_id
        ids = {
            stable_record_id("d", "f.pdf", "pdf", i)
            for i in range(100)
        }
        assert len(ids) == 100  # tất cả unique

    def test_retrieve_chunk_structure(self):
        from API_RAG_NEW.rag_pipeline import RetrievedChunk
        chunk = RetrievedChunk(
            id="chunk_abc",
            document="Nội dung chunk",
            metadata={"source": "test.pdf", "chunk_index": 1},
            distance=0.25,
        )
        assert chunk.id == "chunk_abc"
        assert chunk.distance == 0.25
        assert chunk.metadata["source"] == "test.pdf"


class TestCitationBuilding:
    def test_citations_from_metadatas(self):
        from API_RAG_NEW.citations import build_citations_from_metadatas
        metadatas = [
            {"source": "file.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "file.pdf", "page_number": 2, "chunk_index": 1},
            {"source": "other.pdf", "page_number": 1, "chunk_index": 0},
        ]
        citations = build_citations_from_metadatas(metadatas)
        sources = [c.source for c in citations]
        assert "file.pdf" in sources
        assert "other.pdf" in sources

    def test_empty_metadatas_returns_empty_citations(self):
        from API_RAG_NEW.citations import build_citations_from_metadatas
        assert build_citations_from_metadatas([]) == []

    def test_citation_has_expected_fields(self):
        from API_RAG_NEW.citations import build_citations_from_metadatas
        metadatas = [{"source": "test.pdf", "page_number": 3, "chunk_index": 1}]
        citations = build_citations_from_metadatas(metadatas)
        assert len(citations) == 1
        c = citations[0]
        assert c.source == "test.pdf"
        assert c.id == 1
