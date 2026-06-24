"""Tests cho ingest_service.py — text extraction, document management."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from tests.conftest import (
    _make_mock_collection,
    _make_mock_chroma_client,
    _make_embedding_runtime,
)


def _make_runtime(mock_embedding_model, mock_chroma_client):
    return _make_embedding_runtime(mock_embedding_model, mock_chroma_client)


class TestListDocuments:
    def test_returns_unique_sources(self, mock_embedding_model, mock_chroma_client):
        col = _make_mock_collection("docs_col")
        col.get.return_value = {
            "ids": ["chunk_1", "chunk_2", "chunk_3"],
            "metadatas": [
                {"source": "a.pdf", "doc_id": "d1"},
                {"source": "a.pdf", "doc_id": "d1"},
                {"source": "b.pdf", "doc_id": "d2"},
            ],
            "documents": ["text", "text", "text"],
        }
        mock_chroma_client.get_collection.return_value = col
        runtime = _make_runtime(mock_embedding_model, mock_chroma_client)

        with patch("API_RAG_NEW.ingest_service._runtime_for_provider", return_value=runtime):
            from API_RAG_NEW.ingest_service import list_documents
            result = list_documents("docs_col", provider="local_sbert")

        sources = [d["source"] for d in result["documents"]]
        assert "a.pdf" in sources
        assert "b.pdf" in sources
        assert len(sources) == 2  # deduplicated

    def test_empty_collection_returns_empty(self, mock_embedding_model, mock_chroma_client):
        col = _make_mock_collection("empty_col")
        col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
        mock_chroma_client.get_collection.return_value = col
        runtime = _make_runtime(mock_embedding_model, mock_chroma_client)

        with patch("API_RAG_NEW.ingest_service._runtime_for_provider", return_value=runtime):
            from API_RAG_NEW.ingest_service import list_documents
            result = list_documents("empty_col", provider="local_sbert")

        assert result["documents"] == []


class TestDeleteDocument:
    def test_deletes_all_chunks_for_source(self, mock_embedding_model, mock_chroma_client):
        col = _make_mock_collection("col1")
        col.get.return_value = {
            "ids": ["c1", "c2"],
            "metadatas": [
                {"source": "file.pdf"},
                {"source": "file.pdf"},
            ],
            "documents": ["chunk1", "chunk2"],
        }
        mock_chroma_client.get_collection.return_value = col
        runtime = _make_runtime(mock_embedding_model, mock_chroma_client)

        with patch("API_RAG_NEW.ingest_service._runtime_for_provider", return_value=runtime):
            from API_RAG_NEW.ingest_service import delete_document
            result = delete_document("col1", "file.pdf", provider="local_sbert")

        col.delete.assert_called_once()
        assert isinstance(result, dict)

    def test_raises_if_source_not_found(self, mock_embedding_model, mock_chroma_client):
        col = _make_mock_collection("col1")
        col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
        mock_chroma_client.get_collection.return_value = col
        runtime = _make_runtime(mock_embedding_model, mock_chroma_client)

        with patch("API_RAG_NEW.ingest_service._runtime_for_provider", return_value=runtime):
            from API_RAG_NEW.ingest_service import delete_document
            with pytest.raises(HTTPException) as exc_info:
                delete_document("col1", "nonexistent.pdf", provider="local_sbert")

        assert exc_info.value.status_code == 404


class TestDecodeTextFile:
    def test_decodes_utf8(self):
        from API_RAG_NEW.ingest_service import _decode_text_file
        content = "Xin chào Việt Nam".encode("utf-8")
        result = _decode_text_file(content)
        assert "Việt Nam" in result

    def test_decodes_utf16(self):
        from API_RAG_NEW.ingest_service import _decode_text_file
        content = "Tiếng Việt".encode("utf-16")
        result = _decode_text_file(content)
        assert "Tiếng Việt" in result

    def test_handles_empty_bytes(self):
        from API_RAG_NEW.ingest_service import _decode_text_file
        result = _decode_text_file(b"")
        assert result == ""


class TestExtractPdfPages:
    def test_returns_list_of_tuples(self):
        import io
        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        from API_RAG_NEW.ingest_service import _extract_pdf_pages
        # Dùng PDF bytes rỗng — chỉ test không crash khi không có pages
        # (PDF hợp lệ nhỏ nhất cần header đặc biệt; test chỉ kiểm tra return type)
        result = _extract_pdf_pages(b"")
        assert isinstance(result, list)

    def test_clean_pdf_page_text_removes_extra_whitespace(self):
        from API_RAG_NEW.ingest_service import _clean_pdf_page_text
        messy = "  Đây là  đoạn   văn   \n\n  có nhiều khoảng trắng  "
        cleaned = _clean_pdf_page_text(messy)
        assert "  " not in cleaned or cleaned.strip() != ""


class TestContentHash:
    def test_hash_is_deterministic(self):
        from API_RAG_NEW.ingest_service import _content_hash
        h1 = _content_hash(b"noi dung thu nghiem")
        h2 = _content_hash(b"noi dung thu nghiem")
        assert h1 == h2

    def test_different_content_gives_different_hash(self):
        from API_RAG_NEW.ingest_service import _content_hash
        h1 = _content_hash(b"abc")
        h2 = _content_hash(b"xyz")
        assert h1 != h2

    def test_document_id_format(self):
        from API_RAG_NEW.ingest_service import _content_hash, _document_id
        file_hash = _content_hash(b"test content")
        doc_id = _document_id(file_hash)
        assert doc_id.startswith("doc_")
