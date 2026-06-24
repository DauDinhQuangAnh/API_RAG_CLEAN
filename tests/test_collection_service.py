"""Tests cho collection_service.py — CRUD operations."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from tests.conftest import (
    _make_mock_collection,
    _make_mock_chroma_client,
    _make_embedding_runtime,
)


@pytest.fixture
def runtime(mock_embedding_model, mock_chroma_client):
    from tests.conftest import _make_embedding_runtime
    return _make_embedding_runtime(mock_embedding_model, mock_chroma_client)


class TestListCollections:
    def test_returns_collection_names(self, runtime):
        col = _make_mock_collection("my_col")
        runtime.chroma_client.list_collections.return_value = [col]

        with patch(
            "API_RAG_NEW._services_shared._runtime_for_provider",
            return_value=runtime,
        ), patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import list_collections
            result = list_collections(provider="local_sbert")

        assert "collections" in result
        assert "my_col" in result["collections"]

    def test_excludes_gemini_collections_from_local(self, runtime):
        gemini_col = _make_mock_collection("gemini.some_col")
        gemini_col.metadata = {
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-2",
            "embedding_dimension": 768,
            "storage_collection_name": "gemini.some_col",
        }
        runtime.chroma_client.list_collections.return_value = [gemini_col]

        with patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import list_collections
            result = list_collections(provider="local_sbert")

        assert result["collections"] == []


class TestCreateCollection:
    def test_creates_new_collection(self, runtime):
        runtime.chroma_client.list_collections.return_value = []
        new_col = _make_mock_collection("my_col")
        runtime.chroma_client.get_or_create_collection.return_value = new_col

        with patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import create_collection
            from API_RAG_NEW.schemas import CollectionCreateRequest
            req = CollectionCreateRequest(name="my_col", description="test")
            result = create_collection(req, provider="local_sbert")

        assert result.name == "my_col"

    def test_raises_if_collection_exists(self, runtime):
        existing = _make_mock_collection("existing_col")
        runtime.chroma_client.list_collections.return_value = [existing]

        with patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import create_collection
            from API_RAG_NEW.schemas import CollectionCreateRequest
            req = CollectionCreateRequest(name="existing_col")
            with pytest.raises(HTTPException) as exc_info:
                create_collection(req, provider="local_sbert")

        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail


class TestDeleteCollection:
    def test_deletes_existing_collection(self, runtime):
        col = _make_mock_collection("to_delete")
        runtime.chroma_client.get_collection.return_value = col

        with patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import delete_collection
            result = delete_collection("to_delete", provider="local_sbert")

        assert "deleted" in result["detail"].lower()
        runtime.chroma_client.delete_collection.assert_called_once()

    def test_raises_404_for_missing_collection(self, runtime):
        runtime.chroma_client.get_collection.side_effect = Exception("not found")

        with patch(
            "API_RAG_NEW.collection_service._runtime_for_provider",
            return_value=runtime,
        ):
            from API_RAG_NEW.collection_service import delete_collection
            with pytest.raises(HTTPException) as exc_info:
                delete_collection("nonexistent", provider="local_sbert")

        assert exc_info.value.status_code == 404


class TestStorageCollectionName:
    def test_local_returns_cleaned_name(self):
        from API_RAG_NEW._services_shared import storage_collection_name
        result = storage_collection_name("local_sbert", "my_collection")
        assert result == "my_collection"

    def test_gemini_adds_prefix(self):
        from API_RAG_NEW._services_shared import storage_collection_name
        result = storage_collection_name("gemini", "my_collection")
        assert result.startswith("gemini.")

    def test_gemini_truncates_long_names(self):
        from API_RAG_NEW._services_shared import storage_collection_name
        long_name = "a" * 60
        result = storage_collection_name("gemini", long_name)
        assert len(result) <= 63

    def test_invalid_provider_raises(self):
        from API_RAG_NEW._services_shared import storage_collection_name
        with pytest.raises(HTTPException):
            storage_collection_name("unknown_provider", "col")
