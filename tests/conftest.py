"""Shared fixtures cho toàn bộ test suite."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Mock Chroma collection ────────────────────────────────────────────────────

def _make_mock_collection(name: str = "test_col") -> MagicMock:
    col = MagicMock()
    col.name = name
    col.metadata = {
        "embedding_provider": "local_sbert",
        "embedding_model": "keepitreal/vietnamese-sbert",
        "embedding_dimension": 768,
        "logical_collection_name": name,
        "storage_collection_name": name,
    }
    col.count.return_value = 0
    col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    col.query.return_value = {
        "ids": [[]],
        "metadatas": [[]],
        "documents": [[]],
        "distances": [[]],
    }
    return col


def _make_mock_chroma_client(collection: MagicMock | None = None) -> MagicMock:
    col = collection or _make_mock_collection()
    client = MagicMock()
    client.list_collections.return_value = [col]
    client.get_collection.return_value = col
    client.get_or_create_collection.return_value = col
    client.heartbeat.return_value = True
    return client


def _make_mock_embedding_model(
    model_name: str = "keepitreal/vietnamese-sbert",
    dimension: int = 768,
) -> MagicMock:
    model = MagicMock()
    model.model_name = model_name
    model.provider = "local_sbert"
    model.dimension = dimension
    model.encode.return_value = [[0.1] * dimension]
    model.encode_documents.return_value = [[0.1] * dimension]
    model.encode_queries.return_value = [[0.1] * dimension]
    return model


def _make_embedding_runtime(
    model: MagicMock | None = None,
    client: MagicMock | None = None,
) -> Any:
    # EmbeddingRuntime là frozen dataclass nên dùng object.__setattr__ để gán field
    # vào một instance giả lập thay vì dùng constructor trực tiếp (vì chroma_client
    # phải là PersistentClient nhưng ta cần mock).
    runtime = MagicMock()
    runtime.provider = "local_sbert"
    runtime.model_name = "keepitreal/vietnamese-sbert"
    runtime.dimension = 768
    runtime.embedding_model = model or _make_mock_embedding_model()
    runtime.chroma_client = client or _make_mock_chroma_client()
    runtime.chroma_db_path = "db"
    return runtime


@pytest.fixture
def mock_collection():
    return _make_mock_collection()


@pytest.fixture
def mock_chroma_client(mock_collection):
    return _make_mock_chroma_client(mock_collection)


@pytest.fixture
def mock_embedding_model():
    return _make_mock_embedding_model()


@pytest.fixture
def embedding_runtime(mock_embedding_model, mock_chroma_client):
    return _make_embedding_runtime(mock_embedding_model, mock_chroma_client)
