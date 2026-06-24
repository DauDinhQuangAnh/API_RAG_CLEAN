"""Tests cho EmbeddingCache (LRU cache)."""
from __future__ import annotations

import pytest

from API_RAG_NEW.embeddings import EmbeddingCache, encode_documents, encode_queries


class TestEmbeddingCacheBasic:
    def test_disabled_when_maxsize_zero(self):
        cache = EmbeddingCache(maxsize=0)
        assert not cache.enabled

    def test_enabled_when_maxsize_positive(self):
        cache = EmbeddingCache(maxsize=10)
        assert cache.enabled

    def test_miss_returns_none(self):
        cache = EmbeddingCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_put_and_get(self):
        cache = EmbeddingCache(maxsize=10)
        cache.put("k1", [1.0, 2.0])
        assert cache.get("k1") == [1.0, 2.0]

    def test_lru_eviction(self):
        cache = EmbeddingCache(maxsize=2)
        cache.put("k1", [1.0])
        cache.put("k2", [2.0])
        cache.put("k3", [3.0])  # evicts k1 (oldest)
        assert cache.get("k1") is None
        assert cache.get("k2") == [2.0]
        assert cache.get("k3") == [3.0]

    def test_access_refreshes_lru(self):
        cache = EmbeddingCache(maxsize=2)
        cache.put("k1", [1.0])
        cache.put("k2", [2.0])
        cache.get("k1")          # k1 now most recently used
        cache.put("k3", [3.0])   # evicts k2 (now oldest)
        assert cache.get("k1") == [1.0]
        assert cache.get("k2") is None
        assert cache.get("k3") == [3.0]

    def test_disabled_cache_ignores_put_and_get(self):
        cache = EmbeddingCache(maxsize=0)
        cache.put("k1", [1.0])
        assert cache.get("k1") is None

    def test_make_key_is_deterministic(self):
        key1 = EmbeddingCache.make_key("model", "prefix", "text")
        key2 = EmbeddingCache.make_key("model", "prefix", "text")
        assert key1 == key2

    def test_make_key_differs_on_different_inputs(self):
        k1 = EmbeddingCache.make_key("model", "query", "xin chào")
        k2 = EmbeddingCache.make_key("model", "query", "tạm biệt")
        assert k1 != k2

    def test_thread_safety(self):
        import threading
        cache = EmbeddingCache(maxsize=50)
        errors = []

        def worker(idx: int) -> None:
            try:
                for i in range(10):
                    key = f"k{idx}_{i}"
                    cache.put(key, [float(idx + i)])
                    cache.get(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestEncodeFunctions:
    def test_encode_documents_no_cache(self, mock_embedding_model):
        mock_embedding_model.encode_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        from unittest.mock import patch
        with patch("API_RAG_NEW.embeddings.get_embedding_cache") as mock_get:
            from API_RAG_NEW.embeddings import EmbeddingCache
            mock_get.return_value = EmbeddingCache(maxsize=0)
            result = encode_documents(mock_embedding_model, ["a", "b"])
        assert len(result) == 2

    def test_encode_queries_no_cache(self, mock_embedding_model):
        mock_embedding_model.encode_queries.return_value = [[0.5, 0.6]]
        from unittest.mock import patch
        with patch("API_RAG_NEW.embeddings.get_embedding_cache") as mock_get:
            from API_RAG_NEW.embeddings import EmbeddingCache
            mock_get.return_value = EmbeddingCache(maxsize=0)
            result = encode_queries(mock_embedding_model, ["câu hỏi"])
        assert len(result) == 1

    def test_encode_documents_cache_hit(self, mock_embedding_model):
        from unittest.mock import patch
        cache = EmbeddingCache(maxsize=100)
        with patch("API_RAG_NEW.embeddings.get_embedding_cache", return_value=cache):
            mock_embedding_model.encode_documents.return_value = [[0.1, 0.2]]
            # First call — cache miss
            encode_documents(mock_embedding_model, ["xin chào"])
            first_call_count = mock_embedding_model.encode_documents.call_count
            # Second call — cache hit
            encode_documents(mock_embedding_model, ["xin chào"])
            assert mock_embedding_model.encode_documents.call_count == first_call_count

    def test_empty_input_returns_empty(self, mock_embedding_model):
        result = encode_documents(mock_embedding_model, [])
        assert result == []
        result = encode_queries(mock_embedding_model, [])
        assert result == []
