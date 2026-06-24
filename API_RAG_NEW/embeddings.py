from __future__ import annotations

import collections
import hashlib
import threading
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingProviderProtocol(Protocol):
    provider: str
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class LocalSentenceTransformerEmbeddings:
    provider = "local_sbert"

    def __init__(self, model: Any, model_name: str):
        self.model = model
        self.model_name = model_name
        self.dimension = _detect_local_dimension(model)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts))
        return _to_float_vectors(embeddings)


class EmbeddingCache:
    """Thread-safe LRU cache cho embedding vectors."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = max(0, int(maxsize))
        self._cache: collections.OrderedDict[str, list[float]] = collections.OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._maxsize > 0

    def get(self, key: str) -> list[float] | None:
        if not self._maxsize:
            return None
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, vector: list[float]) -> None:
        if not self._maxsize:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = vector
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @staticmethod
    def make_key(model_name: str, prefix: str, text: str) -> str:
        return hashlib.sha256(
            f"{model_name}|{prefix}|{text}".encode("utf-8")
        ).hexdigest()[:32]


_EMBEDDING_CACHE: EmbeddingCache | None = None
_CACHE_LOCK = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is None:
        with _CACHE_LOCK:
            if _EMBEDDING_CACHE is None:
                from API_RAG_NEW.config import RAG_EMBEDDING_CACHE_SIZE
                _EMBEDDING_CACHE = EmbeddingCache(maxsize=RAG_EMBEDDING_CACHE_SIZE)
    return _EMBEDDING_CACHE


def encode_documents(
    model: Any,
    texts: Sequence[str],
    titles: Sequence[str] | None = None,
) -> list[list[float]]:
    text_list = list(texts)
    if not text_list:
        return []

    cache = get_embedding_cache()
    if not cache.enabled:
        if hasattr(model, "encode_documents"):
            return model.encode_documents(text_list, titles=titles)
        return model.encode(text_list)

    model_name = getattr(model, "model_name", "")
    title_list = _normalize_titles(text_list, titles)
    result: list[list[float] | None] = [None] * len(text_list)
    uncached_texts: list[str] = []
    uncached_titles: list[str] = []
    uncached_indices: list[int] = []

    for i, (text, title) in enumerate(zip(text_list, title_list)):
        key = cache.make_key(model_name, f"doc|{title}", text)
        vector = cache.get(key)
        if vector is not None:
            result[i] = vector
        else:
            uncached_texts.append(text)
            uncached_titles.append(title)
            uncached_indices.append(i)

    if uncached_texts:
        if hasattr(model, "encode_documents"):
            new_vectors = model.encode_documents(uncached_texts, titles=uncached_titles)
        else:
            new_vectors = model.encode(uncached_texts)
        for idx, (text, title, vector) in enumerate(
            zip(uncached_texts, uncached_titles, new_vectors)
        ):
            key = cache.make_key(model_name, f"doc|{title}", text)
            cache.put(key, vector)
            result[uncached_indices[idx]] = vector

    return [v for v in result if v is not None]


def encode_queries(model: Any, texts: Sequence[str]) -> list[list[float]]:
    text_list = list(texts)
    if not text_list:
        return []

    cache = get_embedding_cache()
    if not cache.enabled:
        if hasattr(model, "encode_queries"):
            return model.encode_queries(text_list)
        return model.encode(text_list)

    model_name = getattr(model, "model_name", "")
    result: list[list[float] | None] = [None] * len(text_list)
    uncached_texts: list[str] = []
    uncached_indices: list[int] = []

    for i, text in enumerate(text_list):
        key = cache.make_key(model_name, "query", text)
        vector = cache.get(key)
        if vector is not None:
            result[i] = vector
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    if uncached_texts:
        if hasattr(model, "encode_queries"):
            new_vectors = model.encode_queries(uncached_texts)
        else:
            new_vectors = model.encode(uncached_texts)
        for idx, (text, vector) in enumerate(zip(uncached_texts, new_vectors)):
            key = cache.make_key(model_name, "query", text)
            cache.put(key, vector)
            result[uncached_indices[idx]] = vector

    return [v for v in result if v is not None]


def _detect_local_dimension(model: Any) -> int:
    for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        dimension_getter = getattr(model, method_name, None)
        if callable(dimension_getter):
            dimension = dimension_getter()
            if dimension:
                return int(dimension)

    embeddings = model.encode(["dimension probe"])
    vectors = _to_float_vectors(embeddings)
    if not vectors or not vectors[0]:
        raise RuntimeError("Unable to detect local embedding dimension.")
    return len(vectors[0])


def _normalize_titles(
    texts: Sequence[str],
    titles: Sequence[str] | None,
) -> list[str]:
    if titles is None:
        return ["none" for _ in texts]

    normalized = [str(title).strip() or "none" for title in titles]
    if len(normalized) < len(texts):
        normalized.extend("none" for _ in range(len(texts) - len(normalized)))
    return normalized[: len(texts)]


def _to_float_vectors(embeddings: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", embedding)
        if isinstance(values, dict):
            values = values.get("values") or values.get("embedding") or []
        vectors.append([float(value) for value in values])
    return vectors
