from __future__ import annotations

import collections
import hashlib
import threading
from collections.abc import Sequence
import time
from typing import Any, Protocol

from google import genai
from google.genai import types


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


class GeminiTextEmbeddings:
    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        output_dimensionality: int,
        task: str,
        batch_size: int,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 8.0,
        client: Any | None = None,
    ):
        self.client = client or genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dimension = int(output_dimensionality)
        self.task = task
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.encode_documents(texts)

    def encode_documents(
        self,
        texts: Sequence[str],
        titles: Sequence[str] | None = None,
    ) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        title_list = _normalize_titles(text_list, titles)
        formatted = [
            f"title: {title} | text: {text}"
            for title, text in zip(title_list, text_list)
        ]
        return self._embed_texts(formatted)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [
            f"task: {self.task} | query: {text}"
            for text in (str(text) for text in texts)
        ]
        return self._embed_texts(formatted)

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            for text in batch:
                vectors.append(self._embed_one(text))

        if len(vectors) != len(text_list):
            raise RuntimeError("Gemini embedding returned an invalid vector count.")
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        response = self._embed_one_with_retry(text)

        embeddings = _extract_gemini_embeddings(response)
        vectors = _to_float_vectors(embeddings)
        if not vectors:
            raise RuntimeError("Gemini embedding returned no vectors.")
        if len(vectors) != 1:
            raise RuntimeError("Gemini embedding returned an invalid vector count.")
        vector = vectors[0]
        if not vector:
            raise RuntimeError("Gemini embedding returned an empty vector.")
        return vector

    def _embed_one_with_retry(self, text: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension
                    ),
                )
            except Exception as exc:
                last_error = exc
                if not _is_transient_gemini_embedding_error(exc):
                    raise RuntimeError("Gemini embedding request failed.") from exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self._retry_delay(attempt))

        raise RuntimeError(
            f"Gemini embedding request failed after {self.max_retries} attempts."
        ) from last_error

    def _retry_delay(self, attempt: int) -> float:
        delay = self.retry_base_seconds * (2 ** max(0, attempt - 1))
        return min(delay, self.retry_max_seconds)


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


def _extract_gemini_embeddings(response: Any) -> Any:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is not None:
        return embeddings

    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        return [embedding]

    if isinstance(response, dict):
        if "embeddings" in response:
            return response["embeddings"]
        if "embedding" in response:
            return [response["embedding"]]

    raise RuntimeError("Gemini embedding response did not include embeddings.")


def _is_transient_gemini_embedding_error(exc: Exception) -> bool:
    status_code = _error_status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    if status_code in {400, 401, 403, 404}:
        return False

    text = f"{type(exc).__name__} {exc}".casefold()
    permanent_markers = (
        "api key",
        "apikey",
        "unauthorized",
        "permission denied",
        "permission_denied",
        "invalid argument",
        "invalid_argument",
        "invalid request",
        "bad request",
        "not found",
        "invalid model",
        "model not found",
    )
    if any(marker in text for marker in permanent_markers):
        return False

    transient_markers = (
        "rate limit",
        "ratelimit",
        "resource exhausted",
        "resource_exhausted",
        "quota",
        "temporarily exhausted",
        "timeout",
        "timed out",
        "deadline",
        "deadline_exceeded",
        "unavailable",
        "internal server",
        "server error",
        "connection reset",
        "connection aborted",
        "connection error",
        "reset by peer",
        "temporarily unavailable",
        "503",
        "500",
        "502",
        "504",
        "429",
    )
    return any(marker in text for marker in transient_markers)


def _error_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        return None
    return None


def _to_float_vectors(embeddings: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", embedding)
        if isinstance(values, dict):
            values = values.get("values") or values.get("embedding") or []
        vectors.append([float(value) for value in values])
    return vectors
