from __future__ import annotations

from collections.abc import Sequence
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
        client: Any | None = None,
    ):
        self.client = client or genai.Client(api_key=api_key)
        self.model_name = model_name
        self.dimension = int(output_dimensionality)
        self.task = task
        self.batch_size = max(1, int(batch_size))

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
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
        except Exception as exc:
            raise RuntimeError("Gemini embedding request failed.") from exc

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


def encode_documents(
    model: Any,
    texts: Sequence[str],
    titles: Sequence[str] | None = None,
) -> list[list[float]]:
    if hasattr(model, "encode_documents"):
        return model.encode_documents(texts, titles=titles)
    return model.encode(texts)


def encode_queries(model: Any, texts: Sequence[str]) -> list[list[float]]:
    if hasattr(model, "encode_queries"):
        return model.encode_queries(texts)
    return model.encode(texts)


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


def _to_float_vectors(embeddings: Any) -> list[list[float]]:
    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", embedding)
        if isinstance(values, dict):
            values = values.get("values") or values.get("embedding") or []
        vectors.append([float(value) for value in values])
    return vectors
