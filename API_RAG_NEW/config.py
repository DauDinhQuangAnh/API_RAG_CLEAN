from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import chromadb
from dotenv import load_dotenv

from API_RAG_NEW.embeddings import (
    GeminiTextEmbeddings,
    LocalSentenceTransformerEmbeddings,
)
from download_model import PRIMARY_MODEL_NAME, ensure_embedding_model


load_dotenv()


GEMINI_PROVIDER = "gemini"
LOCAL_EMBEDDING_PROVIDER = "local_sbert"
DEFAULT_COLLECTION_DESCRIPTION = "A collection for RAG system"
INGEST_BATCH_SIZE = 256
SUPPORTED_GEMINI_EMBEDDING_DIMENSIONS = {768, 1536, 3072}
EmbeddingProviderName = Literal["local_sbert", "gemini"]


@dataclass(frozen=True)
class EmbeddingRuntime:
    provider: str
    model_name: str
    dimension: int
    embedding_model: object
    chroma_client: chromadb.PersistentClient
    chroma_db_path: str


def parse_cors_origins(raw_value: str | None) -> list[str]:
    if not raw_value:
        return ["*"]

    origins = [item.strip() for item in raw_value.split(",") if item.strip()]
    return origins or ["*"]


def get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_optional_float_env(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def get_float_env(name: str, default: float) -> float:
    value = get_optional_float_env(name)
    if value is None or value < 0:
        return default
    return value


def get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_choice_env(name: str, default: str, supported_values: set[str]) -> str:
    raw_value = os.getenv(name, default)
    normalized = str(raw_value).strip().casefold()
    return normalized if normalized in supported_values else default


def get_gemini_embedding_dimension_env(name: str, default: int) -> int:
    value = get_int_env(name, default)
    return value if value in SUPPORTED_GEMINI_EMBEDDING_DIMENSIONS else default


ROOT_PATH = os.getenv("ROOT_PATH", "").strip()
ALLOWED_ORIGINS = parse_cors_origins(os.getenv("RAG_CORS_ORIGINS"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")
CHROMA_DB_PATH_LOCAL = os.getenv("CHROMA_DB_PATH_LOCAL", CHROMA_DB_PATH)
CHROMA_DB_PATH_GEMINI = os.getenv("CHROMA_DB_PATH_GEMINI", CHROMA_DB_PATH)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_RERANKER_MODEL = os.getenv("GEMINI_RERANKER_MODEL") or GEMINI_MODEL
RAG_INITIAL_TOP_K = get_int_env("RAG_INITIAL_TOP_K", 20)
RAG_FINAL_TOP_N = get_int_env("RAG_FINAL_TOP_N", 6)
RAG_INCLUDE_NEIGHBORS = get_bool_env("RAG_INCLUDE_NEIGHBORS", True)
RAG_RERANKER_TYPE = os.getenv("RAG_RERANKER_TYPE", "llm")
RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE = max(
    0,
    get_int_env("RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE", 3),
)
RAG_MAX_TOTAL_CANDIDATES = max(1, get_int_env("RAG_MAX_TOTAL_CANDIDATES", 40))
RAG_ENABLE_DISTANCE_GUARD = get_bool_env("RAG_ENABLE_DISTANCE_GUARD", False)
RAG_MAX_DISTANCE = get_optional_float_env("RAG_MAX_DISTANCE")
RAG_INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY") or None
RAG_REQUIRE_INTERNAL_API_KEY = get_bool_env("RAG_REQUIRE_INTERNAL_API_KEY", False)
RAG_MAX_CONCURRENT_QUERIES = max(1, get_int_env("RAG_MAX_CONCURRENT_QUERIES", 32))
RAG_MAX_CONCURRENT_LLM_CALLS = max(1, get_int_env("RAG_MAX_CONCURRENT_LLM_CALLS", 16))
RAG_MAX_CONCURRENT_INGESTS = max(1, get_int_env("RAG_MAX_CONCURRENT_INGESTS", 6))
RAG_MAX_CONCURRENT_EMBEDDING_CALLS = max(
    1,
    get_int_env("RAG_MAX_CONCURRENT_EMBEDDING_CALLS", 12),
)
RAG_QUERY_QUEUE_TIMEOUT_SECONDS = get_float_env(
    "RAG_QUERY_QUEUE_TIMEOUT_SECONDS",
    30.0,
)
RAG_LLM_QUEUE_TIMEOUT_SECONDS = get_float_env(
    "RAG_LLM_QUEUE_TIMEOUT_SECONDS",
    30.0,
)
RAG_INGEST_QUEUE_TIMEOUT_SECONDS = get_float_env(
    "RAG_INGEST_QUEUE_TIMEOUT_SECONDS",
    60.0,
)
RAG_EMBEDDING_QUEUE_TIMEOUT_SECONDS = get_float_env(
    "RAG_EMBEDDING_QUEUE_TIMEOUT_SECONDS",
    60.0,
)
RAG_ENABLE_FINAL_ANSWER_FALLBACK = get_bool_env(
    "RAG_ENABLE_FINAL_ANSWER_FALLBACK",
    True,
)
RAG_CHUNKING_PROFILE = get_choice_env(
    "RAG_CHUNKING_PROFILE",
    "hybrid",
    {"semantic", "hybrid"},
)
RAG_EMBEDDING_PROVIDER = get_choice_env(
    "RAG_EMBEDDING_PROVIDER",
    "local_sbert",
    {"local_sbert", "gemini"},
)
RAG_LOCAL_EMBEDDING_MODEL = os.getenv(
    "RAG_LOCAL_EMBEDDING_MODEL",
    PRIMARY_MODEL_NAME,
)
RAG_LOCAL_EMBEDDING_DIMENSION = max(
    1,
    get_int_env("RAG_LOCAL_EMBEDDING_DIMENSION", 768),
)
RAG_GEMINI_EMBEDDING_MODEL = os.getenv(
    "RAG_GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2",
)
RAG_GEMINI_EMBEDDING_DIMENSION = get_gemini_embedding_dimension_env(
    "RAG_GEMINI_EMBEDDING_DIMENSION",
    768,
)
RAG_GEMINI_EMBEDDING_TASK = (
    os.getenv("RAG_GEMINI_EMBEDDING_TASK", "question answering").strip()
    or "question answering"
)
RAG_GEMINI_EMBEDDING_BATCH_SIZE = max(
    1,
    min(get_int_env("RAG_GEMINI_EMBEDDING_BATCH_SIZE", 32), 128),
)
RAG_GEMINI_EMBEDDING_MAX_RETRIES = max(
    1,
    get_int_env("RAG_GEMINI_EMBEDDING_MAX_RETRIES", 3),
)
RAG_GEMINI_EMBEDDING_RETRY_BASE_SECONDS = get_float_env(
    "RAG_GEMINI_EMBEDDING_RETRY_BASE_SECONDS",
    1.0,
)
RAG_GEMINI_EMBEDDING_RETRY_MAX_SECONDS = get_float_env(
    "RAG_GEMINI_EMBEDDING_RETRY_MAX_SECONDS",
    8.0,
)


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None


def build_local_embedding_provider() -> LocalSentenceTransformerEmbeddings:
    local_model, active_model_name, _ = ensure_embedding_model(
        preferred_model=RAG_LOCAL_EMBEDDING_MODEL
    )
    return LocalSentenceTransformerEmbeddings(local_model, active_model_name)


def build_gemini_embedding_provider() -> GeminiTextEmbeddings:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY must be configured for Gemini embedding routes."
        )
    return GeminiTextEmbeddings(
        api_key=api_key,
        model_name=RAG_GEMINI_EMBEDDING_MODEL,
        output_dimensionality=RAG_GEMINI_EMBEDDING_DIMENSION,
        task=RAG_GEMINI_EMBEDDING_TASK,
        batch_size=RAG_GEMINI_EMBEDDING_BATCH_SIZE,
        max_retries=RAG_GEMINI_EMBEDDING_MAX_RETRIES,
        retry_base_seconds=RAG_GEMINI_EMBEDDING_RETRY_BASE_SECONDS,
        retry_max_seconds=RAG_GEMINI_EMBEDDING_RETRY_MAX_SECONDS,
    )


def build_embedding_provider(provider: EmbeddingProviderName | None = None):
    selected_provider = provider or RAG_EMBEDDING_PROVIDER
    if selected_provider == LOCAL_EMBEDDING_PROVIDER:
        return build_local_embedding_provider()
    if selected_provider == GEMINI_PROVIDER:
        return build_gemini_embedding_provider()
    raise RuntimeError(f"Unsupported embedding provider: {selected_provider}")


_LOCAL_RUNTIME: EmbeddingRuntime | None = None
_GEMINI_RUNTIME: EmbeddingRuntime | None = None
_CHROMA_CLIENTS: dict[str, chromadb.PersistentClient] = {}


def _get_chroma_client(db_path: str) -> chromadb.PersistentClient:
    if db_path not in _CHROMA_CLIENTS:
        _CHROMA_CLIENTS[db_path] = chromadb.PersistentClient(db_path)
    return _CHROMA_CLIENTS[db_path]


def get_embedding_runtime(provider: EmbeddingProviderName) -> EmbeddingRuntime:
    global _LOCAL_RUNTIME, _GEMINI_RUNTIME

    if provider == LOCAL_EMBEDDING_PROVIDER:
        if _LOCAL_RUNTIME is None:
            embedding_model = build_local_embedding_provider()
            _LOCAL_RUNTIME = EmbeddingRuntime(
                provider=embedding_model.provider,
                model_name=embedding_model.model_name,
                dimension=int(embedding_model.dimension),
                embedding_model=embedding_model,
                chroma_client=_get_chroma_client(CHROMA_DB_PATH_LOCAL),
                chroma_db_path=CHROMA_DB_PATH_LOCAL,
            )
        return _LOCAL_RUNTIME

    if provider == GEMINI_PROVIDER:
        if _GEMINI_RUNTIME is None:
            embedding_model = build_gemini_embedding_provider()
            _GEMINI_RUNTIME = EmbeddingRuntime(
                provider=embedding_model.provider,
                model_name=embedding_model.model_name,
                dimension=int(embedding_model.dimension),
                embedding_model=embedding_model,
                chroma_client=_get_chroma_client(CHROMA_DB_PATH_GEMINI),
                chroma_db_path=CHROMA_DB_PATH_GEMINI,
            )
        return _GEMINI_RUNTIME

    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def get_cached_embedding_runtime(
    provider: EmbeddingProviderName,
) -> EmbeddingRuntime | None:
    if provider == LOCAL_EMBEDDING_PROVIDER:
        return _LOCAL_RUNTIME
    if provider == GEMINI_PROVIDER:
        return _GEMINI_RUNTIME
    raise RuntimeError(f"Unsupported embedding provider: {provider}")
