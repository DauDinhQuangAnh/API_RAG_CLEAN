from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from fastapi import HTTPException

from API_RAG_NEW.config import (
    ALLOWED_ORIGINS,
    CHROMA_DB_PATH,
    CHROMA_DB_PATH_GEMINI,
    CHROMA_DB_PATH_LOCAL,
    EmbeddingRuntime,
    GEMINI_MODEL,
    GEMINI_PROVIDER,
    GEMINI_RERANKER_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    RAG_CHUNKING_PROFILE,
    RAG_EMBEDDING_PROVIDER,
    RAG_EMBEDDING_QUEUE_TIMEOUT_SECONDS,
    RAG_ENABLE_DISTANCE_GUARD,
    RAG_ENABLE_FINAL_ANSWER_FALLBACK,
    RAG_FINAL_TOP_N,
    RAG_GEMINI_EMBEDDING_BATCH_SIZE,
    RAG_GEMINI_EMBEDDING_DIMENSION,
    RAG_GEMINI_EMBEDDING_MAX_RETRIES,
    RAG_GEMINI_EMBEDDING_MODEL,
    RAG_GEMINI_EMBEDDING_RETRY_BASE_SECONDS,
    RAG_GEMINI_EMBEDDING_RETRY_MAX_SECONDS,
    RAG_GEMINI_EMBEDDING_TASK,
    RAG_INCLUDE_NEIGHBORS,
    RAG_INITIAL_TOP_K,
    RAG_INGEST_QUEUE_TIMEOUT_SECONDS,
    RAG_INTERNAL_API_KEY,
    RAG_LLM_QUEUE_TIMEOUT_SECONDS,
    RAG_LOCAL_EMBEDDING_DIMENSION,
    RAG_LOCAL_EMBEDDING_MODEL,
    RAG_MAX_CONCURRENT_EMBEDDING_CALLS,
    RAG_MAX_CONCURRENT_INGESTS,
    RAG_MAX_CONCURRENT_LLM_CALLS,
    RAG_MAX_CONCURRENT_QUERIES,
    RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE,
    RAG_MAX_DISTANCE,
    RAG_MAX_TOTAL_CANDIDATES,
    RAG_QUERY_QUEUE_TIMEOUT_SECONDS,
    RAG_RERANKER_TYPE,
    check_chroma_connectivity,
    get_cached_embedding_runtime,
    get_embedding_runtime,
    get_gemini_api_key,
)
from API_RAG_NEW.concurrency import concurrency_status_payload
from API_RAG_NEW.rag_pipeline import clean_collection_name
from API_RAG_NEW.schemas import CollectionInfo, CollectionRecordsResponse


FINAL_ANSWER_FALLBACK_MESSAGE = (
    "Hệ thống AI đang quá tải hoặc tạm thời không thể tạo câu trả lời. "
    "Vui lòng thử lại sau ít phút."
)
ALLOWED_CHUNKING_PROFILES = {"hybrid", "semantic"}
CHUNKING_PROFILE_ERROR = "Invalid chunking_profile. Allowed values: hybrid, semantic."
GEMINI_STORAGE_PREFIX = f"{GEMINI_PROVIDER}."
NO_CONTEXT_ANSWER_MESSAGE = (
    "Hiện tài liệu chưa cung cấp đủ thông tin để trả lời chính xác câu hỏi này."
)
RESERVED_COLLECTION_PREFIX_ERROR = (
    "Collection name prefix 'gemini.' is reserved for internal storage."
)
STORAGE_NAME_HASH_LENGTH = 12
DEFAULT_COLLECTION_DESCRIPTION = "A collection for RAG system"
logger = logging.getLogger(__name__)


def _normalize_provider(provider: str) -> str:
    normalized = str(provider or LOCAL_EMBEDDING_PROVIDER).strip().casefold()
    if normalized in {LOCAL_EMBEDDING_PROVIDER, GEMINI_PROVIDER}:
        return normalized
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def _runtime_for_provider(provider: str) -> EmbeddingRuntime:
    try:
        return get_embedding_runtime(_normalize_provider(provider))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def storage_collection_name(provider: str, logical_name: str) -> str:
    try:
        normalized_provider = _normalize_provider(provider)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Unsupported embedding provider.",
        ) from exc
    cleaned_name = _clean_logical_collection_name(logical_name)

    if normalized_provider == LOCAL_EMBEDDING_PROVIDER:
        return cleaned_name

    if normalized_provider == GEMINI_PROVIDER:
        candidate = f"{GEMINI_STORAGE_PREFIX}{cleaned_name}"
        if len(candidate) <= 63:
            return candidate

        digest = hashlib.sha256(
            f"{normalized_provider}:{cleaned_name}".encode("utf-8")
        ).hexdigest()[:STORAGE_NAME_HASH_LENGTH]
        max_base_length = (
            63
            - len(GEMINI_STORAGE_PREFIX)
            - 1
            - STORAGE_NAME_HASH_LENGTH
        )
        base = re.sub(r"[^a-zA-Z0-9]+$", "", cleaned_name[:max_base_length])
        base = base or "collection"
        return f"{GEMINI_STORAGE_PREFIX}{base}.{digest}"

    raise HTTPException(status_code=400, detail="Unsupported embedding provider.")


def logical_collection_name(
    provider: str,
    storage_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if isinstance(metadata, dict):
        logical_name = metadata.get("logical_collection_name")
        if isinstance(logical_name, str) and logical_name.strip():
            return logical_name

    try:
        normalized_provider = _normalize_provider(provider)
    except RuntimeError:
        return storage_name
    if (
        normalized_provider == GEMINI_PROVIDER
        and storage_name.startswith(GEMINI_STORAGE_PREFIX)
    ):
        return storage_name[len(GEMINI_STORAGE_PREFIX):]

    return storage_name


def _clean_logical_collection_name(name: str) -> str:
    cleaned_name = clean_collection_name(name)
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Invalid collection name.")
    if cleaned_name.casefold().startswith(GEMINI_STORAGE_PREFIX):
        raise HTTPException(status_code=400, detail=RESERVED_COLLECTION_PREFIX_ERROR)
    return cleaned_name


def _get_collection_or_404(runtime: EmbeddingRuntime, storage_name: str) -> Any:
    try:
        return runtime.chroma_client.get_collection(name=storage_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc


def _to_collection_info(runtime: EmbeddingRuntime, collection: Any) -> CollectionInfo:
    return CollectionInfo(
        name=logical_collection_name(
            runtime.provider,
            collection.name,
            collection.metadata,
        ),
        metadata=collection.metadata,
        count=collection.count(),
    )


def _gemini_configured() -> bool:
    return bool(get_gemini_api_key())


def _runtime_payload(runtime: EmbeddingRuntime) -> dict[str, object]:
    return {
        "provider": runtime.provider,
        "model_name": runtime.model_name,
        "dimension": runtime.dimension,
        "chroma_db_path": runtime.chroma_db_path,
    }


def _local_runtime_payload() -> dict[str, object]:
    runtime = get_cached_embedding_runtime(LOCAL_EMBEDDING_PROVIDER)
    if runtime is not None:
        return _runtime_payload(runtime)
    return {
        "provider": LOCAL_EMBEDDING_PROVIDER,
        "model_name": RAG_LOCAL_EMBEDDING_MODEL,
        "dimension": RAG_LOCAL_EMBEDDING_DIMENSION,
        "chroma_db_path": CHROMA_DB_PATH_LOCAL,
    }


def _gemini_runtime_payload() -> dict[str, object]:
    return {
        "provider": GEMINI_PROVIDER,
        "model_name": RAG_GEMINI_EMBEDDING_MODEL,
        "dimension": RAG_GEMINI_EMBEDDING_DIMENSION,
        "configured": _gemini_configured(),
        "chroma_db_path": CHROMA_DB_PATH_GEMINI,
    }


def resolve_chunking_profile(chunking_profile: str | None = None) -> str:
    selected = (
        str(chunking_profile).strip().casefold()
        if chunking_profile is not None
        else RAG_CHUNKING_PROFILE
    )
    if not selected:
        selected = RAG_CHUNKING_PROFILE
    if selected not in ALLOWED_CHUNKING_PROFILES:
        raise HTTPException(status_code=400, detail=CHUNKING_PROFILE_ERROR)
    return selected


def health_payload() -> dict[str, object]:
    checks: dict[str, object] = {}

    local_ok, local_msg = check_chroma_connectivity(CHROMA_DB_PATH_LOCAL)
    checks["chroma_local"] = local_msg

    if CHROMA_DB_PATH_GEMINI != CHROMA_DB_PATH_LOCAL:
        gemini_ok, gemini_msg = check_chroma_connectivity(CHROMA_DB_PATH_GEMINI)
        checks["chroma_gemini"] = gemini_msg
    else:
        gemini_ok = local_ok

    checks["local_model_loaded"] = (
        get_cached_embedding_runtime(LOCAL_EMBEDDING_PROVIDER) is not None
    )
    checks["gemini_configured"] = _gemini_configured()

    overall_ok = local_ok and gemini_ok
    return {"status": "ok" if overall_ok else "degraded", "checks": checks}


def runtime_config_payload() -> dict[str, object]:
    return {
        "rag_initial_top_k": RAG_INITIAL_TOP_K,
        "rag_final_top_n": RAG_FINAL_TOP_N,
        "rag_chunking_profile": RAG_CHUNKING_PROFILE,
        "available_chunking_profiles": ["hybrid", "semantic"],
        "default_chunking_profile": RAG_CHUNKING_PROFILE,
        "rag_include_neighbors": RAG_INCLUDE_NEIGHBORS,
        "rag_reranker_type": RAG_RERANKER_TYPE,
        "rag_max_context_expansion_per_candidate": (
            RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE
        ),
        "rag_max_total_candidates": RAG_MAX_TOTAL_CANDIDATES,
        "rag_enable_distance_guard": RAG_ENABLE_DISTANCE_GUARD,
        "rag_max_distance": RAG_MAX_DISTANCE,
        "rag_internal_api_key_enabled": bool(RAG_INTERNAL_API_KEY),
        "rag_max_concurrent_queries": RAG_MAX_CONCURRENT_QUERIES,
        "rag_max_concurrent_llm_calls": RAG_MAX_CONCURRENT_LLM_CALLS,
        "rag_max_concurrent_ingests": RAG_MAX_CONCURRENT_INGESTS,
        "rag_max_concurrent_embedding_calls": RAG_MAX_CONCURRENT_EMBEDDING_CALLS,
        "rag_query_queue_timeout_seconds": RAG_QUERY_QUEUE_TIMEOUT_SECONDS,
        "rag_llm_queue_timeout_seconds": RAG_LLM_QUEUE_TIMEOUT_SECONDS,
        "rag_ingest_queue_timeout_seconds": RAG_INGEST_QUEUE_TIMEOUT_SECONDS,
        "rag_embedding_queue_timeout_seconds": RAG_EMBEDDING_QUEUE_TIMEOUT_SECONDS,
        "rag_enable_final_answer_fallback": RAG_ENABLE_FINAL_ANSWER_FALLBACK,
        "concurrency": concurrency_status_payload(),
        "gemini_model": GEMINI_MODEL,
        "gemini_reranker_model": GEMINI_RERANKER_MODEL,
        "embedding_routes": {
            "root": LOCAL_EMBEDDING_PROVIDER,
            "local": LOCAL_EMBEDDING_PROVIDER,
            "gemini": GEMINI_PROVIDER,
        },
        "local_embedding": _local_runtime_payload(),
        "gemini_embedding": _gemini_runtime_payload(),
        "rag_embedding_provider": RAG_EMBEDDING_PROVIDER,
        "rag_local_embedding_model": RAG_LOCAL_EMBEDDING_MODEL,
        "rag_gemini_embedding_model": RAG_GEMINI_EMBEDDING_MODEL,
        "rag_gemini_embedding_dimension": RAG_GEMINI_EMBEDDING_DIMENSION,
        "rag_gemini_embedding_task": RAG_GEMINI_EMBEDDING_TASK,
        "rag_gemini_embedding_batch_size": RAG_GEMINI_EMBEDDING_BATCH_SIZE,
        "rag_gemini_embedding_max_retries": RAG_GEMINI_EMBEDDING_MAX_RETRIES,
        "rag_gemini_embedding_retry_base_seconds": (
            RAG_GEMINI_EMBEDDING_RETRY_BASE_SECONDS
        ),
        "rag_gemini_embedding_retry_max_seconds": (
            RAG_GEMINI_EMBEDDING_RETRY_MAX_SECONDS
        ),
        "chroma_db_path": CHROMA_DB_PATH,
        "chroma_db_path_local": CHROMA_DB_PATH_LOCAL,
        "chroma_db_path_gemini": CHROMA_DB_PATH_GEMINI,
        "cors_origins": ALLOWED_ORIGINS,
    }


def runtime_status_payload() -> dict[str, object]:
    return {
        "health": health_payload(),
        "concurrency": concurrency_status_payload(),
        "available_chunking_profiles": ["hybrid", "semantic"],
        "default_chunking_profile": RAG_CHUNKING_PROFILE,
        "embedding_routes": {
            "root": LOCAL_EMBEDDING_PROVIDER,
            "local": LOCAL_EMBEDDING_PROVIDER,
            "gemini": GEMINI_PROVIDER,
        },
        "local_embedding": _local_runtime_payload(),
        "gemini_embedding": _gemini_runtime_payload(),
        "gemini_model": GEMINI_MODEL,
        "gemini_reranker_model": GEMINI_RERANKER_MODEL,
    }
