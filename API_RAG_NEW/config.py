from __future__ import annotations

import os

import chromadb
from dotenv import load_dotenv

from download_model import PRIMARY_MODEL_NAME, ensure_embedding_model


load_dotenv()


GEMINI_PROVIDER = "gemini"
DEFAULT_COLLECTION_DESCRIPTION = "A collection for RAG system"
INGEST_BATCH_SIZE = 256
DEFAULT_EMBEDDING_MODEL_NAME = PRIMARY_MODEL_NAME


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


ROOT_PATH = os.getenv("ROOT_PATH", "").strip()
ALLOWED_ORIGINS = parse_cors_origins(os.getenv("RAG_CORS_ORIGINS"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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
RAG_CHUNKING_PROFILE = get_choice_env(
    "RAG_CHUNKING_PROFILE",
    "hybrid",
    {"semantic", "hybrid"},
)

CHROMA_CLIENT = chromadb.PersistentClient(CHROMA_DB_PATH)
EMBEDDING_MODEL, ACTIVE_EMBEDDING_MODEL_NAME, _ = ensure_embedding_model(
    preferred_model=DEFAULT_EMBEDDING_MODEL_NAME
)


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None
