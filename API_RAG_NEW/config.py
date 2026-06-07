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


ROOT_PATH = os.getenv("ROOT_PATH", "").strip()
ALLOWED_ORIGINS = parse_cors_origins(os.getenv("RAG_CORS_ORIGINS"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CHROMA_CLIENT = chromadb.PersistentClient(CHROMA_DB_PATH)
EMBEDDING_MODEL, ACTIVE_EMBEDDING_MODEL_NAME, _ = ensure_embedding_model(
    preferred_model=DEFAULT_EMBEDDING_MODEL_NAME
)


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None
