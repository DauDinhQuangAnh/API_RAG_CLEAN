from __future__ import annotations

import re
import hashlib
import math
from datetime import date, datetime
from typing import Any, Sequence


_INTERNAL_RECORD_ID_KEYS = {"id", "_id"}


def stable_record_id(*parts: Any, prefix: str = "chunk", length: int = 32) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return f"{prefix}_{hasher.hexdigest()[:length]}"


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in _INTERNAL_RECORD_ID_KEYS or value is None:
            continue

        sanitized_key = str(key)
        if isinstance(value, bool):
            sanitized[sanitized_key] = value
        elif isinstance(value, int):
            sanitized[sanitized_key] = value
        elif isinstance(value, float):
            sanitized[sanitized_key] = value if math.isfinite(value) else str(value)
        elif isinstance(value, str):
            sanitized[sanitized_key] = value
        elif isinstance(value, (date, datetime)):
            sanitized[sanitized_key] = value.isoformat()
        else:
            sanitized[sanitized_key] = str(value)

    return sanitized


def add_records_to_collection(
    records: Sequence[dict[str, Any]], model: Any, collection: Any
) -> int:
    if not records:
        return 0

    try:
        documents = [str(record["chunk"]) for record in records]
        embeddings = model.encode(documents)
        collection.upsert(
            ids=[_resolve_record_id(record) for record in records],
            embeddings=embeddings,
            documents=documents,
            metadatas=[sanitize_metadata(record) for record in records],
        )
    except AttributeError as exc:
        if "encode" in str(exc):
            raise RuntimeError(
                "Please configure the embedding model before ingesting data."
            ) from exc
        raise RuntimeError(f"Error saving data to Chroma: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error saving data to Chroma: {exc}") from exc

    return len(records)


def _resolve_record_id(record: dict[str, Any]) -> str:
    record_id = record.get("id") or record.get("_id")
    if record_id:
        return str(record_id)
    return stable_record_id(
        record.get("doc_id", ""),
        record.get("source", ""),
        record.get("source_type", ""),
        record.get("chunk_index", ""),
        record.get("chunk", ""),
    )


def clean_collection_name(name: str) -> str | None:
    cleaned_name = re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    cleaned_name = re.sub(r"\.{2,}", ".", cleaned_name)
    cleaned_name = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", cleaned_name)
    return cleaned_name[:63] if 3 <= len(cleaned_name) <= 63 else None


def format_retrieved_data(
    metadatas: list[dict[str, Any]], columns_to_answer: Sequence[str]
) -> str:
    lines: list[str] = []
    available_columns: set[str] = set()
    normalized_metadatas: list[dict[str, Any]] = []

    for metadata in metadatas:
        normalized_metadata = {}
        if isinstance(metadata, dict):
            normalized_metadata = {
                str(key).casefold(): value for key, value in metadata.items()
            }
            available_columns.update(normalized_metadata.keys())
        normalized_metadatas.append(normalized_metadata)

    if columns_to_answer:
        selected_columns = list(columns_to_answer)
    else:
        default_columns = ["chunk", "source", "source_type", "chunk_index"]
        selected_columns = [
            column for column in default_columns if column.casefold() in available_columns
        ]
        if not selected_columns:
            selected_columns = sorted(available_columns)

    normalized_columns = [
        (str(column), str(column).casefold()) for column in selected_columns
    ]
    missing_columns = [
        column
        for column, normalized_column in normalized_columns
        if normalized_column not in available_columns
    ]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(
            f"Requested columns not found in collection metadata: {missing_list}"
        )

    for index, normalized_metadata in enumerate(normalized_metadatas, start=1):
        columns = [
            f"{column}: {normalized_metadata.get(normalized_column)}"
            for column, normalized_column in normalized_columns
            if normalized_column in normalized_metadata
        ]
        line = f"{index}) {' '.join(columns)}".rstrip()
        lines.append(line)

    return "\n".join(lines)


def vector_search(
    model: Any,
    query: str,
    collection: Any,
    columns_to_answer: Sequence[str],
    number_docs_retrieval: int,
) -> tuple[list[Any], str]:
    query_embeddings = model.encode([query])
    search_results = collection.query(
        query_embeddings=query_embeddings,
        n_results=number_docs_retrieval,
    )
    metadatas = search_results["metadatas"]
    first_result = metadatas[0] if metadatas else []
    if not first_result:
        return metadatas, ""
    return metadatas, format_retrieved_data(first_result, columns_to_answer)
