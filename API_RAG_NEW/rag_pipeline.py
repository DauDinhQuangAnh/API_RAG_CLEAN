from __future__ import annotations

import re
import uuid
from typing import Any, Sequence


def add_records_to_collection(
    records: Sequence[dict[str, Any]], model: Any, collection: Any
) -> int:
    if not records:
        return 0

    try:
        embeddings = model.encode([record["chunk"] for record in records])
        collection.add(
            ids=[str(uuid.uuid4()) for _ in records],
            embeddings=embeddings,
            metadatas=[dict(record) for record in records],
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


def clean_collection_name(name: str) -> str | None:
    cleaned_name = re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    cleaned_name = re.sub(r"\.{2,}", ".", cleaned_name)
    cleaned_name = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", cleaned_name)
    return cleaned_name[:63] if 3 <= len(cleaned_name) <= 63 else None


def format_retrieved_data(
    metadatas: list[dict[str, Any]], columns_to_answer: Sequence[str]
) -> str:
    lines: list[str] = []
    normalized_columns = [
        (str(column), str(column).casefold()) for column in columns_to_answer
    ]
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
