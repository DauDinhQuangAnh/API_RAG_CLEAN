from __future__ import annotations

import re
import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from API_RAG_NEW.reranker import rerank_candidate_ids


_INTERNAL_RECORD_ID_KEYS = {"id", "_id"}


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float | None = None
    vector_rank: int = 0


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
    *,
    initial_top_k: int | None = None,
    include_neighbors: bool = False,
    reranker_type: str = "none",
    rerank_llm: Any | None = None,
) -> tuple[list[Any], str]:
    final_n = max(1, int(number_docs_retrieval))
    initial_n = max(final_n, int(initial_top_k or final_n))
    query_embeddings = model.encode([query])
    search_results = collection.query(
        query_embeddings=query_embeddings,
        n_results=initial_n,
        include=["metadatas", "documents", "distances"],
    )
    candidates = _chunks_from_query_results(search_results)
    if include_neighbors:
        candidates = _expand_neighbor_chunks(collection, candidates)

    if not candidates:
        return [[]], ""

    ranked_chunks = _rank_chunks(
        query,
        candidates,
        final_n,
        reranker_type=reranker_type,
        rerank_llm=rerank_llm,
    )
    final_metadatas = [chunk.metadata for chunk in ranked_chunks]
    return [final_metadatas], format_retrieved_data(final_metadatas, columns_to_answer)


def _chunks_from_query_results(search_results: dict[str, Any]) -> list[RetrievedChunk]:
    ids = _first_result_list(search_results.get("ids"))
    metadatas = _first_result_list(search_results.get("metadatas"))
    documents = _first_result_list(search_results.get("documents"))
    distances = _first_result_list(search_results.get("distances"))

    chunks: list[RetrievedChunk] = []
    for index in range(max(len(ids), len(metadatas), len(documents))):
        record_id = _value_at(ids, index, f"candidate_{index + 1}")
        metadata = _value_at(metadatas, index, {})
        document = _value_at(documents, index, "")
        distance = _value_at(distances, index, None)
        chunks.append(
            _to_retrieved_chunk(
                record_id,
                metadata,
                document,
                distance=distance,
                vector_rank=index,
            )
        )
    return _dedupe_chunks(chunks)


def _expand_neighbor_chunks(
    collection: Any,
    candidates: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    expanded: list[RetrievedChunk] = list(candidates)
    records_by_doc_id: dict[str, list[RetrievedChunk]] = {}
    next_rank = len(expanded)

    for candidate in candidates:
        doc_id = candidate.metadata.get("doc_id")
        chunk_index = _safe_int(candidate.metadata.get("chunk_index"))
        if not doc_id or chunk_index is None:
            continue

        doc_id = str(doc_id)
        if doc_id not in records_by_doc_id:
            try:
                records_by_doc_id[doc_id] = _get_records_for_doc_id(collection, doc_id)
            except Exception:
                records_by_doc_id[doc_id] = []

        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            for record in records_by_doc_id[doc_id]:
                if _safe_int(record.metadata.get("chunk_index")) == neighbor_index:
                    expanded.append(
                        RetrievedChunk(
                            id=record.id,
                            document=record.document,
                            metadata=record.metadata,
                            distance=record.distance,
                            vector_rank=next_rank,
                        )
                    )
                    next_rank += 1

    return _dedupe_chunks(expanded)


def _get_records_for_doc_id(collection: Any, doc_id: str) -> list[RetrievedChunk]:
    include = ["metadatas", "documents"]
    try:
        payload = collection.get(where={"doc_id": doc_id}, include=include)
    except Exception:
        payload = collection.get(include=include)

    ids = payload.get("ids") or []
    metadatas = payload.get("metadatas") or []
    documents = payload.get("documents") or []

    chunks: list[RetrievedChunk] = []
    for index in range(max(len(ids), len(metadatas), len(documents))):
        metadata = _value_at(metadatas, index, {})
        if isinstance(metadata, dict) and str(metadata.get("doc_id")) != doc_id:
            continue
        chunks.append(
            _to_retrieved_chunk(
                _value_at(ids, index, f"neighbor_{doc_id}_{index + 1}"),
                metadata,
                _value_at(documents, index, ""),
                vector_rank=index,
            )
        )
    return _dedupe_chunks(chunks)


def _rank_chunks(
    query: str,
    candidates: Sequence[RetrievedChunk],
    final_n: int,
    *,
    reranker_type: str,
    rerank_llm: Any | None,
) -> list[RetrievedChunk]:
    original_order = list(candidates)
    if reranker_type.casefold() == "llm" and rerank_llm is not None:
        ranked_ids = rerank_candidate_ids(query, original_order, final_n, rerank_llm)
    else:
        ranked_ids = []

    chunks_by_id = {chunk.id: chunk for chunk in original_order}
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()
    for record_id in ranked_ids:
        chunk = chunks_by_id.get(record_id)
        if chunk is None or record_id in seen:
            continue
        selected.append(chunk)
        seen.add(record_id)
        if len(selected) >= final_n:
            return selected

    for chunk in original_order:
        if chunk.id in seen:
            continue
        selected.append(chunk)
        seen.add(chunk.id)
        if len(selected) >= final_n:
            break
    return selected


def _to_retrieved_chunk(
    record_id: Any,
    metadata: Any,
    document: Any,
    *,
    distance: Any = None,
    vector_rank: int = 0,
) -> RetrievedChunk:
    normalized_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    document_text = "" if document is None else str(document)
    if document_text:
        normalized_metadata["chunk"] = document_text

    normalized_distance = None
    if isinstance(distance, (int, float)):
        normalized_distance = float(distance)

    return RetrievedChunk(
        id=str(record_id),
        document=document_text or str(normalized_metadata.get("chunk", "")),
        metadata=normalized_metadata,
        distance=normalized_distance,
        vector_rank=vector_rank,
    )


def _dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            continue
        deduped.append(chunk)
        seen.add(chunk.id)
    return deduped


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _value_at(values: Sequence[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
