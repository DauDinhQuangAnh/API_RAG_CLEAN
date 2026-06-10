from __future__ import annotations

import re
import hashlib
import math
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from API_RAG_NEW.embeddings import encode_documents, encode_queries
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
        titles = [_record_embedding_title(record) for record in records]
        embeddings = encode_documents(model, documents, titles=titles)
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


def _record_embedding_title(record: dict[str, Any]) -> str:
    for key in ("section_title", "section_path", "table_title", "source"):
        value = record.get(key)
        if value:
            return str(value)
    return "none"


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


def format_retrieved_data_with_markers(metadatas: list[dict[str, Any]]) -> str:
    normalized_metadatas = _normalize_metadatas_for_display(metadatas)

    blocks: list[str] = []
    for index, normalized_metadata in enumerate(normalized_metadatas, start=1):
        header_parts = [
            _location_part(normalized_metadata),
            f"chunk_index={_display_value(normalized_metadata.get('chunk_index'))}",
        ]
        section = normalized_metadata.get("section_path") or normalized_metadata.get(
            "section_title"
        )
        if section:
            header_parts.append(f"section={_display_value(section)}")
        chunk_type = normalized_metadata.get("chunk_type")
        if chunk_type:
            header_parts.append(f"chunk_type={_display_value(chunk_type)}")
        table_title = normalized_metadata.get("table_title")
        if table_title:
            header_parts.append(f"table_title={_display_value(table_title)}")
        table_row_index = normalized_metadata.get("table_row_index")
        if table_row_index is not None:
            header_parts.append(f"table_row={_display_value(table_row_index)}")
        table_row_part_index = normalized_metadata.get("table_row_part_index")
        if table_row_part_index is not None:
            header_parts.append(
                f"table_row_part={_display_value(table_row_part_index)}"
            )
        header = (
            f"[{index}] "
            f"source={_display_value(normalized_metadata.get('source'))}"
        )
        body_lines = [
            f"chunk: {_display_value(normalized_metadata.get('chunk'))}",
        ]
        blocks.append(
            f"{header} | {' | '.join(header_parts)}\n" + "\n".join(body_lines)
        )

    return "\n\n".join(blocks)


def vector_search(
    model: Any,
    query: str,
    collection: Any,
    number_docs_retrieval: int,
    *,
    initial_top_k: int | None = None,
    include_neighbors: bool = False,
    reranker_type: str = "none",
    rerank_llm: Any | None = None,
    max_context_expansion_per_candidate: int = 3,
    max_total_candidates: int = 40,
    enable_distance_guard: bool = False,
    max_distance: float | None = None,
) -> tuple[list[Any], str]:
    final_n = max(1, int(number_docs_retrieval))
    initial_n = max(final_n, int(initial_top_k or final_n))
    query_hints = detect_query_hints(query)
    query_embeddings = encode_queries(model, [query])
    search_results = collection.query(
        query_embeddings=query_embeddings,
        n_results=initial_n,
        include=["metadatas", "documents", "distances"],
    )
    candidates = _chunks_from_query_results(search_results)
    if _distance_guard_blocks(candidates, enable_distance_guard, max_distance):
        return [[]], ""

    initial_candidate_ids = [chunk.id for chunk in candidates]
    if include_neighbors:
        candidates = _expand_context_chunks(
            collection,
            candidates,
            max_context_expansion_per_candidate=max_context_expansion_per_candidate,
        )
    candidates = _cap_candidates(
        candidates,
        max_total_candidates=max_total_candidates,
        initial_candidate_ids=initial_candidate_ids,
        query_hints=query_hints,
    )

    if not candidates:
        return [[]], ""

    ranked_chunks = _rank_chunks(
        query,
        candidates,
        final_n,
        reranker_type=reranker_type,
        rerank_llm=rerank_llm,
        query_hints=query_hints,
    )
    final_metadatas = [chunk.metadata for chunk in ranked_chunks]
    return [final_metadatas], format_retrieved_data_with_markers(final_metadatas)


def detect_query_hints(query: str) -> dict[str, bool]:
    normalized = _normalize_query_text(query)
    return {
        "asks_table_or_structured_info": _contains_any(
            normalized,
            (
                "table",
                "row",
                "column",
                "bang",
                "dong",
                "cot",
                "tieu chi",
                "so sanh",
                "criteria",
                "comparison",
                "field",
                "structured",
            ),
        ),
        "asks_number_or_money": _contains_any(
            normalized,
            (
                "gia",
                "phi",
                "chi phi",
                "doanh thu",
                "revenue",
                "cost",
                "price",
                "fee",
                "vnd",
                "dong",
                "usd",
                "%",
            ),
        ),
        "asks_person_or_entity": _contains_any(
            normalized,
            (
                "ai",
                "nguoi nao",
                "ten",
                "thanh vien",
                "truong nhom",
                "founder",
                "co founder",
                "leader",
                "member",
                "person",
            ),
        ),
        "asks_source_or_page": _contains_any(
            normalized,
            (
                "trang",
                "nguon",
                "source",
                "page",
                "citation",
                "tai lieu",
            ),
        ),
    }


def _normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    without_marks = without_marks.casefold()
    without_marks = without_marks.replace("\u0111", "d").replace("\u0110", "d")
    return re.sub(r"\s+", " ", without_marks).strip()


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_metadatas_for_display(
    metadatas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_metadatas: list[dict[str, Any]] = []
    for metadata in metadatas:
        normalized_metadata = {}
        if isinstance(metadata, dict):
            normalized_metadata = {
                str(key).casefold(): value for key, value in metadata.items()
            }
        normalized_metadatas.append(normalized_metadata)
    return normalized_metadatas


def _location_part(normalized_metadata: dict[str, Any]) -> str:
    row_index = normalized_metadata.get("row_index")
    if row_index is not None:
        return f"row={_display_value(row_index)}"

    page_number = normalized_metadata.get("page_number")
    if page_number is not None:
        return f"page={_display_value(page_number)}"
    return "page=N/A"


def _display_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


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
    return _expand_context_chunks(
        collection,
        candidates,
        max_context_expansion_per_candidate=0,
    )


def _expand_context_chunks(
    collection: Any,
    candidates: Sequence[RetrievedChunk],
    *,
    max_context_expansion_per_candidate: int,
) -> list[RetrievedChunk]:
    expanded: list[RetrievedChunk] = list(candidates)
    records_by_doc_id: dict[str, list[RetrievedChunk]] = {}
    next_rank = len(expanded)
    per_candidate_limit = max(0, int(max_context_expansion_per_candidate))

    for candidate in candidates:
        doc_id = candidate.metadata.get("doc_id")
        if not doc_id:
            continue

        doc_id = str(doc_id)
        if doc_id not in records_by_doc_id:
            try:
                records_by_doc_id[doc_id] = _get_records_for_doc_id(collection, doc_id)
            except Exception:
                records_by_doc_id[doc_id] = []

    for candidate in candidates:
        doc_records = _doc_records_for_candidate(candidate, records_by_doc_id)
        chunk_index = _safe_int(candidate.metadata.get("chunk_index"))
        if chunk_index is None:
            continue
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            for record in doc_records:
                if _safe_int(record.metadata.get("chunk_index")) == neighbor_index:
                    expanded.append(_with_vector_rank(record, next_rank))
                    next_rank += 1

    for candidate in candidates:
        doc_records = _doc_records_for_candidate(candidate, records_by_doc_id)
        parent_id = candidate.metadata.get("parent_id")
        if not parent_id:
            continue
        for record in _limited_matches(
            doc_records,
            lambda record: (
                record.id != candidate.id
                and record.metadata.get("parent_id") == parent_id
            ),
            per_candidate_limit,
        ):
            expanded.append(_with_vector_rank(record, next_rank))
            next_rank += 1

    for candidate in candidates:
        if candidate.metadata.get("chunk_type") != "table_row":
            continue
        doc_records = _doc_records_for_candidate(candidate, records_by_doc_id)
        table_index = candidate.metadata.get("table_index")
        page_number = candidate.metadata.get("page_number")
        if table_index is None or page_number is None:
            continue
        for record in _limited_matches(
            doc_records,
            lambda record: (
                record.id != candidate.id
                and record.metadata.get("chunk_type") == "table_row"
                and record.metadata.get("table_index") == table_index
                and record.metadata.get("page_number") == page_number
            ),
            per_candidate_limit,
        ):
            expanded.append(_with_vector_rank(record, next_rank))
            next_rank += 1

    for candidate in candidates:
        doc_records = _doc_records_for_candidate(candidate, records_by_doc_id)
        section_path = candidate.metadata.get("section_path")
        if not section_path:
            continue
        for record in _limited_matches(
            doc_records,
            lambda record: (
                record.id != candidate.id
                and record.metadata.get("section_path") == section_path
            ),
            per_candidate_limit,
        ):
            expanded.append(_with_vector_rank(record, next_rank))
            next_rank += 1

    return _dedupe_chunks(expanded)


def _doc_records_for_candidate(
    candidate: RetrievedChunk,
    records_by_doc_id: dict[str, list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    doc_id = candidate.metadata.get("doc_id")
    if not doc_id:
        return []
    return records_by_doc_id.get(str(doc_id), [])


def _limited_matches(
    records: Sequence[RetrievedChunk],
    predicate: Any,
    limit: int,
) -> list[RetrievedChunk]:
    if limit <= 0:
        return []
    matches: list[RetrievedChunk] = []
    for record in _records_in_doc_order(records):
        if predicate(record):
            matches.append(record)
        if len(matches) >= limit:
            break
    return matches


def _records_in_doc_order(records: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(
        records,
        key=lambda record: (
            _safe_int(record.metadata.get("chunk_index")) is None,
            _safe_int(record.metadata.get("chunk_index")) or 0,
            _safe_int(record.metadata.get("page_chunk_index")) or 0,
            _safe_int(record.metadata.get("row_chunk_index")) or 0,
            record.id,
        ),
    )


def _with_vector_rank(chunk: RetrievedChunk, vector_rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk.id,
        document=chunk.document,
        metadata=chunk.metadata,
        distance=chunk.distance,
        vector_rank=vector_rank,
    )


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


def _distance_guard_blocks(
    candidates: Sequence[RetrievedChunk],
    enabled: bool,
    max_distance: float | None,
) -> bool:
    if not enabled or max_distance is None:
        return False

    numeric_distances = [
        chunk.distance for chunk in candidates if isinstance(chunk.distance, (int, float))
    ]
    if not numeric_distances:
        return False
    return all(distance > max_distance for distance in numeric_distances)


def _cap_candidates(
    candidates: Sequence[RetrievedChunk],
    *,
    max_total_candidates: int,
    initial_candidate_ids: Sequence[str],
    query_hints: dict[str, bool],
) -> list[RetrievedChunk]:
    cap = max(1, int(max_total_candidates))
    deduped = _dedupe_chunks(candidates)
    if len(deduped) <= cap:
        return deduped

    initial_ids = set(initial_candidate_ids)
    original_candidates = [chunk for chunk in deduped if chunk.id in initial_ids]
    expansion_candidates = [chunk for chunk in deduped if chunk.id not in initial_ids]

    if _should_boost_table_rows(query_hints):
        expansion_candidates = sorted(
            enumerate(expansion_candidates),
            key=lambda item: (item[1].metadata.get("chunk_type") != "table_row", item[0]),
        )
        expansion_candidates = [chunk for _, chunk in expansion_candidates]

    return (original_candidates + expansion_candidates)[:cap]


def _should_boost_table_rows(query_hints: dict[str, bool]) -> bool:
    return bool(
        query_hints.get("asks_table_or_structured_info")
        or query_hints.get("asks_number_or_money")
    )


def _rank_chunks(
    query: str,
    candidates: Sequence[RetrievedChunk],
    final_n: int,
    *,
    reranker_type: str,
    rerank_llm: Any | None,
    query_hints: dict[str, bool] | None = None,
) -> list[RetrievedChunk]:
    original_order = list(candidates)
    if reranker_type.casefold() == "llm" and rerank_llm is not None:
        ranked_ids = rerank_candidate_ids(
            query,
            original_order,
            final_n,
            rerank_llm,
            query_hints=query_hints,
        )
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
