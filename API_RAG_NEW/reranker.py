from __future__ import annotations

import json
import re
from typing import Any, Protocol, Sequence


MAX_RERANK_CANDIDATE_CHARS = 1500


class RerankCandidate(Protocol):
    id: str
    document: str
    metadata: dict[str, Any]


class SupportsGenerateContent(Protocol):
    def generate_content(self, prompt: str) -> str:
        ...


def rerank_candidate_ids(
    query: str,
    candidates: Sequence[RerankCandidate],
    final_n: int,
    llm: SupportsGenerateContent,
    query_hints: dict[str, bool] | None = None,
) -> list[str]:
    original_ids = [candidate.id for candidate in candidates]
    if final_n <= 0 or not original_ids:
        return []

    try:
        response = llm.generate_content(
            _build_rerank_prompt(query, candidates, final_n, query_hints=query_hints)
        )
        ranked_ids = _parse_ranked_ids(response)
    except Exception:
        ranked_ids = []

    known_ids = set(original_ids)
    selected: list[str] = []
    seen: set[str] = set()
    for candidate_id in ranked_ids:
        if candidate_id in known_ids and candidate_id not in seen:
            selected.append(candidate_id)
            seen.add(candidate_id)
        if len(selected) >= final_n:
            return selected

    for candidate_id in original_ids:
        if candidate_id not in seen:
            selected.append(candidate_id)
            seen.add(candidate_id)
        if len(selected) >= final_n:
            break

    return selected


def _build_rerank_prompt(
    query: str,
    candidates: Sequence[RerankCandidate],
    final_n: int,
    *,
    query_hints: dict[str, bool] | None = None,
) -> str:
    payload = [
        {
            "id": candidate.id,
            "text": _truncate_candidate_text(candidate.document),
            "metadata": _compact_metadata(candidate.metadata),
        }
        for candidate in candidates
    ]
    hint_payload = {
        key: bool(value) for key, value in (query_hints or {}).items() if value
    }
    return (
        "You are a retrieval reranker for a RAG system.\n"
        "Task: Rank candidate IDs by usefulness for answering the user question.\n"
        "Prefer exact, source-backed evidence from the provided candidates.\n"
        "Prefer table_row chunks for questions about tables, criteria, pricing, "
        "rows, numbers, fees, costs, revenue, comparison, or structured fields.\n"
        "Prefer chunks with matching section_path or table_title when they clearly "
        "match the user question.\n"
        "Do not answer the question. Do not add explanations. Do not invent IDs.\n"
        'Return only valid JSON in this exact shape: {"ranked_ids": ["id1", "id2"]}.\n\n'
        f"User question: {query}\n"
        f"Internal query hints: {json.dumps(hint_payload, ensure_ascii=False)}\n"
        f"Return at most {final_n} IDs.\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "source_type",
        "chunk_index",
        "page_number",
        "page_chunk_index",
        "row_index",
        "row_chunk_index",
        "chunk_type",
        "section_title",
        "section_path",
        "block_index",
        "parent_id",
        "table_index",
        "table_title",
        "table_row_index",
        "table_row_part_index",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _truncate_candidate_text(text: str) -> str:
    if len(text) <= MAX_RERANK_CANDIDATE_CHARS:
        return text
    return text[:MAX_RERANK_CANDIDATE_CHARS]


def _parse_ranked_ids(text: str) -> list[str]:
    payload = json.loads(_extract_json_text(text))
    if isinstance(payload, list):
        return _ids_from_list(payload)
    if isinstance(payload, dict):
        for key in ("ranked_ids", "ids", "ranking", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return _ids_from_list(value)
    return []


def _ids_from_list(values: list[Any]) -> list[str]:
    ids: list[str] = []
    for value in values:
        if isinstance(value, str):
            ids.append(value)
        elif isinstance(value, dict):
            candidate_id = value.get("id") or value.get("candidate_id")
            if isinstance(candidate_id, str):
                ids.append(candidate_id)
    return ids


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    if stripped.startswith(("{", "[")):
        return stripped

    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not starts:
        return stripped
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        return stripped
    return stripped[start : end + 1]
