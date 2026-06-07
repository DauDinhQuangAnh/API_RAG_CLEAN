from __future__ import annotations

import io
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import HTTPException

from chunking import ProtonxSemanticChunker
from llms.onlinellms import OnlineLLMs

from API_RAG_NEW.config import (
    CHROMA_CLIENT,
    DEFAULT_COLLECTION_DESCRIPTION,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    GEMINI_PROVIDER,
    INGEST_BATCH_SIZE,
    get_gemini_api_key,
)
from API_RAG_NEW.rag_pipeline import (
    add_records_to_collection,
    clean_collection_name,
    vector_search,
)
from API_RAG_NEW.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionUpdateRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


def health_payload() -> dict[str, str]:
    return {"status": "ok"}


def list_collections() -> dict[str, list[str]]:
    collections = CHROMA_CLIENT.list_collections()
    return {"collections": [collection.name for collection in collections]}


def create_collection(req: CollectionCreateRequest) -> CollectionInfo:
    cleaned_name = clean_collection_name(req.name)
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Invalid collection name.")

    existing_names = {collection.name for collection in CHROMA_CLIENT.list_collections()}
    if cleaned_name in existing_names:
        raise HTTPException(status_code=400, detail="Collection already exists.")

    metadata = {"description": req.description} if req.description else None
    collection = CHROMA_CLIENT.get_or_create_collection(
        name=cleaned_name,
        metadata=metadata,
    )
    return _to_collection_info(collection)


def get_collection_info(collection_name: str) -> CollectionInfo:
    return _to_collection_info(_get_collection_or_404(collection_name))


def update_collection(
    collection_name: str, req: CollectionUpdateRequest
) -> CollectionInfo:
    collection = _get_collection_or_404(collection_name)
    new_name = req.new_name or None
    new_metadata = req.metadata or None

    if not new_name and not new_metadata:
        raise HTTPException(
            status_code=400,
            detail="Nothing to update (new_name or metadata required).",
        )

    if new_name:
        cleaned_name = clean_collection_name(new_name)
        if not cleaned_name:
            raise HTTPException(status_code=400, detail="Invalid new_name.")
        new_name = cleaned_name

    collection.modify(name=new_name, metadata=new_metadata)
    return _to_collection_info(_get_collection_or_404(new_name or collection_name))


def delete_collection(collection_name: str) -> dict[str, str]:
    _get_collection_or_404(collection_name)
    CHROMA_CLIENT.delete_collection(name=collection_name)
    return {"detail": "Collection deleted successfully."}


def ingest_csv_content(
    file_name: str,
    raw_content: bytes,
    index_column: str,
    requested_collection_name: str | None,
) -> IngestResponse:
    if not file_name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        dataframe = pd.read_csv(io.BytesIO(raw_content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    if index_column not in dataframe.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{index_column}' not found.",
        )

    dataframe = dataframe.copy()
    dataframe["doc_id"] = [str(uuid.uuid4()) for _ in range(len(dataframe))]

    final_collection_name = _resolve_collection_name(file_name, requested_collection_name)
    collection = CHROMA_CLIENT.get_or_create_collection(
        name=final_collection_name,
        metadata={"description": DEFAULT_COLLECTION_DESCRIPTION},
    )

    chunker = ProtonxSemanticChunker(model=EMBEDDING_MODEL)
    pending_records: list[dict[str, Any]] = []
    chunk_count = 0

    for record in _iter_chunk_records(dataframe, index_column, chunker):
        pending_records.append(record)
        if len(pending_records) >= INGEST_BATCH_SIZE:
            chunk_count += add_records_to_collection(
                pending_records,
                EMBEDDING_MODEL,
                collection,
            )
            pending_records.clear()

    if pending_records:
        chunk_count += add_records_to_collection(
            pending_records,
            EMBEDDING_MODEL,
            collection,
        )

    if chunk_count == 0:
        raise HTTPException(status_code=400, detail="No valid text to chunk.")

    return IngestResponse(
        collection_name=final_collection_name,
        rows=len(dataframe),
        chunks=chunk_count,
    )


def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    collection = _get_collection_or_404(collection_name)
    metadatas, retrieved_data = vector_search(
        EMBEDDING_MODEL,
        req.query,
        collection,
        req.columns_to_answer,
        req.number_docs_retrieval,
    )
    full_prompt = _build_query_prompt(req.query, retrieved_data)
    answer = _build_llm().generate_content(full_prompt)
    return QueryResponse(
        metadatas=metadatas,
        retrieved_data=retrieved_data,
        answer=answer,
        full_prompt=full_prompt,
    )


def _build_llm(api_key: str | None = None) -> OnlineLLMs:
    resolved_api_key = api_key or get_gemini_api_key()
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not configured")

    return OnlineLLMs(
        name=GEMINI_PROVIDER,
        api_key=resolved_api_key,
        model_version=GEMINI_MODEL,
    )


def _build_query_prompt(query: str, retrieved_data: str) -> str:
    return (
        "You are Weavey, a RAG question-answering assistant for Vietnamese textile "
        "and apparel teams.\n"
        "Answer only from the reference data below. Do not add next steps, decisions, "
        "legal content, financial content, or unsupported claims. If the reference "
        "data does not contain enough information, say so briefly in Vietnamese.\n"
        "Answer in Vietnamese and cite specific details from the reference data when "
        "available.\n\n"
        f"User question: {query}\n\n"
        f"Reference data:\n{retrieved_data}"
    )


def _get_collection_or_404(collection_name: str) -> Any:
    try:
        return CHROMA_CLIENT.get_collection(name=collection_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Collection not found.") from exc


def _to_collection_info(collection: Any) -> CollectionInfo:
    return CollectionInfo(
        name=collection.name,
        metadata=collection.metadata,
        count=collection.count(),
    )


def _resolve_collection_name(
    file_name: str, requested_collection_name: str | None
) -> str:
    if requested_collection_name:
        cleaned_name = clean_collection_name(requested_collection_name)
        if not cleaned_name:
            raise HTTPException(status_code=400, detail="Invalid collection_name.")
        return cleaned_name

    base_name = clean_collection_name(os.path.splitext(file_name)[0]) or "rag_collection"
    return f"rag_collection_{base_name}_{uuid.uuid4().hex[:6]}"


def _iter_chunk_records(
    dataframe: pd.DataFrame,
    index_column: str,
    chunker: ProtonxSemanticChunker,
):
    for _, row in dataframe.iterrows():
        row_data = {
            key: _normalize_dataframe_value(value)
            for key, value in row.to_dict().items()
        }
        text = row_data.get(index_column)
        if not isinstance(text, str) or not text.strip():
            continue

        for chunk in chunker.split_text(text):
            if not chunk.strip():
                continue
            yield {
                "chunk": chunk,
                **{
                    key: value
                    for key, value in row_data.items()
                    if key not in {"chunk", "_id"}
                },
            }


def _normalize_dataframe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value
