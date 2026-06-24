from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from API_RAG_NEW.config import (
    EmbeddingRuntime,
    GEMINI_PROVIDER,
    LOCAL_EMBEDDING_PROVIDER,
)
from API_RAG_NEW.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionRecordsResponse,
    CollectionUpdateRequest,
)
from API_RAG_NEW._services_shared import (
    GEMINI_STORAGE_PREFIX,
    _clean_logical_collection_name,
    _get_collection_or_404,
    _runtime_for_provider,
    _to_collection_info,
    logical_collection_name,
    storage_collection_name,
)


def list_collections(provider: str = LOCAL_EMBEDDING_PROVIDER) -> dict[str, list[str]]:
    runtime = _runtime_for_provider(provider)
    collections = runtime.chroma_client.list_collections()
    return {
        "collections": [
            logical_collection_name(runtime.provider, collection.name, collection.metadata)
            for collection in collections
            if _collection_matches_runtime_metadata(runtime, collection)
        ]
    }


def create_collection(
    req: CollectionCreateRequest,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> CollectionInfo:
    runtime = _runtime_for_provider(provider)
    logical_name = _clean_logical_collection_name(req.name)
    storage_name = storage_collection_name(runtime.provider, logical_name)

    existing_names = {
        collection.name for collection in runtime.chroma_client.list_collections()
    }
    if storage_name in existing_names:
        raise HTTPException(status_code=400, detail="Collection already exists.")

    collection = runtime.chroma_client.get_or_create_collection(
        name=storage_name,
        metadata=_embedding_collection_metadata(
            runtime,
            req.description,
            logical_name=logical_name,
            storage_name=storage_name,
        ),
    )
    return _to_collection_info(runtime, collection)


def get_collection_info(
    collection_name: str,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> CollectionInfo:
    runtime = _runtime_for_provider(provider)
    storage_name = storage_collection_name(runtime.provider, collection_name)
    collection = _get_collection_or_404(runtime, storage_name)
    _validate_collection_embedding_metadata(runtime, collection)
    return _to_collection_info(runtime, collection)


def get_collection_records(
    collection_name: str,
    limit: int,
    offset: int,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> CollectionRecordsResponse:
    runtime = _runtime_for_provider(provider)
    storage_name = storage_collection_name(runtime.provider, collection_name)
    collection = _get_collection_or_404(runtime, storage_name)
    _validate_collection_embedding_metadata(runtime, collection)
    payload = collection.get(
        limit=limit,
        offset=offset,
        include=["metadatas", "documents"],
    )
    return CollectionRecordsResponse(
        collection_name=logical_collection_name(
            runtime.provider,
            collection.name,
            collection.metadata,
        ),
        count=collection.count(),
        limit=limit,
        offset=offset,
        ids=payload.get("ids") or [],
        metadatas=payload.get("metadatas") or [],
        documents=payload.get("documents") or [],
    )


def update_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> CollectionInfo:
    runtime = _runtime_for_provider(provider)
    current_storage_name = storage_collection_name(runtime.provider, collection_name)
    collection = _get_collection_or_404(runtime, current_storage_name)
    _validate_collection_embedding_metadata(runtime, collection)
    new_logical_name = req.new_name or None
    new_storage_name = None
    requested_metadata = req.metadata or None

    if not new_logical_name and not requested_metadata:
        raise HTTPException(
            status_code=400,
            detail="Nothing to update (new_name or metadata required).",
        )

    if new_logical_name:
        new_logical_name = _clean_logical_collection_name(new_logical_name)
        new_storage_name = storage_collection_name(runtime.provider, new_logical_name)
        if new_storage_name != current_storage_name:
            existing_names = {
                collection.name
                for collection in runtime.chroma_client.list_collections()
            }
            if new_storage_name in existing_names:
                raise HTTPException(
                    status_code=400,
                    detail="Collection already exists.",
                )

    new_metadata = None
    if requested_metadata is not None:
        _ensure_embedding_metadata_not_changed(
            collection.metadata or {},
            requested_metadata,
        )
        new_metadata = _preserve_embedding_metadata(
            collection.metadata or {},
            requested_metadata,
        )

    target_logical_name = new_logical_name or logical_collection_name(
        runtime.provider,
        collection.name,
        collection.metadata,
    )
    target_storage_name = new_storage_name or current_storage_name

    merged_metadata = _collection_identity_metadata(
        runtime,
        new_metadata if new_metadata is not None else dict(collection.metadata or {}),
        target_logical_name,
        target_storage_name,
    )

    if new_storage_name is None:
        collection.modify(metadata=merged_metadata)
    else:
        collection.modify(name=new_storage_name, metadata=merged_metadata)
    return _to_collection_info(
        runtime,
        _get_collection_or_404(runtime, target_storage_name),
    )


def delete_collection(
    collection_name: str,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> dict[str, str]:
    runtime = _runtime_for_provider(provider)
    storage_name = storage_collection_name(runtime.provider, collection_name)
    collection = _get_collection_or_404(runtime, storage_name)
    _validate_collection_embedding_metadata(runtime, collection)
    runtime.chroma_client.delete_collection(name=storage_name)
    return {"detail": "Collection deleted successfully."}


def _embedding_collection_metadata(
    runtime: EmbeddingRuntime,
    description: str | None,
    chunking_profile: str | None = None,
    *,
    logical_name: str | None = None,
    storage_name: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "embedding_provider": runtime.provider,
        "embedding_model": runtime.model_name,
        "embedding_dimension": int(runtime.dimension),
    }
    if logical_name:
        metadata["logical_collection_name"] = logical_name
    if storage_name:
        metadata["storage_collection_name"] = storage_name
    if chunking_profile:
        metadata["chunking_profile"] = chunking_profile
    if description:
        metadata["description"] = description
    return metadata


def _ensure_embedding_metadata_not_changed(
    current_metadata: dict[str, Any],
    requested_metadata: dict[str, Any],
) -> None:
    for key in (
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "chunking_profile",
        "logical_collection_name",
        "storage_collection_name",
    ):
        if key not in requested_metadata:
            continue
        if key not in current_metadata:
            raise HTTPException(
                status_code=400,
                detail="Embedding metadata cannot be changed.",
            )
        if str(requested_metadata[key]) != str(current_metadata[key]):
            raise HTTPException(
                status_code=400,
                detail="Embedding metadata cannot be changed.",
            )


def _preserve_embedding_metadata(
    current_metadata: dict[str, Any],
    requested_metadata: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(requested_metadata)
    for key in (
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "chunking_profile",
        "logical_collection_name",
        "storage_collection_name",
    ):
        if key in current_metadata:
            merged[key] = current_metadata[key]
    return merged


def _collection_identity_metadata(
    runtime: EmbeddingRuntime,
    metadata: dict[str, Any],
    logical_name: str,
    storage_name: str,
) -> dict[str, Any]:
    merged = dict(metadata)
    merged.update(
        _embedding_collection_metadata(
            runtime,
            None,
            logical_name=logical_name,
            storage_name=storage_name,
        )
    )
    return merged


def _validate_collection_embedding_metadata(
    runtime: EmbeddingRuntime,
    collection: Any,
) -> None:
    if _collection_matches_runtime_metadata(runtime, collection):
        return

    metadata = collection.metadata or {}
    provider = metadata.get("embedding_provider")
    model = metadata.get("embedding_model")
    dimension = metadata.get("embedding_dimension")

    if provider is None and model is None and dimension is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Collection has no embedding metadata. Please re-ingest documents "
                "into a Gemini embedding collection."
            ),
        )

    raise HTTPException(
        status_code=400,
        detail=(
            "Collection was created with a different embedding provider/model. "
            "Please create a new collection or re-ingest documents."
        ),
    )


def _collection_matches_runtime_metadata(
    runtime: EmbeddingRuntime,
    collection: Any,
) -> bool:
    metadata = collection.metadata or {}
    collection_name = str(collection.name)
    storage_metadata = metadata.get("storage_collection_name")
    if storage_metadata is not None and str(storage_metadata) != collection_name:
        return False

    if runtime.provider == GEMINI_PROVIDER:
        if not collection_name.startswith(GEMINI_STORAGE_PREFIX):
            return False
    elif collection_name.startswith(GEMINI_STORAGE_PREFIX):
        return False

    provider = metadata.get("embedding_provider")
    model = metadata.get("embedding_model")
    dimension = metadata.get("embedding_dimension")

    if provider is None and model is None and dimension is None:
        return runtime.provider == LOCAL_EMBEDDING_PROVIDER

    return (
        str(provider) == runtime.provider
        and str(model) == runtime.model_name
        and _metadata_dimension(dimension) == int(runtime.dimension)
    )


def _metadata_dimension(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
