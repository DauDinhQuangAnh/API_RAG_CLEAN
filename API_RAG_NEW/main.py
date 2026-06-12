from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from API_RAG_NEW import services
from API_RAG_NEW.concurrency import acquire_ingest_slot, acquire_query_slot
from API_RAG_NEW.config import ALLOWED_ORIGINS, ROOT_PATH
from API_RAG_NEW.security import require_internal_api_key
from API_RAG_NEW.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionRecordsResponse,
    CollectionUpdateRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


app = FastAPI(title="RAG API", version="1.0.0", root_path=ROOT_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_PROVIDER = "local_sbert"
GEMINI_EMBEDDING_PROVIDER = "gemini"


@app.get("/health")
def health() -> dict[str, str]:
    return services.health_payload()


def _list_collections_for_provider(provider: str) -> dict[str, list[str]]:
    return services.list_collections(provider=provider)


def _create_collection_for_provider(
    provider: str,
    req: CollectionCreateRequest,
) -> CollectionInfo:
    return services.create_collection(req, provider=provider)


def _get_collection_info_for_provider(
    provider: str,
    collection_name: str,
) -> CollectionInfo:
    return services.get_collection_info(collection_name, provider=provider)


def _get_collection_records_for_provider(
    provider: str,
    collection_name: str,
    limit: int,
    offset: int,
) -> CollectionRecordsResponse:
    return services.get_collection_records(
        collection_name,
        limit,
        offset,
        provider=provider,
    )


def _update_collection_for_provider(
    provider: str,
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return services.update_collection(collection_name, req, provider=provider)


def _delete_collection_for_provider(
    provider: str,
    collection_name: str,
) -> dict[str, str]:
    return services.delete_collection(collection_name, provider=provider)


async def _ingest_file_for_provider(
    provider: str,
    file: UploadFile,
    collection_name: str | None,
    chunking_profile: str | None,
) -> IngestResponse:
    with acquire_ingest_slot():
        raw_content = await file.read()
        if not raw_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        effective_chunking_profile = services.resolve_chunking_profile(chunking_profile)
        return services.ingest_file_content(
            file.filename or "upload.txt",
            raw_content,
            collection_name,
            provider=provider,
            chunking_profile=effective_chunking_profile,
        )


def _query_collection_for_provider(
    provider: str,
    collection_name: str,
    req: QueryRequest,
) -> QueryResponse:
    with acquire_query_slot():
        return services.query_collection(collection_name, req, provider=provider)


@app.get("/runtime-config", dependencies=[Depends(require_internal_api_key)])
def runtime_config() -> dict[str, object]:
    return services.runtime_config_payload()


@app.get("/runtime-status", dependencies=[Depends(require_internal_api_key)])
def runtime_status() -> dict[str, object]:
    return services.runtime_status_payload()


@app.get("/collections", dependencies=[Depends(require_internal_api_key)])
def list_collections() -> dict[str, list[str]]:
    return _list_collections_for_provider(LOCAL_PROVIDER)


@app.post(
    "/collections",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def create_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return _create_collection_for_provider(LOCAL_PROVIDER, req)


@app.get(
    "/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def get_collection_info(collection_name: str) -> CollectionInfo:
    return _get_collection_info_for_provider(LOCAL_PROVIDER, collection_name)


@app.get(
    "/collections/{collection_name}/records",
    response_model=CollectionRecordsResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def get_collection_records(
    collection_name: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> CollectionRecordsResponse:
    return _get_collection_records_for_provider(
        LOCAL_PROVIDER,
        collection_name,
        limit,
        offset,
    )


@app.patch(
    "/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def update_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return _update_collection_for_provider(LOCAL_PROVIDER, collection_name, req)


@app.delete(
    "/collections/{collection_name}",
    dependencies=[Depends(require_internal_api_key)],
)
def delete_collection(collection_name: str) -> dict[str, str]:
    return _delete_collection_for_provider(LOCAL_PROVIDER, collection_name)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_internal_api_key)],
)
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    chunking_profile: str | None = Form(None),
) -> IngestResponse:
    return await _ingest_file_for_provider(
        LOCAL_PROVIDER,
        file,
        collection_name,
        chunking_profile,
    )


@app.post(
    "/collections/{collection_name}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    return _query_collection_for_provider(LOCAL_PROVIDER, collection_name, req)


@app.get("/local/collections", dependencies=[Depends(require_internal_api_key)])
def list_local_collections() -> dict[str, list[str]]:
    return _list_collections_for_provider(LOCAL_PROVIDER)


@app.post(
    "/local/collections",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def create_local_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return _create_collection_for_provider(LOCAL_PROVIDER, req)


@app.get(
    "/local/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def get_local_collection_info(collection_name: str) -> CollectionInfo:
    return _get_collection_info_for_provider(LOCAL_PROVIDER, collection_name)


@app.get(
    "/local/collections/{collection_name}/records",
    response_model=CollectionRecordsResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def get_local_collection_records(
    collection_name: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> CollectionRecordsResponse:
    return _get_collection_records_for_provider(
        LOCAL_PROVIDER,
        collection_name,
        limit,
        offset,
    )


@app.patch(
    "/local/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def update_local_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return _update_collection_for_provider(LOCAL_PROVIDER, collection_name, req)


@app.delete(
    "/local/collections/{collection_name}",
    dependencies=[Depends(require_internal_api_key)],
)
def delete_local_collection(collection_name: str) -> dict[str, str]:
    return _delete_collection_for_provider(LOCAL_PROVIDER, collection_name)


@app.post(
    "/local/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_internal_api_key)],
)
async def ingest_local_file(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    chunking_profile: str | None = Form(None),
) -> IngestResponse:
    return await _ingest_file_for_provider(
        LOCAL_PROVIDER,
        file,
        collection_name,
        chunking_profile,
    )


@app.post(
    "/local/collections/{collection_name}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def query_local_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    return _query_collection_for_provider(LOCAL_PROVIDER, collection_name, req)


@app.get("/gemini/collections", dependencies=[Depends(require_internal_api_key)])
def list_gemini_collections() -> dict[str, list[str]]:
    return _list_collections_for_provider(GEMINI_EMBEDDING_PROVIDER)


@app.post(
    "/gemini/collections",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def create_gemini_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return _create_collection_for_provider(GEMINI_EMBEDDING_PROVIDER, req)


@app.get(
    "/gemini/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def get_gemini_collection_info(collection_name: str) -> CollectionInfo:
    return _get_collection_info_for_provider(GEMINI_EMBEDDING_PROVIDER, collection_name)


@app.get(
    "/gemini/collections/{collection_name}/records",
    response_model=CollectionRecordsResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def get_gemini_collection_records(
    collection_name: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> CollectionRecordsResponse:
    return _get_collection_records_for_provider(
        GEMINI_EMBEDDING_PROVIDER,
        collection_name,
        limit,
        offset,
    )


@app.patch(
    "/gemini/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def update_gemini_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return _update_collection_for_provider(
        GEMINI_EMBEDDING_PROVIDER,
        collection_name,
        req,
    )


@app.delete(
    "/gemini/collections/{collection_name}",
    dependencies=[Depends(require_internal_api_key)],
)
def delete_gemini_collection(collection_name: str) -> dict[str, str]:
    return _delete_collection_for_provider(GEMINI_EMBEDDING_PROVIDER, collection_name)


@app.post(
    "/gemini/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_internal_api_key)],
)
async def ingest_gemini_file(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
    chunking_profile: str | None = Form(None),
) -> IngestResponse:
    return await _ingest_file_for_provider(
        GEMINI_EMBEDDING_PROVIDER,
        file,
        collection_name,
        chunking_profile,
    )


@app.post(
    "/gemini/collections/{collection_name}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def query_gemini_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    return _query_collection_for_provider(
        GEMINI_EMBEDDING_PROVIDER,
        collection_name,
        req,
    )
