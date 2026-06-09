from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from API_RAG_NEW import services
from API_RAG_NEW.concurrency import acquire_query_slot
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


@app.get("/health")
def health() -> dict[str, str]:
    return services.health_payload()


@app.get("/runtime-config", dependencies=[Depends(require_internal_api_key)])
def runtime_config() -> dict[str, object]:
    return services.runtime_config_payload()


@app.get("/runtime-status", dependencies=[Depends(require_internal_api_key)])
def runtime_status() -> dict[str, object]:
    return services.runtime_status_payload()


@app.get("/ui", include_in_schema=False)
def local_ui() -> FileResponse:
    ui_path = Path(__file__).resolve().parent.parent / "api_test_ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="api_test_ui.html not found")
    return FileResponse(ui_path)


@app.get("/collections", dependencies=[Depends(require_internal_api_key)])
def list_collections() -> dict[str, list[str]]:
    return services.list_collections()


@app.post(
    "/collections",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def create_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return services.create_collection(req)


@app.get(
    "/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def get_collection_info(collection_name: str) -> CollectionInfo:
    return services.get_collection_info(collection_name)


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
    return services.get_collection_records(collection_name, limit, offset)


@app.patch(
    "/collections/{collection_name}",
    response_model=CollectionInfo,
    dependencies=[Depends(require_internal_api_key)],
)
def update_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return services.update_collection(collection_name, req)


@app.delete(
    "/collections/{collection_name}",
    dependencies=[Depends(require_internal_api_key)],
)
def delete_collection(collection_name: str) -> dict[str, str]:
    return services.delete_collection(collection_name)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_internal_api_key)],
)
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str | None = Form(None),
) -> IngestResponse:
    raw_content = await file.read()
    if not raw_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return services.ingest_file_content(
        file.filename or "upload.txt",
        raw_content,
        collection_name,
    )


@app.post(
    "/collections/{collection_name}/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_internal_api_key)],
)
def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    with acquire_query_slot():
        return services.query_collection(collection_name, req)
