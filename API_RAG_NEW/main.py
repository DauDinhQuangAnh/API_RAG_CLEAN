from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from API_RAG_NEW import services
from API_RAG_NEW.config import ALLOWED_ORIGINS, ROOT_PATH
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


@app.get("/runtime-config")
def runtime_config() -> dict[str, object]:
    return services.runtime_config_payload()


@app.get("/ui", include_in_schema=False)
def local_ui() -> FileResponse:
    ui_path = Path(__file__).resolve().parent.parent / "api_test_ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="api_test_ui.html not found")
    return FileResponse(ui_path)


@app.get("/collections")
def list_collections() -> dict[str, list[str]]:
    return services.list_collections()


@app.post("/collections", response_model=CollectionInfo)
def create_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return services.create_collection(req)


@app.get("/collections/{collection_name}", response_model=CollectionInfo)
def get_collection_info(collection_name: str) -> CollectionInfo:
    return services.get_collection_info(collection_name)


@app.get(
    "/collections/{collection_name}/records",
    response_model=CollectionRecordsResponse,
)
def get_collection_records(
    collection_name: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> CollectionRecordsResponse:
    return services.get_collection_records(collection_name, limit, offset)


@app.patch("/collections/{collection_name}", response_model=CollectionInfo)
def update_collection(
    collection_name: str,
    req: CollectionUpdateRequest,
) -> CollectionInfo:
    return services.update_collection(collection_name, req)


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str) -> dict[str, str]:
    return services.delete_collection(collection_name)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    index_column: str | None = Form(
        None,
        description="CSV/XLSX column to index and chunk",
    ),
    collection_name: str | None = Form(None),
) -> IngestResponse:
    raw_content = await file.read()
    if not raw_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return services.ingest_file_content(
        file.filename or "upload.csv",
        raw_content,
        collection_name,
        index_column=index_column,
    )


@app.post("/collections/{collection_name}/query", response_model=QueryResponse)
def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    return services.query_collection(collection_name, req)
