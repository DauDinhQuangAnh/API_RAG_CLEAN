from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from API_RAG_NEW import services
from API_RAG_NEW.config import ALLOWED_ORIGINS, ROOT_PATH
from API_RAG_NEW.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
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


@app.get("/collections")
def list_collections() -> dict[str, list[str]]:
    return services.list_collections()


@app.post("/collections", response_model=CollectionInfo)
def create_collection(req: CollectionCreateRequest) -> CollectionInfo:
    return services.create_collection(req)


@app.get("/collections/{collection_name}", response_model=CollectionInfo)
def get_collection_info(collection_name: str) -> CollectionInfo:
    return services.get_collection_info(collection_name)


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
async def ingest_csv(
    file: UploadFile = File(...),
    index_column: str = Form(..., description="Column to index and chunk"),
    collection_name: str | None = Form(None),
) -> IngestResponse:
    raw_content = await file.read()
    return services.ingest_csv_content(
        file.filename or "upload.csv",
        raw_content,
        index_column,
        collection_name,
    )


@app.post("/collections/{collection_name}/query", response_model=QueryResponse)
def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    return services.query_collection(collection_name, req)
