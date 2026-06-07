from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    collection_name: str
    rows: int
    chunks: int


class CollectionCreateRequest(BaseModel):
    name: str
    description: str | None = None


class CollectionUpdateRequest(BaseModel):
    new_name: str | None = None
    metadata: dict[str, Any] | None = None


class CollectionInfo(BaseModel):
    name: str
    metadata: dict[str, Any] | None = None
    count: int


class QueryRequest(BaseModel):
    query: str
    columns_to_answer: list[str]
    number_docs_retrieval: int = Field(default=3, ge=1, le=50)


class QueryResponse(BaseModel):
    metadatas: list[Any]
    retrieved_data: str
    answer: str
    full_prompt: str
