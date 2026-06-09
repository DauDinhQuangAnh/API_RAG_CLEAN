from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    collection_name: str
    rows: int
    chunks: int
    warnings: list[str] = Field(default_factory=list)
    chunking_profile: str | None = None
    chunk_stats: dict[str, Any] = Field(default_factory=dict)


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


class CollectionRecordsResponse(BaseModel):
    collection_name: str
    count: int
    limit: int
    offset: int
    ids: list[str]
    metadatas: list[Any]
    documents: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    number_docs_retrieval: int = Field(default=3, ge=1, le=50)


class Citation(BaseModel):
    id: int
    source: str | None = None
    source_type: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    page_chunk_index: int | None = None
    row_index: int | None = None
    row_chunk_index: int | None = None
    doc_id: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    chunk_type: str | None = None
    table_index: int | None = None
    table_title: str | None = None
    table_row_index: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    metadatas: list[Any]
    retrieved_data: str
    answer: str
    full_prompt: str
    citations: list[Citation] = Field(default_factory=list)
