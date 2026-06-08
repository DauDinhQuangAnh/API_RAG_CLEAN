from __future__ import annotations

import io
import hashlib
import os
import re
import uuid
from typing import Any

import pandas as pd
from fastapi import HTTPException
from docx import Document
import pdfplumber

from chunking import ProtonxSemanticChunker
from llms.onlinellms import OnlineLLMs

from API_RAG_NEW.config import (
    CHROMA_CLIENT,
    DEFAULT_COLLECTION_DESCRIPTION,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    GEMINI_PROVIDER,
    INGEST_BATCH_SIZE,
    RAG_FINAL_TOP_N,
    RAG_INCLUDE_NEIGHBORS,
    RAG_INITIAL_TOP_K,
    RAG_RERANKER_TYPE,
    get_gemini_api_key,
)
from API_RAG_NEW.rag_pipeline import (
    add_records_to_collection,
    clean_collection_name,
    stable_record_id,
    vector_search,
)
from API_RAG_NEW.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionRecordsResponse,
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


def get_collection_records(
    collection_name: str,
    limit: int,
    offset: int,
) -> CollectionRecordsResponse:
    collection = _get_collection_or_404(collection_name)
    payload = collection.get(
        limit=limit,
        offset=offset,
        include=["metadatas", "documents"],
    )
    return CollectionRecordsResponse(
        collection_name=collection.name,
        count=collection.count(),
        limit=limit,
        offset=offset,
        ids=payload.get("ids") or [],
        metadatas=payload.get("metadatas") or [],
        documents=payload.get("documents") or [],
    )


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
    return ingest_file_content(
        file_name,
        raw_content,
        requested_collection_name,
        index_column=index_column,
    )


def ingest_file_content(
    file_name: str,
    raw_content: bytes,
    requested_collection_name: str | None,
    index_column: str | None = None,
) -> IngestResponse:
    extension = os.path.splitext(file_name)[1].casefold()
    if extension not in {".csv", ".xlsx", ".docx", ".pdf", ".txt", ".text"}:
        raise HTTPException(
            status_code=400,
            detail="Only DOCX, PDF, TXT, TEXT, CSV, and XLSX files are supported.",
        )

    final_collection_name = _resolve_collection_name(file_name, requested_collection_name)
    file_hash = _content_hash(raw_content)
    collection = CHROMA_CLIENT.get_or_create_collection(
        name=final_collection_name,
        metadata={"description": DEFAULT_COLLECTION_DESCRIPTION},
    )

    chunker = ProtonxSemanticChunker(model=EMBEDDING_MODEL)
    pending_records: list[dict[str, Any]] = []
    chunk_count = 0

    source_count = 1
    if extension in {".csv", ".xlsx"}:
        if not index_column:
            raise HTTPException(
                status_code=400,
                detail="index_column is required for CSV/XLSX ingest.",
            )
        dataframe = _read_tabular_file(raw_content, extension, index_column)
        source_count = len(dataframe)
        records = _iter_tabular_chunk_records(
            dataframe,
            index_column,
            file_name,
            extension,
            file_hash,
            chunker,
        )
    elif extension == ".pdf":
        pages = _extract_pdf_pages(raw_content)
        records = _iter_pdf_chunk_records(pages, file_name, extension, file_hash, chunker)
    else:
        text = _extract_non_pdf_document_text(file_name, raw_content, extension)
        records = _iter_document_chunk_records(
            text,
            file_name,
            extension,
            file_hash,
            chunker,
        )

    for record in records:
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
        rows=source_count,
        chunks=chunk_count,
    )


def query_collection(collection_name: str, req: QueryRequest) -> QueryResponse:
    collection = _get_collection_or_404(collection_name)
    final_n = _resolve_final_docs_retrieval(req)
    rerank_llm = _build_optional_rerank_llm()
    try:
        metadatas, retrieved_data = vector_search(
            EMBEDDING_MODEL,
            req.query,
            collection,
            req.columns_to_answer,
            final_n,
            initial_top_k=RAG_INITIAL_TOP_K,
            include_neighbors=RAG_INCLUDE_NEIGHBORS,
            reranker_type=RAG_RERANKER_TYPE,
            rerank_llm=rerank_llm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    full_prompt = _build_query_prompt(req.query, retrieved_data)
    answer_llm = rerank_llm or _build_llm()
    answer = answer_llm.generate_content(full_prompt)
    return QueryResponse(
        metadatas=metadatas,
        retrieved_data=retrieved_data,
        answer=answer,
        full_prompt=full_prompt,
    )


def _resolve_final_docs_retrieval(req: QueryRequest) -> int:
    fields_set = getattr(req, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(req, "__fields_set__", set())
    if "number_docs_retrieval" in fields_set:
        return req.number_docs_retrieval
    return RAG_FINAL_TOP_N


def _build_optional_rerank_llm() -> OnlineLLMs | None:
    if RAG_RERANKER_TYPE.casefold() != "llm":
        return None
    try:
        return _build_llm()
    except Exception:
        return None


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


def _content_hash(raw_content: bytes) -> str:
    return hashlib.sha256(raw_content).hexdigest()


def _document_id(file_hash: str) -> str:
    return f"doc_{file_hash[:32]}"


def _read_tabular_file(
    raw_content: bytes,
    extension: str,
    index_column: str,
) -> pd.DataFrame:
    file_type = extension.lstrip(".").upper()
    try:
        if extension == ".csv":
            dataframe = pd.read_csv(io.BytesIO(raw_content))
        else:
            dataframe = pd.read_excel(io.BytesIO(raw_content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read {file_type}: {exc}",
        ) from exc

    if index_column not in dataframe.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{index_column}' not found in {file_type} file.",
        )

    return dataframe.copy()


def _extract_non_pdf_document_text(
    file_name: str, raw_content: bytes, extension: str
) -> str:
    try:
        if extension == ".docx":
            return _extract_docx_text(raw_content)
        if extension in {".txt", ".text"}:
            return _decode_text_file(raw_content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read {extension.lstrip('.').upper()} file: {exc}",
        ) from exc

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_name}")


def _extract_pdf_pages(raw_content: bytes) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(io.BytesIO(raw_content)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            page_text = _clean_pdf_page_text(page_text)
            if page_text:
                pages.append((page_number, page_text))

    return pages


def _clean_pdf_page_text(text: str) -> str:
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"Trang\s+\d+\s*/\s*\d+", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"[•●○▪\-–—]+", line):
            continue
        lines.append(line)

    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if not current:
            current = line
            continue

        if _should_join_pdf_line(current, line):
            current = f"{current} {line}"
        else:
            paragraphs.append(current)
            current = line

    if current:
        paragraphs.append(current)

    cleaned = "\n".join(paragraphs)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _should_join_pdf_line(previous: str, current: str) -> bool:
    if re.fullmatch(r"\d+\.", previous):
        return True
    if previous.endswith(("/", "-", "–", "—")):
        return True
    if re.search(r"[.!?;:]$", previous):
        return False
    if re.match(r"^\d+\.|^[a-zA-Z]\)", current):
        return False
    return True


def _extract_docx_text(raw_content: bytes) -> str:
    document = Document(io.BytesIO(raw_content))
    blocks: list[str] = []

    blocks.extend(paragraph.text.strip() for paragraph in document.paragraphs)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    return "\n\n".join(block for block in blocks if block)


def _decode_text_file(raw_content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            return raw_content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise HTTPException(status_code=400, detail="Failed to decode text file.")


def _iter_document_chunk_records(
    text: str,
    file_name: str,
    extension: str,
    file_hash: str,
    chunker: ProtonxSemanticChunker,
):
    if not text.strip():
        return

    doc_id = _document_id(file_hash)
    source_type = extension.lstrip(".")
    for chunk_index, chunk in enumerate(chunker.split_text(text), start=1):
        if not chunk.strip():
            continue
        yield {
            "id": stable_record_id(doc_id, file_name, source_type, chunk_index, chunk),
            "chunk": chunk,
            "doc_id": doc_id,
            "source": file_name,
            "source_type": source_type,
            "chunk_index": chunk_index,
        }


def _iter_pdf_chunk_records(
    pages: list[tuple[int, str]],
    file_name: str,
    extension: str,
    file_hash: str,
    chunker: ProtonxSemanticChunker,
):
    doc_id = _document_id(file_hash)
    source_type = extension.lstrip(".")
    chunk_index = 0
    for page_number, page_text in pages:
        page_chunk_index = 0
        for chunk in chunker.split_text(page_text):
            if not chunk.strip():
                continue
            chunk_index += 1
            page_chunk_index += 1
            yield {
                "id": stable_record_id(
                    doc_id,
                    file_name,
                    source_type,
                    page_number,
                    page_chunk_index,
                    chunk,
                ),
                "chunk": chunk,
                "doc_id": doc_id,
                "source": file_name,
                "source_type": source_type,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "page_chunk_index": page_chunk_index,
            }


def _iter_tabular_chunk_records(
    dataframe: pd.DataFrame,
    index_column: str,
    file_name: str,
    extension: str,
    file_hash: str,
    chunker: ProtonxSemanticChunker,
):
    source_type = extension.lstrip(".")
    reserved_metadata_keys = {
        "id",
        "_id",
        "chunk",
        "doc_id",
        "source",
        "source_type",
        "chunk_index",
        "row_index",
        "row_chunk_index",
        "page_number",
        "page_chunk_index",
    }
    chunk_index = 0
    for row_offset, row in dataframe.reset_index(drop=True).iterrows():
        row_index = int(row_offset) + 1
        row_data = {
            key: _normalize_dataframe_value(value)
            for key, value in row.to_dict().items()
        }
        text = row_data.get(index_column)
        if not isinstance(text, str) or not text.strip():
            continue

        row_doc_id = stable_record_id("row", file_hash, row_index, prefix="doc")
        row_chunk_index = 0
        for chunk in chunker.split_text(text):
            if not chunk.strip():
                continue
            chunk_index += 1
            row_chunk_index += 1
            yield {
                "id": stable_record_id(
                    row_doc_id,
                    file_name,
                    source_type,
                    row_index,
                    row_chunk_index,
                    chunk,
                ),
                "chunk": chunk,
                "doc_id": row_doc_id,
                "source": file_name,
                "source_type": source_type,
                "chunk_index": chunk_index,
                "row_index": row_index,
                "row_chunk_index": row_chunk_index,
                **{
                    key: value
                    for key, value in row_data.items()
                    if key not in reserved_metadata_keys
                },
            }


def _normalize_dataframe_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
