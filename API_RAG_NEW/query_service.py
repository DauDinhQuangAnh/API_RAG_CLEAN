from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from fastapi import HTTPException

from llms.onlinellms import OnlineLLMs

from API_RAG_NEW.citations import build_citations_from_metadatas
from API_RAG_NEW.config import (
    GEMINI_MODEL,
    GEMINI_RERANKER_MODEL,
    LOCAL_EMBEDDING_PROVIDER,
    RAG_CROSS_ENCODER_MODEL,
    RAG_DEBUG_MODE,
    RAG_ENABLE_DISTANCE_GUARD,
    RAG_ENABLE_FINAL_ANSWER_FALLBACK,
    RAG_FINAL_TOP_N,
    RAG_INCLUDE_NEIGHBORS,
    RAG_INITIAL_TOP_K,
    RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE,
    RAG_MAX_DISTANCE,
    RAG_MAX_TOTAL_CANDIDATES,
    RAG_RERANKER_TYPE,
    get_gemini_api_key,
)
from API_RAG_NEW.rag_pipeline import vector_search
from API_RAG_NEW.schemas import QueryRequest, QueryResponse
from API_RAG_NEW._services_shared import (
    FINAL_ANSWER_FALLBACK_MESSAGE,
    NO_CONTEXT_ANSWER_MESSAGE,
    _get_collection_or_404,
    _runtime_for_provider,
    storage_collection_name,
)
from API_RAG_NEW.collection_service import _validate_collection_embedding_metadata


def query_collection(
    collection_name: str,
    req: QueryRequest,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    runtime = _runtime_for_provider(provider)
    storage_name = storage_collection_name(runtime.provider, collection_name)
    collection = _get_collection_or_404(runtime, storage_name)
    _validate_collection_embedding_metadata(runtime, collection)
    final_n = _resolve_final_docs_retrieval(req)
    try:
        metadatas, retrieved_data = vector_search(
            runtime.embedding_model,
            req.query,
            collection,
            final_n,
            initial_top_k=RAG_INITIAL_TOP_K,
            include_neighbors=RAG_INCLUDE_NEIGHBORS,
            reranker_type=RAG_RERANKER_TYPE,
            rerank_llm_factory=_build_optional_rerank_llm,
            max_context_expansion_per_candidate=(
                RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE
            ),
            max_total_candidates=RAG_MAX_TOTAL_CANDIDATES,
            enable_distance_guard=RAG_ENABLE_DISTANCE_GUARD,
            max_distance=RAG_MAX_DISTANCE,
            cross_encoder_model=RAG_CROSS_ENCODER_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    final_metadatas = metadatas[0] if metadatas else []
    citations = build_citations_from_metadatas(final_metadatas)
    full_prompt = _build_query_prompt(req.query, retrieved_data)
    include_debug = getattr(req, "include_debug_info", False) or RAG_DEBUG_MODE

    if not str(retrieved_data or "").strip():
        return QueryResponse(
            metadatas=metadatas,
            retrieved_data=retrieved_data,
            answer=NO_CONTEXT_ANSWER_MESSAGE,
            full_prompt=full_prompt if include_debug else None,
            citations=[],
        )

    answer_llm = _build_llm()
    try:
        answer = answer_llm.generate_content(full_prompt)
    except Exception as exc:
        if not RAG_ENABLE_FINAL_ANSWER_FALLBACK:
            raise
        print(f"Final answer generation failed: {exc}")
        answer = FINAL_ANSWER_FALLBACK_MESSAGE

    return QueryResponse(
        metadatas=metadatas,
        retrieved_data=retrieved_data,
        answer=answer,
        full_prompt=full_prompt if include_debug else None,
        citations=citations,
    )


def query_collection_stream(
    collection_name: str,
    req: QueryRequest,
    provider: str = LOCAL_EMBEDDING_PROVIDER,
) -> Generator[str, None, None]:
    """Sync generator trả SSE events: metadata → token... → done."""
    if not req.query.strip():
        yield _sse({"type": "error", "message": "Query must not be empty."})
        return

    runtime = _runtime_for_provider(provider)
    storage_name = storage_collection_name(runtime.provider, collection_name)
    try:
        collection = _get_collection_or_404(runtime, storage_name)
        _validate_collection_embedding_metadata(runtime, collection)
    except HTTPException as exc:
        yield _sse({"type": "error", "message": exc.detail})
        return

    final_n = _resolve_final_docs_retrieval(req)
    try:
        metadatas, retrieved_data = vector_search(
            runtime.embedding_model,
            req.query,
            collection,
            final_n,
            initial_top_k=RAG_INITIAL_TOP_K,
            include_neighbors=RAG_INCLUDE_NEIGHBORS,
            reranker_type=RAG_RERANKER_TYPE,
            rerank_llm_factory=_build_optional_rerank_llm,
            max_context_expansion_per_candidate=(
                RAG_MAX_CONTEXT_EXPANSION_PER_CANDIDATE
            ),
            max_total_candidates=RAG_MAX_TOTAL_CANDIDATES,
            enable_distance_guard=RAG_ENABLE_DISTANCE_GUARD,
            max_distance=RAG_MAX_DISTANCE,
            cross_encoder_model=RAG_CROSS_ENCODER_MODEL,
        )
    except (ValueError, HTTPException) as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        yield _sse({"type": "error", "message": message})
        return

    final_metadatas = metadatas[0] if metadatas else []
    citations = build_citations_from_metadatas(final_metadatas)
    citations_data = [c.model_dump() for c in citations]

    yield _sse({
        "type": "metadata",
        "retrieved_data": retrieved_data,
        "citations": citations_data,
        "metadatas": final_metadatas,
    })

    if not str(retrieved_data or "").strip():
        yield _sse({"type": "answer", "content": NO_CONTEXT_ANSWER_MESSAGE})
        yield _sse({"type": "done"})
        return

    full_prompt = _build_query_prompt(req.query, retrieved_data)
    try:
        llm = _build_llm()
        for token in llm.generate_content_stream(full_prompt):
            yield _sse({"type": "token", "content": token})
    except Exception as exc:
        if not RAG_ENABLE_FINAL_ANSWER_FALLBACK:
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})
            return
        print(f"Streaming answer generation failed: {exc}")
        yield _sse({"type": "answer", "content": FINAL_ANSWER_FALLBACK_MESSAGE})

    yield _sse({"type": "done"})


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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
        return _build_llm(model_version=GEMINI_RERANKER_MODEL)
    except Exception:
        return None


def _build_llm(
    api_key: str | None = None,
    model_version: str | None = None,
) -> OnlineLLMs:
    resolved_api_key = api_key or get_gemini_api_key()
    if not resolved_api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY not configured")

    return OnlineLLMs(
        name="gemini",
        api_key=resolved_api_key,
        model_version=model_version or GEMINI_MODEL,
    )


def _build_query_prompt(query: str, retrieved_data: str) -> str:
    return (
        "Bạn là một trợ lý tư vấn AI thân thiện, chuyên nghiệp và đáng tin cậy.\n"
        "Bạn trả lời câu hỏi của người dùng dựa trên tài liệu đã được hệ thống cung cấp.\n"
        "Tài liệu có thể thuộc nhiều lĩnh vực khác nhau như bệnh viện, y tế, giáo dục, "
        "doanh nghiệp, sản phẩm, dịch vụ, quy trình, chính sách hoặc chăm sóc khách hàng.\n\n"

        "Mục tiêu của bạn là giúp người dùng hiểu đúng thông tin trong tài liệu, "
        "trả lời tự nhiên như một nhân viên tư vấn, không phải như một hệ thống trích dẫn học thuật.\n\n"

        "QUY TẮC TRẢ LỜI:\n"
        "- Chỉ trả lời dựa trên Reference data.\n"
        "- Không tự thêm thông tin ngoài tài liệu.\n"
        "- Không bịa số liệu, chính sách, giá, lịch, tên người, quy trình hoặc kết luận.\n"
        "- Không hiển thị mã nguồn/trích dẫn dạng [1], [2], [3] trong câu trả lời.\n"
        "- Không nhắc 'Reference data', 'block dữ liệu', 'marker' hoặc 'theo nguồn [1]' trong câu trả lời.\n"
        "- Nếu tài liệu chưa đủ thông tin, hãy nói: "
        "\"Hiện tài liệu chưa cung cấp đủ thông tin để trả lời chính xác câu hỏi này.\"\n\n"

        "CÁCH TRÌNH BÀY:\n"
        "- Trả lời bằng tiếng Việt.\n"
        "- Tự nhiên, dễ hiểu, lịch sự.\n"
        "- Nếu câu hỏi cần hướng dẫn, hãy trả lời theo từng bước.\n"
        "- Nếu câu hỏi hỏi về điều kiện/quy định, hãy nêu rõ điều kiện áp dụng.\n"
        "- Nếu câu hỏi hỏi về dịch vụ/quy trình, hãy trả lời ngắn gọn nhưng đủ ý.\n"
        "- Nếu câu hỏi liên quan đến y tế, chỉ cung cấp thông tin tham khảo từ tài liệu; "
        "không chẩn đoán, không kê đơn và không thay thế tư vấn của bác sĩ.\n\n"

        f"Câu hỏi của người dùng:\n{query}\n\n"
        f"Tài liệu tham chiếu:\n{retrieved_data}"
    )
