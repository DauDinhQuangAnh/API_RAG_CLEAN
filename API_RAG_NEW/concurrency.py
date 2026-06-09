from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Callable, Iterator

from fastapi import HTTPException

from API_RAG_NEW.config import (
    RAG_LLM_QUEUE_TIMEOUT_SECONDS,
    RAG_MAX_CONCURRENT_LLM_CALLS,
    RAG_MAX_CONCURRENT_QUERIES,
    RAG_QUERY_QUEUE_TIMEOUT_SECONDS,
)


class LLMOverloadedError(RuntimeError):
    pass


class SlotLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self._lock = threading.Lock()
        self._active = 0
        self._rejected = 0

    @contextmanager
    def acquire(
        self,
        *,
        timeout: float,
        error_factory: Callable[[], Exception],
    ) -> Iterator[None]:
        acquired = self._semaphore.acquire(timeout=max(0.0, float(timeout)))
        if not acquired:
            with self._lock:
                self._rejected += 1
            raise error_factory()

        with self._lock:
            self._active += 1

        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit": self.limit,
                "active": self._active,
                "available": max(0, self.limit - self._active),
                "rejected": self._rejected,
            }


# Limits are per process. With multiple Uvicorn workers, approximate total
# capacity is workers * limit; each worker also loads its own embedding model.
_query_limiter = SlotLimiter(RAG_MAX_CONCURRENT_QUERIES)
_llm_limiter = SlotLimiter(RAG_MAX_CONCURRENT_LLM_CALLS)


@contextmanager
def acquire_query_slot(timeout: float | None = None) -> Iterator[None]:
    with _query_limiter.acquire(
        timeout=(
            RAG_QUERY_QUEUE_TIMEOUT_SECONDS if timeout is None else timeout
        ),
        error_factory=lambda: HTTPException(
            status_code=503,
            detail="RAG server is busy. Please try again later.",
        ),
    ):
        yield


@contextmanager
def acquire_llm_slot(timeout: float | None = None) -> Iterator[None]:
    with _llm_limiter.acquire(
        timeout=RAG_LLM_QUEUE_TIMEOUT_SECONDS if timeout is None else timeout,
        error_factory=lambda: LLMOverloadedError("LLM server is busy."),
    ):
        yield


def concurrency_status_payload() -> dict[str, object]:
    return {
        "query": _query_limiter.snapshot(),
        "llm": _llm_limiter.snapshot(),
    }
