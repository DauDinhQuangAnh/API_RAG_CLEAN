from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from API_RAG_NEW.config import RAG_INTERNAL_API_KEY


def require_internal_api_key(
    x_internal_api_key: str | None = Header(
        default=None,
        alias="X-Internal-API-Key",
    ),
) -> None:
    if not RAG_INTERNAL_API_KEY:
        return

    if not x_internal_api_key or not hmac.compare_digest(
        x_internal_api_key,
        RAG_INTERNAL_API_KEY,
    ):
        raise HTTPException(status_code=401, detail="Invalid internal API key.")
