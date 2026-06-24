"""
Thin facade — re-exports public API từ các sub-module.

Giữ file này để main.py có thể import `from API_RAG_NEW import services`
và gọi `services.X()` mà không cần thay đổi.
"""
from __future__ import annotations

from API_RAG_NEW._services_shared import (  # noqa: F401
    health_payload,
    resolve_chunking_profile,
    runtime_config_payload,
    runtime_status_payload,
)
from API_RAG_NEW.collection_service import (  # noqa: F401
    create_collection,
    delete_collection,
    get_collection_info,
    get_collection_records,
    list_collections,
    update_collection,
)
from API_RAG_NEW.ingest_service import (  # noqa: F401
    delete_document,
    ingest_file_content,
    list_documents,
)
from API_RAG_NEW.query_service import (  # noqa: F401
    query_collection,
    query_collection_stream,
)
