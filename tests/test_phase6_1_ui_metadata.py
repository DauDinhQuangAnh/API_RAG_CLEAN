from __future__ import annotations

from pathlib import Path


def test_local_ui_exposes_phase6_runtime_and_metadata_views():
    html = Path("api_test_ui.html").read_text(encoding="utf-8")

    assert "modeMetadataBtn" in html
    assert "formatRetrievedMetadata" in html
    assert "finalRetrievedMetadatas" in html
    assert "recordsModalStats" in html
    assert "renderRecordsStats" in html
    assert "rag_max_context_expansion_per_candidate" in html
    assert "rag_max_total_candidates" in html
    assert "rag_enable_distance_guard" in html
    assert "rag_max_distance" in html
    assert "parent_id" in html
    assert "table_row_part_index" in html


def test_local_ui_keeps_single_query_pipeline_without_search_mode_control():
    html = Path("api_test_ui.html").read_text(encoding="utf-8")

    assert "/collections/${collectionName}/query" in html
    assert "search_mode" not in html
    assert "searchMode" not in html
    assert "hybrid search" not in html.casefold()
