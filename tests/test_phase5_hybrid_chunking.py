from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from API_RAG_NEW import services
from API_RAG_NEW.document_structure import (
    BLOCK_TABLE_ROW,
    build_logical_blocks,
    clean_table_title,
    detect_heading,
    detect_table_caption,
    is_bullet,
    table_to_logical_blocks,
)
from API_RAG_NEW.rag_pipeline import (
    add_records_to_collection,
    format_retrieved_data_with_markers,
)


class IdentityChunker:
    def split_text(self, text: str) -> list[str]:
        return [text.strip()] if text.strip() else []


class PipeChunker:
    def split_text(self, text: str) -> list[str]:
        return [chunk.strip() for chunk in text.split("||") if chunk.strip()]


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = None
        self.records: dict[str, dict[str, object]] = {}

    def upsert(self, ids, embeddings, metadatas, documents):
        for record_id, embedding, metadata, document in zip(
            ids,
            embeddings,
            metadatas,
            documents,
        ):
            self.records[record_id] = {
                "embedding": embedding,
                "metadata": metadata,
                "document": document,
            }

    def count(self) -> int:
        return len(self.records)


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata=None) -> FakeCollection:
        collection = self.collections.setdefault(name, FakeCollection(name))
        collection.metadata = metadata
        return collection

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections[name]


def test_heading_caption_and_bullet_detection_supports_vi_and_en():
    roman = "II. Th\u00f4ng tin v\u1ec1 \u00dd t\u01b0\u1edfng/D\u1ef1 \u00e1n"
    numbered = "5. Gi\u1ea3i ph\u00e1p \u0111\u1ec1 gi\u1ea3i quy\u1ebft v\u1ea5n \u0111\u1ec1"
    nested = "5.1 Ph\u1ea1m vi tri\u1ec3n khai"
    caption = (
        "B\u1ea3ng 2. \u0110i\u1ec3m kh\u00e1c bi\u1ec7t so v\u1edbi "
        "gi\u1ea3i ph\u00e1p hi\u1ec7n c\u00f3"
    )

    assert detect_heading(roman).kind == "roman"
    assert detect_heading(numbered).kind == "numbered"
    assert detect_heading(nested).level == 2
    assert detect_table_caption(caption) == caption
    assert detect_table_caption("Table 1. Delivery milestones") is not None
    assert is_bullet("\u2022 Noi dung gach dau dong")
    assert is_bullet("- Noi dung gach dau dong")


def test_section_path_inherits_and_tracks_nested_headings():
    roman = "II. Th\u00f4ng tin v\u1ec1 \u00dd t\u01b0\u1edfng/D\u1ef1 \u00e1n"
    numbered = "5. Gi\u1ea3i ph\u00e1p \u0111\u1ec1 gi\u1ea3i quy\u1ebft v\u1ea5n \u0111\u1ec1"
    nested = "5.1 Ki\u1ebfn tr\u00fac gi\u1ea3i ph\u00e1p"
    blocks = build_logical_blocks(
        "\n".join(
            [
                roman,
                numbered,
                "Doan van trong muc 5.",
                nested,
                "Doan van trong muc 5.1.",
            ]
        )
    )

    first_paragraph = blocks[2]
    nested_paragraph = blocks[4]

    assert first_paragraph.section_title == numbered
    assert first_paragraph.section_path == f"{roman} > {numbered}"
    assert nested_paragraph.section_title == nested
    assert nested_paragraph.section_path == f"{roman} > {numbered} > {nested}"


def test_table_row_conversion_uses_headers_and_skips_empty_rows():
    title = (
        "B\u1ea3ng 2. \u0110i\u1ec3m kh\u00e1c bi\u1ec7t so v\u1edbi "
        "gi\u1ea3i ph\u00e1p hi\u1ec7n c\u00f3"
    )
    blocks = table_to_logical_blocks(
        [
            ["Ti\u00eau ch\u00ed", "C\u00e1ch l\u00e0m hi\u1ec7n t\u1ea1i", "Weave Carbon"],
            [
                "\u0110\u1ed9 tin c\u1eady",
                "Thi\u1ebfu d\u1ea5u v\u1ebft ki\u1ec3m to\u00e1n",
                "M\u00e3 b\u0103m SHA-256 t\u1ea1o Audit Trail",
            ],
            ["", "", ""],
            ["---", "", ""],
        ],
        table_index=2,
        table_title=title,
        page_number=4,
        section_title="5. Gi\u1ea3i ph\u00e1p",
        section_path="II. D\u1ef1 \u00e1n > 5. Gi\u1ea3i ph\u00e1p",
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block.block_type == BLOCK_TABLE_ROW
    assert block.table_title == title
    assert block.table_row_index == 1
    assert "Ti\u00eau ch\u00ed: \u0110\u1ed9 tin c\u1eady" in block.text
    assert "Weave Carbon: M\u00e3 b\u0103m SHA-256" in block.text


def test_clean_table_title_trims_flattened_headers_and_caps_length():
    first = (
        "B\u1ea3ng 2. \u0110i\u1ec3m kh\u00e1c bi\u1ec7t so v\u1edbi "
        "gi\u1ea3i ph\u00e1p hi\u1ec7n c\u00f3 C\u00e1ch l\u00e0m hi\u1ec7n "
        "t\u1ea1i/gi\u1ea3i ph\u00e1p Ti\u00eau ch\u00ed Weave Carbon kh\u00e1c "
        "T\u1eadp trung d\u1eef li\u1ec7u..."
    )
    second = (
        "B\u1ea3ng 1. Value Proposition Canvas c\u1ee7a Weave Carbon "
        "Th\u00e0nh ph\u1ea7n N\u1ed9i dung Customer Jobs..."
    )
    very_long = "Table 9. " + ("Long title words " * 20)

    assert clean_table_title(first) == (
        "B\u1ea3ng 2. \u0110i\u1ec3m kh\u00e1c bi\u1ec7t so v\u1edbi "
        "gi\u1ea3i ph\u00e1p hi\u1ec7n c\u00f3"
    )
    assert clean_table_title(second) == (
        "B\u1ea3ng 1. Value Proposition Canvas c\u1ee7a Weave Carbon"
    )
    assert len(clean_table_title(very_long)) <= 120


def test_hybrid_pdf_records_include_old_and_phase5_metadata():
    records = list(
        services._iter_hybrid_pdf_chunk_records(
            [
                {
                    "page_number": 2,
                    "text": "\n".join(
                        [
                            "II. Project",
                            "5. Solution",
                            "Paragraph inside the solution section.",
                        ]
                    ),
                    "tables": [],
                }
            ],
            "demo.pdf",
            ".pdf",
            "a" * 64,
            IdentityChunker(),
        )
    )

    assert len(records) == 1
    record = records[0]
    assert {"doc_id", "source", "source_type", "chunk", "chunk_index"}.issubset(record)
    assert record["page_number"] == 2
    assert record["page_chunk_index"] == 1
    assert record["chunk_type"] == "section_child"
    assert record["section_title"] == "5. Solution"
    assert record["section_path"] == "II. Project > 5. Solution"
    assert record["parent_id"].startswith("parent_")


def test_hybrid_pdf_table_records_include_row_metadata_and_clean_title():
    table_title = (
        "B\u1ea3ng 2. So s\u00e1nh gi\u1ea3i ph\u00e1p C\u00e1ch l\u00e0m "
        "hi\u1ec7n t\u1ea1i Ti\u00eau ch\u00ed Weave Carbon"
    )
    cleaned_title = "B\u1ea3ng 2. So s\u00e1nh gi\u1ea3i ph\u00e1p"
    records = list(
        services._iter_hybrid_pdf_chunk_records(
            [
                {
                    "page_number": 4,
                    "text": "\n".join(["II. Project", "5. Solution", table_title]),
                    "tables": [
                        {
                            "table_index": 1,
                            "rows": [
                                ["Criteria", "Current", "Weave Carbon"],
                                ["Reliability", "No audit trail", "SHA-256 audit"],
                            ],
                        }
                    ],
                }
            ],
            "demo.pdf",
            ".pdf",
            "b" * 64,
            IdentityChunker(),
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record["chunk_type"] == "table_row"
    assert record["table_index"] == 1
    assert record["table_title"] == cleaned_title
    assert record["table_row_index"] == 1
    assert record["page_number"] == 4
    assert record["section_path"] == "II. Project > 5. Solution"
    assert f"Table: {cleaned_title}" in record["chunk"]
    assert "Criteria: Reliability" in record["chunk"]


def test_oversized_table_row_chunks_split_with_part_metadata():
    long_a = "A " * 700
    long_b = "B " * 700
    long_c = "C " * 700
    records = list(
        services._iter_hybrid_pdf_chunk_records(
            [
                {
                    "page_number": 3,
                    "text": "Table 1. Long row details",
                    "tables": [
                        {
                            "table_index": 1,
                            "rows": [
                                ["First", "Second", "Third"],
                                [long_a, long_b, long_c],
                            ],
                        }
                    ],
                }
            ],
            "demo.pdf",
            ".pdf",
            "d" * 64,
            IdentityChunker(),
        )
    )

    assert len(records) >= 2
    assert all(record["chunk_type"] == "table_row" for record in records)
    assert {record["table_row_index"] for record in records} == {1}
    assert [record["table_row_part_index"] for record in records] == list(
        range(1, len(records) + 1)
    )
    assert all(record["table_index"] == 1 for record in records)
    assert all(record["table_title"] == "Table 1. Long row details" for record in records)
    assert all(len(record["chunk"]) <= 2000 for record in records)


def test_hybrid_pdf_skips_obvious_flattened_table_semantic_but_keeps_paragraph():
    cleanup_stats = {"skipped_flattened_table_chunks": 0}
    records = list(
        services._iter_hybrid_pdf_chunk_records(
            [
                {
                    "page_number": 4,
                    "text": "\n".join(
                        [
                            "II. Project",
                            "5. Solution",
                            "Project summary explains market, business model, and solution. || Criteria Current Weave Carbon Reliability No audit trail SHA-256 audit",
                            "Table 1. Solution comparison",
                        ]
                    ),
                    "tables": [
                        {
                            "table_index": 1,
                            "rows": [
                                ["Criteria", "Current", "Weave Carbon"],
                                ["Reliability", "No audit trail", "SHA-256 audit"],
                            ],
                        }
                    ],
                }
            ],
            "demo.pdf",
            ".pdf",
            "e" * 64,
            PipeChunker(),
            cleanup_stats=cleanup_stats,
        )
    )

    semantic_chunks = [
        record["chunk"] for record in records if record["chunk_type"] != "table_row"
    ]
    table_chunks = [
        record["chunk"] for record in records if record["chunk_type"] == "table_row"
    ]
    assert cleanup_stats["skipped_flattened_table_chunks"] == 1
    assert any("Project summary explains market" in chunk for chunk in semantic_chunks)
    assert not any("Criteria Current Weave Carbon Reliability" in chunk for chunk in semantic_chunks)
    assert table_chunks


def test_ingest_chunk_stats_reports_skipped_flattened_table_chunks(monkeypatch):
    fake_client = FakeChromaClient()
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "ProtonxSemanticChunker", lambda model: PipeChunker())
    monkeypatch.setattr(services, "RAG_CHUNKING_PROFILE", "hybrid")
    monkeypatch.setattr(
        services,
        "_extract_pdf_pages_with_tables",
        lambda raw_content: [
            {
                "page_number": 1,
                "text": "\n".join(
                    [
                        "I. Overview",
                        "Normal paragraph about project summary. || Criteria Current Weave Carbon Reliability No audit trail SHA-256 audit",
                        "Table 1. Solution comparison",
                    ]
                ),
                "tables": [
                    {
                        "table_index": 1,
                        "rows": [
                            ["Criteria", "Current", "Weave Carbon"],
                            ["Reliability", "No audit trail", "SHA-256 audit"],
                        ],
                    }
                ],
            }
        ],
    )

    response = services.ingest_file_content(
        "demo.pdf",
        b"patched extractor",
        "phase5-cleanup-stats",
    )

    assert response.chunk_stats["skipped_flattened_table_chunks"] == 1
    assert response.chunk_stats["table_chunks"] == 1
    assert response.chunk_stats["semantic_chunks"] == 1


def test_semantic_profile_ingest_preserves_old_metadata(monkeypatch):
    fake_client = FakeChromaClient()
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "ProtonxSemanticChunker", lambda model: IdentityChunker())
    monkeypatch.setattr(services, "RAG_CHUNKING_PROFILE", "semantic")

    response = services.ingest_file_content(
        "demo.txt",
        b"plain semantic text",
        "phase5-semantic",
    )

    collection = fake_client.get_collection("phase5-semantic")
    metadata = next(iter(collection.records.values()))["metadata"]
    assert response.chunking_profile == "semantic"
    assert response.warnings == []
    assert response.chunks == 1
    assert metadata["chunk"] == "plain semantic text"
    assert "chunk_type" not in metadata
    assert "parent_id" not in metadata


def test_hybrid_failure_falls_back_to_semantic_with_warning(monkeypatch):
    fake_client = FakeChromaClient()
    monkeypatch.setattr(services, "CHROMA_CLIENT", fake_client)
    monkeypatch.setattr(services, "EMBEDDING_MODEL", FakeEmbeddingModel())
    monkeypatch.setattr(services, "ProtonxSemanticChunker", lambda model: IdentityChunker())
    monkeypatch.setattr(services, "RAG_CHUNKING_PROFILE", "hybrid")
    monkeypatch.setattr(
        services,
        "_extract_pdf_pages_with_tables",
        lambda raw_content: (_ for _ in ()).throw(RuntimeError("table boom")),
    )
    monkeypatch.setattr(
        services,
        "_extract_pdf_pages",
        lambda raw_content: [(1, "fallback semantic chunk")],
    )

    response = services.ingest_file_content(
        "demo.pdf",
        b"not a real pdf because extractors are patched",
        "phase5-fallback",
    )

    collection = fake_client.get_collection("phase5-fallback")
    metadata = next(iter(collection.records.values()))["metadata"]
    assert response.chunking_profile == "semantic"
    assert response.chunks == 1
    assert response.warnings
    assert "Hybrid chunking failed; fell back to semantic chunking" in response.warnings[0]
    assert metadata["chunk"] == "fallback semantic chunk"
    assert "chunk_type" not in metadata


def test_hybrid_ids_are_deterministic_and_upsert_compatible():
    first = list(
        services._iter_hybrid_document_chunk_records(
            "I. Overview\nalpha || beta",
            "demo.txt",
            ".txt",
            "c" * 64,
            PipeChunker(),
        )
    )
    second = list(
        services._iter_hybrid_document_chunk_records(
            "I. Overview\nalpha || beta",
            "demo.txt",
            ".txt",
            "c" * 64,
            PipeChunker(),
        )
    )
    collection = FakeCollection("phase5-upsert")

    assert [record["id"] for record in first] == [record["id"] for record in second]
    add_records_to_collection(first, FakeEmbeddingModel(), collection)
    add_records_to_collection(second, FakeEmbeddingModel(), collection)
    assert collection.count() == len(first)


def test_retrieved_data_adds_phase5_metadata_without_losing_markers():
    retrieved_data = format_retrieved_data_with_markers(
        [
            {
                "source": "document.pdf",
                "page_number": 4,
                "section_path": "II. Project > 5. Solution",
                "chunk_type": "table_row",
                "table_row_index": 3,
                "chunk_index": 8,
                "chunk": "table row chunk",
            }
        ],
        [],
    )

    assert "[1] source=document.pdf | page=4 | chunk_index=8" in retrieved_data
    assert "section=II. Project > 5. Solution" in retrieved_data
    assert "chunk_type=table_row" in retrieved_data
    assert "table_row=3" in retrieved_data
    assert "chunk: table row chunk" in retrieved_data


def test_records_modal_prefers_metadata_chunk_then_document_fallback():
    html = Path("api_test_ui.html").read_text(encoding="utf-8")

    assert "const documents = Array.isArray(payload.documents) ? payload.documents : [];" in html
    assert "const documentChunk = documents[index];" in html
    assert html.index("metadata && metadata.chunk") < html.index(": documentChunk")
    assert "chunking_profile" in html
    assert "chunk_stats" in html
    assert "skipped_flattened_table_chunks" in html
    assert "chunk_type" in html
    assert "section_path" in html
    assert "table_row_part_index" in html
    assert "rag_chunking_profile" in html
