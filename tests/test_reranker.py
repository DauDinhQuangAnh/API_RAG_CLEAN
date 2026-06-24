"""Tests cho reranker.py — LLM reranker và cross-encoder reranker."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from API_RAG_NEW.reranker import (
    _build_rerank_prompt,
    _extract_json_text,
    _parse_ranked_ids,
    rerank_candidate_ids,
)


def _make_candidate(candidate_id: str, text: str = "Nội dung") -> MagicMock:
    c = MagicMock()
    c.id = candidate_id
    c.document = text
    c.metadata = {"source": "test.pdf", "chunk_index": 1}
    return c


class TestParseRankedIds:
    def test_parses_ranked_ids_dict(self):
        result = _parse_ranked_ids('{"ranked_ids": ["id1", "id2"]}')
        assert result == ["id1", "id2"]

    def test_parses_list_directly(self):
        result = _parse_ranked_ids('["id1", "id2", "id3"]')
        assert result == ["id1", "id2", "id3"]

    def test_parses_alternative_keys(self):
        result = _parse_ranked_ids('{"ids": ["a", "b"]}')
        assert result == ["a", "b"]

    def test_handles_fenced_json(self):
        text = '```json\n{"ranked_ids": ["x1"]}\n```'
        result = _parse_ranked_ids(text)
        assert result == ["x1"]

    def test_returns_empty_on_invalid_json(self):
        result = _parse_ranked_ids("not json at all")
        assert result == []


class TestExtractJsonText:
    def test_passthrough_valid_json(self):
        assert _extract_json_text('{"a": 1}') == '{"a": 1}'

    def test_extracts_from_fenced_block(self):
        text = "Some text\n```json\n{\"key\": \"val\"}\n```\nMore text"
        result = _extract_json_text(text)
        assert result == '{"key": "val"}'

    def test_extracts_json_from_mixed_text(self):
        text = 'Here is the result: {"ranked_ids": ["a"]} done'
        result = _extract_json_text(text)
        assert '"ranked_ids"' in result


class TestBuildRerankPrompt:
    def test_includes_query_in_prompt(self):
        candidates = [_make_candidate("id1", "Đây là nội dung")]
        prompt = _build_rerank_prompt("câu hỏi tiếng Việt", candidates, 1)
        assert "câu hỏi tiếng Việt" in prompt

    def test_includes_candidate_id(self):
        candidates = [_make_candidate("abc123")]
        prompt = _build_rerank_prompt("query", candidates, 1)
        assert "abc123" in prompt

    def test_includes_final_n(self):
        candidates = [_make_candidate("id1")]
        prompt = _build_rerank_prompt("q", candidates, 5)
        assert "5" in prompt


class TestRerankCandidateIds:
    def test_uses_llm_response(self):
        llm = MagicMock()
        llm.generate_content.return_value = '{"ranked_ids": ["id2", "id1"]}'
        candidates = [_make_candidate("id1"), _make_candidate("id2")]
        result = rerank_candidate_ids("query", candidates, 2, llm)
        assert result[0] == "id2"

    def test_falls_back_on_llm_failure(self):
        llm = MagicMock()
        llm.generate_content.side_effect = Exception("API error")
        candidates = [_make_candidate("id1"), _make_candidate("id2")]
        result = rerank_candidate_ids("query", candidates, 2, llm)
        # Fallback: should return original order
        assert set(result) == {"id1", "id2"}

    def test_only_known_ids_returned(self):
        llm = MagicMock()
        llm.generate_content.return_value = '{"ranked_ids": ["fake_id", "id1"]}'
        candidates = [_make_candidate("id1")]
        result = rerank_candidate_ids("query", candidates, 2, llm)
        assert "fake_id" not in result
        assert "id1" in result

    def test_respects_final_n_limit(self):
        llm = MagicMock()
        llm.generate_content.return_value = '{"ranked_ids": ["id1", "id2", "id3"]}'
        candidates = [_make_candidate(f"id{i}") for i in range(1, 4)]
        result = rerank_candidate_ids("query", candidates, 2, llm)
        assert len(result) <= 2

    def test_empty_candidates_returns_empty(self):
        llm = MagicMock()
        result = rerank_candidate_ids("query", [], 5, llm)
        assert result == []
