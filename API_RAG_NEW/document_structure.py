from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence


BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_BULLET = "bullet"
BLOCK_TABLE_CAPTION = "table_caption"
BLOCK_TABLE_ROW = "table_row"
BLOCK_UNKNOWN = "unknown"


@dataclass(frozen=True)
class LogicalBlock:
    text: str
    block_type: str
    section_title: str | None = None
    section_path: str | None = None
    page_number: int | None = None
    block_index: int = 0
    table_index: int | None = None
    table_title: str | None = None
    table_row_index: int | None = None


@dataclass(frozen=True)
class HeadingInfo:
    title: str
    path_entry: str
    level: int
    kind: str


@dataclass(frozen=True)
class TableContext:
    table_title: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    block_index: int = 0


def normalize_whitespace(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def is_table_caption(line: str) -> bool:
    return detect_table_caption(line) is not None


def detect_table_caption(line: str) -> str | None:
    text = normalize_whitespace(line)
    if not text:
        return None

    normalized = _strip_accents(text)
    if re.match(r"^(?:bang|table)\s+\d+[.:)]?\s+\S", normalized):
        return text
    return None


def is_bullet(line: str) -> bool:
    text = normalize_whitespace(line)
    return bool(re.match(r"^(?:[\u2022\u25cf\u25cb\u25aa]|\-|\*|\+)\s+\S", text))


def detect_heading(line: str) -> HeadingInfo | None:
    text = normalize_whitespace(line)
    if not text or detect_table_caption(text):
        return None

    normalized = _strip_accents(text)

    roman_match = re.match(
        r"^(?P<label>[ivxlcdm]+)[.)]\s+(?P<title>\S.*)$",
        normalized,
    )
    if roman_match:
        return HeadingInfo(
            title=text,
            path_entry=text,
            level=1,
            kind="roman",
        )

    appendix_match = re.match(
        r"^(?P<label>phu\s+luc|appendix)\b[.:)\-\s]*(?P<title>.*)$",
        normalized,
    )
    if appendix_match:
        return HeadingInfo(
            title=text,
            path_entry=text,
            level=1,
            kind="appendix",
        )

    nested_match = re.match(
        r"^(?P<label>\d+\.\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>\S.*)$",
        normalized,
    )
    if nested_match:
        label = nested_match.group("label")
        return HeadingInfo(
            title=text,
            path_entry=text,
            level=len(label.split(".")),
            kind="numbered",
        )

    numbered_match = re.match(
        r"^(?P<label>\d+)[.)]\s+(?P<title>\S.*)$",
        normalized,
    )
    if numbered_match:
        return HeadingInfo(
            title=text,
            path_entry=text,
            level=1,
            kind="numbered",
        )

    return None


def build_logical_blocks(
    text: str,
    *,
    page_number: int | None = None,
    start_block_index: int = 1,
) -> list[LogicalBlock]:
    blocks: list[LogicalBlock] = []
    section_stack: list[tuple[int, str]] = []
    top_heading_kind: str | None = None
    block_index = start_block_index

    for line in _iter_lines(text):
        caption = detect_table_caption(line)
        if caption:
            blocks.append(
                LogicalBlock(
                    text=caption,
                    block_type=BLOCK_TABLE_CAPTION,
                    section_title=_current_section_title(section_stack),
                    section_path=_current_section_path(section_stack),
                    page_number=page_number,
                    block_index=block_index,
                    table_title=caption,
                )
            )
            block_index += 1
            continue

        heading = detect_heading(line)
        if heading:
            effective_level = _effective_heading_level(heading, top_heading_kind)
            section_stack = [
                item for item in section_stack if item[0] < effective_level
            ]
            section_stack.append((effective_level, heading.path_entry))
            if effective_level == 1:
                top_heading_kind = heading.kind

            blocks.append(
                LogicalBlock(
                    text=line,
                    block_type=BLOCK_HEADING,
                    section_title=heading.path_entry,
                    section_path=_current_section_path(section_stack),
                    page_number=page_number,
                    block_index=block_index,
                )
            )
            block_index += 1
            continue

        block_type = BLOCK_BULLET if is_bullet(line) else BLOCK_PARAGRAPH
        blocks.append(
            LogicalBlock(
                text=line,
                block_type=block_type,
                section_title=_current_section_title(section_stack),
                section_path=_current_section_path(section_stack),
                page_number=page_number,
                block_index=block_index,
            )
        )
        block_index += 1

    return blocks


def table_contexts_from_text(text: str, page_number: int | None = None) -> list[TableContext]:
    blocks = build_logical_blocks(text, page_number=page_number)
    contexts = [
        TableContext(
            table_title=block.table_title or block.text,
            section_title=block.section_title,
            section_path=block.section_path,
            block_index=block.block_index,
        )
        for block in blocks
        if block.block_type == BLOCK_TABLE_CAPTION
    ]
    if contexts:
        return contexts

    last_context_block = next(
        (
            block
            for block in reversed(blocks)
            if block.section_title or block.section_path
        ),
        None,
    )
    if last_context_block is None:
        return []

    return [
        TableContext(
            section_title=last_context_block.section_title,
            section_path=last_context_block.section_path,
            block_index=last_context_block.block_index,
        )
    ]


def table_to_logical_blocks(
    table: Sequence[Sequence[Any]],
    *,
    table_index: int,
    table_title: str | None = None,
    page_number: int | None = None,
    section_title: str | None = None,
    section_path: str | None = None,
    start_block_index: int = 1,
) -> list[LogicalBlock]:
    rows = [_normalize_table_row(row) for row in table or []]
    rows = [row for row in rows if _is_meaningful_row(row)]
    if not rows:
        return []

    header: list[str] | None = None
    data_rows = rows
    if _looks_like_header(rows[0], rows[1:]):
        header = rows[0]
        data_rows = rows[1:]

    blocks: list[LogicalBlock] = []
    table_row_index = 0
    block_index = start_block_index
    for row in data_rows:
        if not _is_meaningful_row(row):
            continue
        table_row_index += 1
        row_text = _format_table_row_text(
            row,
            header=header,
            row_index=table_row_index,
            table_title=table_title,
        )
        if not row_text:
            continue
        blocks.append(
            LogicalBlock(
                text=row_text,
                block_type=BLOCK_TABLE_ROW,
                section_title=section_title,
                section_path=section_path,
                page_number=page_number,
                block_index=block_index,
                table_index=table_index,
                table_title=table_title,
                table_row_index=table_row_index,
            )
        )
        block_index += 1

    return blocks


def stable_parent_id(
    doc_id: str,
    source: str,
    page_number: int | None,
    section_path: str | None,
    block_index: int | None,
    parent_text: str | None,
) -> str:
    hasher = hashlib.sha256()
    for part in (
        doc_id,
        source,
        page_number or "",
        section_path or "",
        block_index or "",
        normalize_whitespace(parent_text or ""),
    ):
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return f"parent_{hasher.hexdigest()[:32]}"


def _iter_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = normalize_whitespace(raw_line)
        if line:
            lines.append(line)
    return lines


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return without_marks.casefold()


def _effective_heading_level(
    heading: HeadingInfo,
    top_heading_kind: str | None,
) -> int:
    if heading.kind == "numbered" and top_heading_kind in {"roman", "appendix"}:
        return heading.level + 1
    return heading.level


def _current_section_title(section_stack: list[tuple[int, str]]) -> str | None:
    return section_stack[-1][1] if section_stack else None


def _current_section_path(section_stack: list[tuple[int, str]]) -> str | None:
    if not section_stack:
        return None
    return " > ".join(entry for _, entry in section_stack)


def _normalize_table_row(row: Sequence[Any]) -> list[str]:
    return [normalize_whitespace(cell) for cell in row]


def _is_meaningful_row(row: Sequence[str]) -> bool:
    joined = " ".join(cell for cell in row if cell)
    return any(char.isalnum() for char in joined)


def _looks_like_header(row: Sequence[str], following_rows: Sequence[Sequence[str]]) -> bool:
    non_empty = [cell for cell in row if cell]
    if len(non_empty) < 2 or not following_rows:
        return False
    if all(_looks_numeric(cell) for cell in non_empty):
        return False
    return True


def _looks_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"[\d.,%+\-\s]+", value or ""))


def _format_table_row_text(
    row: Sequence[str],
    *,
    header: Sequence[str] | None,
    row_index: int,
    table_title: str | None,
) -> str:
    lines = [f"Table: {table_title or 'N/A'}", f"Row: {row_index}"]
    for index, cell in enumerate(row):
        if not cell:
            continue
        label = _header_label(header, index)
        lines.append(f"{label}: {cell}")
    return "\n".join(lines).strip()


def _header_label(header: Sequence[str] | None, index: int) -> str:
    if header and index < len(header) and header[index]:
        return header[index]
    return f"Column {index + 1}"
