"""Token-aware chunking for prose and structured Docling tables.

Prose blocks become semantic paragraph chunks. Tables are serialized row by row
as ``header: value`` pairs; Markdown syntax is intentionally not the retrieval
representation because it adds noise and weakens the column/value relationship.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Literal

from src.config import settings
from src.ingestion.loaders import ContentBlock, LoadedDoc, StructuredTableBlock

log = logging.getLogger(__name__)

ChunkType = Literal["paragraph", "table"]
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    source: str
    title: str
    section: str
    page_start: int
    page_end: int
    text: str
    n_tokens: int
    chunk_type: ChunkType = "paragraph"
    table_caption: str = ""
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[dict[str, str]] = field(default_factory=list)
    table_row_start: int = 0
    table_row_end: int = 0

    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        section = f", {self.section}" if self.section else ""
        return f"{self.source}{section}, {pages}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.chunk_type == "paragraph":
            for key in (
                "table_caption",
                "table_headers",
                "table_rows",
                "table_row_start",
                "table_row_end",
            ):
                payload.pop(key)
        return payload


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class _Unit:
    text: str
    page_start: int
    page_end: int
    section: str
    n_tokens: int


@dataclass
class _TableUnit:
    text: str
    row_number: int
    row: dict[str, str]


def _split_text_block(block: ContentBlock, max_tokens: int) -> list[_Unit]:
    n_tokens = count_tokens(block.text)
    if n_tokens <= max_tokens:
        return [
            _Unit(
                block.text,
                block.page_start,
                block.page_end,
                block.section,
                n_tokens,
            )
        ]

    units: list[_Unit] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for sentence in _SENT_RE.split(block.text):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)
        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            units.append(
                _Unit(
                    " ".join(buffer),
                    block.page_start,
                    block.page_end,
                    block.section,
                    buffer_tokens,
                )
            )
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += sentence_tokens
    if buffer:
        units.append(
            _Unit(
                " ".join(buffer),
                block.page_start,
                block.page_end,
                block.section,
                buffer_tokens,
            )
        )
    return units


def _overlap_tail(units: list[_Unit], budget: int) -> list[_Unit]:
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.n_tokens > budget:
            break
        tail.insert(0, unit)
        total += unit.n_tokens
    return tail


def _table_context(block: StructuredTableBlock) -> list[str]:
    context = []
    if block.section:
        context.append(f"Section: {block.section}")
    if block.caption:
        context.append(f"Table: {block.caption}")
    return context


def _render_row(prefix: list[str], field_lines: list[str]) -> str:
    return "\n".join([*prefix, *field_lines])


def _split_field_value(
    header: str,
    value: str,
    prefix: list[str],
    max_tokens: int,
) -> list[str]:
    """Split one very long cell while repeating row identity and field name."""
    words = value.split()
    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        rendered = _render_row(prefix, [f"{header}: {candidate}"])
        if current and count_tokens(rendered) > max_tokens:
            pieces.append(_render_row(prefix, [f"{header}: {' '.join(current)}"]))
            current = []
        current.append(word)
    if current:
        pieces.append(_render_row(prefix, [f"{header}: {' '.join(current)}"]))
    return pieces


def _row_units(
    block: StructuredTableBlock,
    row: dict[str, str],
    row_number: int,
    max_tokens: int,
) -> list[_TableUnit]:
    pairs = [(header, row.get(header, "")) for header in block.table_headers]
    pairs = [(header, value) for header, value in pairs if value]
    context = _table_context(block)
    row_label = f"Row: {row_number}"
    all_lines = [f"{header}: {value}" for header, value in pairs]
    complete = _render_row([*context, row_label], all_lines)
    if count_tokens(complete) <= max_tokens:
        return [_TableUnit(complete, row_number, row)]

    # Repeat one short identifier (usually ID/control/name) across fragments so
    # a long description never becomes detached from the row it belongs to.
    identity_pair = next(
        (
            pair
            for pair in pairs
            if count_tokens(f"{pair[0]}: {pair[1]}") <= min(64, max_tokens // 4)
        ),
        None,
    )
    identity_line = f"{identity_pair[0]}: {identity_pair[1]}" if identity_pair else ""
    prefix = [*context, row_label]
    if identity_line:
        prefix.append(identity_line)

    remaining = [pair for pair in pairs if pair != identity_pair]
    if not remaining:
        remaining = pairs
        prefix = [*context, row_label]

    units: list[_TableUnit] = []
    pending: list[str] = []
    for header, value in remaining:
        line = f"{header}: {value}"
        candidate = _render_row(prefix, [*pending, line])
        if count_tokens(candidate) <= max_tokens:
            pending.append(line)
            continue

        if pending:
            units.append(_TableUnit(_render_row(prefix, pending), row_number, row))
            pending = []

        if count_tokens(_render_row(prefix, [line])) <= max_tokens:
            pending.append(line)
            continue

        for piece in _split_field_value(header, value, prefix, max_tokens):
            units.append(_TableUnit(piece, row_number, row))

    if pending:
        units.append(_TableUnit(_render_row(prefix, pending), row_number, row))

    if not units:
        # Degenerate all-empty rows are normally removed by the loader. This is
        # a safe fallback for callers constructing StructuredTableBlock directly.
        units.append(_TableUnit(_render_row(prefix, all_lines), row_number, row))
    return units


def _table_units(block: StructuredTableBlock, max_tokens: int) -> list[_TableUnit]:
    units: list[_TableUnit] = []
    for row_number, row in enumerate(block.table_rows, start=1):
        units.extend(_row_units(block, row, row_number, max_tokens))

    if units:
        return units

    context = _table_context(block)
    if block.table_headers:
        context.append(f"Columns: {', '.join(block.table_headers)}")
    elif block.text:
        context.append(f"Table content: {block.text}")
    return [_TableUnit("\n".join(context), 0, {})] if context else []


def chunk_document(doc: LoadedDoc) -> list[Chunk]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_tokens = 0

    def append_chunk(
        text: str,
        chunk_type: ChunkType,
        section: str,
        page_start: int,
        page_end: int,
        *,
        table_block: StructuredTableBlock | None = None,
        table_unit: _TableUnit | None = None,
    ) -> None:
        chunk_index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{chunk_index:05d}",
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                source=doc.source,
                title=doc.title,
                section=section,
                page_start=page_start,
                page_end=page_end,
                text=text,
                n_tokens=count_tokens(text),
                chunk_type=chunk_type,
                table_caption=table_block.caption if table_block else "",
                table_headers=list(table_block.table_headers) if table_block else [],
                table_rows=[dict(table_unit.row)] if table_unit and table_unit.row else [],
                table_row_start=table_unit.row_number if table_unit else 0,
                table_row_end=table_unit.row_number if table_unit else 0,
            )
        )

    def flush_text(keep_overlap: bool = True) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        append_chunk(
            text="\n\n".join(unit.text for unit in current),
            chunk_type="paragraph",
            section=current[0].section,
            page_start=min(unit.page_start for unit in current),
            page_end=max(unit.page_end for unit in current),
        )
        tail = _overlap_tail(current, overlap) if keep_overlap else []
        current = list(tail)
        current_tokens = sum(unit.n_tokens for unit in tail)

    for block in doc.blocks:
        if isinstance(block, StructuredTableBlock):
            flush_text(keep_overlap=False)
            for table_unit in _table_units(block, size):
                append_chunk(
                    text=table_unit.text,
                    chunk_type="table",
                    section=block.section,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    table_block=block,
                    table_unit=table_unit,
                )
            continue

        for unit in _split_text_block(block, size):
            section_changed = bool(current) and unit.section != current[-1].section
            if section_changed:
                flush_text(keep_overlap=False)
            elif current and current_tokens + unit.n_tokens > size:
                flush_text()
            current.append(unit)
            current_tokens += unit.n_tokens

    flush_text(keep_overlap=False)
    return chunks


def chunk_corpus(docs: list[LoadedDoc]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc)
        paragraph_count = sum(chunk.chunk_type == "paragraph" for chunk in chunks)
        table_count = sum(chunk.chunk_type == "table" for chunk in chunks)
        log.info(
            "Chunked %s -> %d paragraph + %d table row chunks",
            doc.source,
            paragraph_count,
            table_count,
        )
        all_chunks.extend(chunks)
    if all_chunks:
        sizes = [chunk.n_tokens for chunk in all_chunks]
        log.info(
            "Corpus: %d chunks, tokens min/mean/max = %d/%d/%d",
            len(all_chunks),
            min(sizes),
            sum(sizes) // len(sizes),
            max(sizes),
        )
    return all_chunks
