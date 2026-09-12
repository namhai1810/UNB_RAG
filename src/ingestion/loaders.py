"""Docling PDF conversion and structured block extraction.

The complete Markdown export is persisted for human inspection. Retrieval does
not have to reverse-engineer those Markdown tables: Docling ``TableItem`` data
is retained as typed headers, rows, and cells until the chunking stage.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel

from src.config import settings

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")
_NORM_RE = re.compile(r"[^a-z0-9]+")
_SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def normalize_heading(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def _clean(text: str) -> str:
    """Normalize text without flattening Markdown rows or paragraphs."""
    text = text.replace("\u00ad", "")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


@dataclass
class ContentBlock:
    """A prose block in Docling reading order."""

    text: str
    page_start: int
    page_end: int
    section: str = ""
    block_type: Literal["text"] = "text"


@dataclass(frozen=True)
class StructuredTableCell:
    """One Docling table cell, including span and semantic header flags."""

    text: str
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    is_column_header: bool = False
    is_row_header: bool = False
    is_row_section: bool = False


@dataclass
class StructuredTableBlock:
    """A table with both audit Markdown and retrieval-ready structure."""

    text: str
    page_start: int
    page_end: int
    section: str = ""
    caption: str = ""
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[dict[str, str]] = field(default_factory=list)
    table_cells: list[StructuredTableCell] = field(default_factory=list)
    block_type: Literal["table"] = "table"


DocumentBlock = ContentBlock | StructuredTableBlock


@dataclass
class LoadedDoc:
    doc_id: str
    source: str
    title: str
    blocks: list[DocumentBlock] = field(default_factory=list)
    markdown_path: Path | None = None

    @property
    def n_chars(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def n_tables(self) -> int:
        return sum(block.block_type == "table" for block in self.blocks)


def _doc_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = settings.docling_do_ocr
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode(settings.docling_table_mode)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def _item_pages(item) -> tuple[int, int]:
    pages = sorted(
        {
            int(prov.page_no)
            for prov in getattr(item, "prov", [])
            if getattr(prov, "page_no", None) is not None
        }
    )
    return (pages[0], pages[-1]) if pages else (0, 0)


def _repeated_chrome(document, items: list[tuple[object, int]]) -> set[str]:
    """Detect short text repeated on most pages when the PDF labels it poorly."""
    page_count = len(document.pages)
    if page_count < 3:
        return set()

    occurrences: set[tuple[str, int]] = set()
    for item, _ in items:
        text = _clean(getattr(item, "text", ""))
        page_start, page_end = _item_pages(item)
        if not text or page_start != page_end or len(text) > 160:
            continue
        normal = normalize_heading(text)
        if normal:
            occurrences.add((normal, page_start))

    page_frequency: Counter[str] = Counter(normal for normal, _ in occurrences)
    threshold = max(3, math.ceil(page_count * 0.5))
    return {normal for normal, count in page_frequency.items() if count >= threshold}


def _text_markdown(item) -> str:
    text = _clean(getattr(item, "text", ""))
    if not text:
        return ""
    if item.label == DocItemLabel.SECTION_HEADER:
        level = min(max(int(getattr(item, "level", 1)), 1), 6)
        return f"{'#' * level} {text}"
    if item.label == DocItemLabel.TITLE:
        return f"# {text}"
    if item.label == DocItemLabel.LIST_ITEM:
        marker = (getattr(item, "marker", "") or "-").strip()
        return f"{marker} {text}".strip()
    if item.label == DocItemLabel.CODE:
        language = getattr(item, "code_language", "") or ""
        return f"```{language}\n{text}\n```"
    if item.label == DocItemLabel.FORMULA:
        return f"$${text}$$"
    return text


def _unique_headers(columns) -> list[str]:
    """Produce stable, non-empty dict keys even for duplicate/absent headers."""
    counts: Counter[str] = Counter()
    headers: list[str] = []
    for index, column in enumerate(columns):
        base = "" if isinstance(column, int) else _clean(str(column))
        if not base or base.lower() in {"none", "nan"}:
            base = f"Column {index + 1}"
        counts[base] += 1
        headers.append(base if counts[base] == 1 else f"{base} ({counts[base]})")
    return headers


def _cell_value(value) -> str:
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _clean(str(value)) if value is not None else ""


def _grid_headers_and_rows(item) -> tuple[list[str], list[dict[str, str]]]:
    """Fallback for a malformed table that cannot be exported as a DataFrame."""
    grid = item.data.grid
    header_count = 0
    for row_index, row in enumerate(grid):
        if any(cell.column_header and cell.start_row_offset_idx == row_index for cell in row):
            header_count += 1
        else:
            break

    raw_headers: list[str] = []
    for column_index in range(item.data.num_cols):
        parts: list[str] = []
        for row_index in range(header_count):
            value = _clean(grid[row_index][column_index].text)
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        raw_headers.append(". ".join(parts))
    headers = _unique_headers(raw_headers)

    rows = []
    for grid_row in grid[header_count:]:
        row = {
            header: _clean(cell.text)
            for header, cell in zip(headers, grid_row)
        }
        if any(row.values()):
            rows.append(row)
    return headers, rows


def _table_headers_and_rows(item, document) -> tuple[list[str], list[dict[str, str]]]:
    try:
        frame = item.export_to_dataframe(document)
        headers = _unique_headers(frame.columns)
        rows = [
            {header: _cell_value(value) for header, value in zip(headers, values)}
            for values in frame.itertuples(index=False, name=None)
        ]
        return headers, [row for row in rows if any(row.values())]
    except Exception as exc:
        log.warning("DataFrame export failed; using Docling grid fallback: %s", exc)
        return _grid_headers_and_rows(item)


def _table_block(item, document, page_start: int, page_end: int, section: str):
    markdown = _clean(item.export_to_markdown(document))
    caption = _clean(item.caption_text(document))
    headers, rows = _table_headers_and_rows(item, document)

    cells = [
        StructuredTableCell(
            text=_clean(cell.text),
            row_start=cell.start_row_offset_idx,
            row_end=cell.end_row_offset_idx,
            column_start=cell.start_col_offset_idx,
            column_end=cell.end_col_offset_idx,
            is_column_header=cell.column_header,
            is_row_header=cell.row_header,
            is_row_section=cell.row_section,
        )
        for cell in item.data.table_cells
    ]
    return StructuredTableBlock(
        text=markdown,
        caption=caption,
        table_headers=headers,
        table_rows=rows,
        table_cells=cells,
        page_start=page_start,
        page_end=page_end,
        section=section,
    )


def _caption_refs(items: list[tuple[object, int]]) -> set[str]:
    refs: set[str] = set()
    for item, _ in items:
        if getattr(item, "label", None) != DocItemLabel.TABLE:
            continue
        refs.update(str(ref.cref) for ref in getattr(item, "captions", []))
    return refs


def _save_markdown(document, path: Path, markdown_dir: Path) -> Path:
    markdown_dir.mkdir(parents=True, exist_ok=True)
    destination = markdown_dir / f"{path.stem}.md"
    markdown = document.export_to_markdown(page_break_placeholder="<!-- page break -->")
    destination.write_text(markdown, encoding="utf-8")
    return destination


def _extract_blocks(document) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    current_section = ""
    items = list(document.iterate_items())
    chrome = _repeated_chrome(document, items)
    table_captions = _caption_refs(items)

    for item, _ in items:
        is_table_caption = str(getattr(item, "self_ref", "")) in table_captions
        if item.label in _SKIP_LABELS or is_table_caption:
            continue

        page_start, page_end = _item_pages(item)
        if item.label == DocItemLabel.TABLE:
            block = _table_block(item, document, page_start, page_end, current_section)
            if block.text or block.table_rows:
                blocks.append(block)
            continue

        text = _clean(getattr(item, "text", ""))
        if not text or normalize_heading(text) in chrome:
            continue
        if text.isdigit() and len(text) <= 4:
            continue

        if item.label == DocItemLabel.SECTION_HEADER:
            current_section = text

        markdown = _text_markdown(item)
        if markdown:
            blocks.append(
                ContentBlock(
                    text=markdown,
                    page_start=page_start,
                    page_end=page_end,
                    section=current_section,
                )
            )
    return blocks


def load_pdf(path: Path, markdown_dir: Path | None = None) -> LoadedDoc:
    """Convert a PDF with Docling, persist Markdown, and return typed blocks."""
    result = _converter().convert(path)
    document = result.document
    markdown_path = _save_markdown(
        document, path, markdown_dir or settings.resolved_markdown_dir
    )
    blocks = _extract_blocks(document)
    return LoadedDoc(
        doc_id=_doc_id(path),
        source=path.name,
        title=document.name or path.stem,
        blocks=blocks,
        markdown_path=markdown_path,
    )


def load_corpus(
    raw_dir: Path | None = None, markdown_dir: Path | None = None
) -> list[LoadedDoc]:
    raw_dir = raw_dir or settings.raw_dir
    paths = sorted(p for p in raw_dir.glob("**/*") if p.suffix.lower() == ".pdf")
    if not paths:
        raise FileNotFoundError(f"No PDFs found under {raw_dir}")

    docs = []
    for path in paths:
        loaded = load_pdf(path, markdown_dir=markdown_dir)
        log.info(
            "Converted %s: %d blocks (%d tables), %d chars; Markdown: %s",
            path.name,
            len(loaded.blocks),
            loaded.n_tables,
            loaded.n_chars,
            loaded.markdown_path,
        )
        docs.append(loaded)
    return docs
