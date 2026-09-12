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
_GENERIC_HEADER_RE = re.compile(
    r"^column\s+\d+(?:\s+\(\d+\))?$", re.IGNORECASE
)
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


def _header_row_is_semantic(row, row_index: int, num_cols: int) -> bool:
    """Require a header flag to cover the row, not merely one stray cell.

    Docling occasionally flags only the non-empty cell of a continuation row as
    ``column_header``. Requiring at least half of the columns (and at least two
    for multi-column tables) prevents that cell from promoting the whole row to
    DataFrame column names. A genuinely merged header spanning every column is
    still accepted because its declared span covers the complete row.
    """
    covered_columns: set[int] = set()
    for cell in row:
        if not getattr(cell, "column_header", False):
            continue
        if getattr(cell, "start_row_offset_idx", row_index) != row_index:
            continue
        start = max(0, int(getattr(cell, "start_col_offset_idx", 0)))
        end = min(num_cols, int(getattr(cell, "end_col_offset_idx", start + 1)))
        covered_columns.update(range(start, max(start + 1, end)))

    if num_cols <= 1:
        return bool(covered_columns)
    required = max(2, math.ceil(num_cols / 2))
    return len(covered_columns) >= required


def _headers_are_suspicious(headers: list[str]) -> bool:
    """Identify likely data mistakenly promoted to column names.

    Real column labels are normally short noun phrases. A mostly-placeholder
    header row, or one containing a long sentence/paragraph, is much more likely
    to be a sparse continuation row at a PDF page boundary. Thresholds are
    deliberately conservative so legitimate descriptive labels remain intact.
    """
    if not headers:
        return True

    meaningful = [
        header
        for header in headers
        if not _GENERIC_HEADER_RE.fullmatch(header.strip())
    ]
    if len(headers) > 1 and len(meaningful) <= 1:
        return True

    for header in meaningful:
        word_count = len(header.split())
        if len(header) >= 160 or (len(header) >= 80 and word_count >= 14):
            return True
    return False


def _semantic_header_count(grid, num_cols: int) -> int:
    """Return the number of consecutive, confidently classified header rows."""
    count = 0
    for row_index, row in enumerate(grid):
        if not _header_row_is_semantic(row, row_index, num_cols):
            break
        count += 1
    return count


def _grid_headers_and_rows(item) -> tuple[list[str], list[dict[str, str]]]:
    """Fallback for a malformed table that cannot be exported as a DataFrame."""
    grid = item.data.grid
    num_cols = int(getattr(item.data, "num_cols", len(grid[0]) if grid else 0))
    header_count = _semantic_header_count(grid, num_cols)

    raw_headers: list[str] = []
    for column_index in range(num_cols):
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

        # The DataFrame has already applied Docling's semantic header flags. If
        # the leading grid row has only partial header coverage, or its resulting
        # labels look like prose, rebuild from the raw grid with our stricter
        # rule so the alleged header remains available as ordinary row data.
        grid = getattr(item.data, "grid", None)
        if grid:
            num_cols = int(getattr(item.data, "num_cols", len(grid[0])))
            first_has_header_flag = any(
                getattr(cell, "column_header", False) for cell in grid[0]
            )
            ambiguous_header = first_has_header_flag and not _header_row_is_semantic(
                grid[0], 0, num_cols
            )
            if ambiguous_header or _headers_are_suspicious(headers):
                return _grid_headers_and_rows(item)

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


def _row_values(row: dict[str, str], headers: list[str]) -> list[str]:
    return [_clean(row.get(header, "")) for header in headers]


def _physical_first_row(block: StructuredTableBlock) -> list[str]:
    """Recover row zero from Docling cells when it was promoted to headers."""
    values = [""] * len(block.table_headers)
    for cell in block.table_cells:
        if not (cell.row_start <= 0 < cell.row_end) or not cell.text:
            continue
        start = max(0, cell.column_start)
        end = min(len(values), max(start + 1, cell.column_end))
        for column_index in range(start, end):
            values[column_index] = cell.text
    return values


def _continuation_appended_to_headers(
    block: StructuredTableBlock, expected_headers: list[str]
) -> list[str]:
    """Recover text Docling appended to a repeated page header.

    At some page boundaries TableFormer puts continuation prose in the same
    physical cell as a repeated column label. ``export_to_dataframe`` keeps the
    clean label as the DataFrame column name, so that prose otherwise disappears
    from both headers and rows. Accept the suffix only when every populated cell
    in physical row zero starts with its expected header and exactly one column
    has extra text. This keeps the repair specific to repeated table headers.
    """
    if not expected_headers or len(block.table_headers) != len(expected_headers):
        return []

    physical_values = _physical_first_row(block)
    suffixes = [""] * len(expected_headers)
    matched_columns = 0
    for column_index, (physical, expected) in enumerate(
        zip(physical_values, expected_headers)
    ):
        physical = _clean(physical)
        expected = _clean(expected)
        if not physical:
            continue
        if not expected or not physical.casefold().startswith(expected.casefold()):
            return []

        matched_columns += 1
        suffixes[column_index] = physical[len(expected) :].strip(" \t:-")

    populated_suffixes = [suffix for suffix in suffixes if suffix]
    if matched_columns != len(expected_headers) or len(populated_suffixes) != 1:
        return []
    return suffixes


def _single_nonempty_column(values: list[str]) -> int | None:
    populated = [index for index, value in enumerate(values) if value.strip()]
    return populated[0] if len(values) > 1 and len(populated) == 1 else None


def _rekey_rows(
    rows: list[dict[str, str]], old_headers: list[str], new_headers: list[str]
) -> list[dict[str, str]]:
    """Map rows positionally when a continuation table inherits prior headers."""
    return [
        {
            header: value
            for header, value in zip(new_headers, _row_values(row, old_headers))
        }
        for row in rows
    ]


def _append_continuation(existing: str, continuation: str) -> str:
    existing, continuation = _clean(existing), _clean(continuation)
    if not existing:
        return continuation
    if not continuation or continuation == existing or existing.endswith(continuation):
        return existing
    return f"{existing} {continuation}"


def _repair_continued_table(
    previous: StructuredTableBlock, current: StructuredTableBlock
) -> bool:
    """Repair one likely page-boundary table fragment in place.

    Returns true only when a sparse leading row was consumed into the prior
    table's final row. Header reuse can still occur without consuming a row.
    """
    old_headers = list(current.table_headers)
    same_width = bool(previous.table_headers) and len(previous.table_headers) == len(
        old_headers
    )
    adjacent_page = current.page_start == previous.page_end + 1
    same_section = _clean(current.section).casefold() == _clean(
        previous.section
    ).casefold()
    if not (same_width and adjacent_page and same_section):
        return False

    same_headers = [_clean(header).casefold() for header in old_headers] == [
        _clean(header).casefold() for header in previous.table_headers
    ]
    suspicious_headers = _headers_are_suspicious(old_headers)

    # A suspicious header may itself be physical row zero, before the rows in
    # the exported DataFrame. Prefer that raw row when it has the sparse shape;
    # otherwise inspect the first extracted data row. If both representations
    # contain row zero, remember to remove its duplicate after merging.
    extracted_values = (
        _row_values(current.table_rows[0], old_headers) if current.table_rows else []
    )
    first_values = extracted_values
    continuation_column = _single_nonempty_column(first_values)
    came_from_rows = continuation_column is not None

    # A repeated page header can share its final physical cell with prose that
    # continues the preceding page's last row. The DataFrame exposes only the
    # clean header, so inspect raw cells even when the exported headers look
    # completely normal.
    header_suffixes = _continuation_appended_to_headers(
        current, previous.table_headers
    )
    header_suffix_column = _single_nonempty_column(header_suffixes)
    if header_suffix_column is not None:
        first_values = header_suffixes
        continuation_column = header_suffix_column
        came_from_rows = False

    if suspicious_headers:
        physical_values = _physical_first_row(current)
        physical_column = _single_nonempty_column(physical_values)
        if physical_column is not None:
            first_values = physical_values
            continuation_column = physical_column
            came_from_rows = bool(current.table_rows) and (
                extracted_values == physical_values
            )

    # Reusing unrelated headers solely because the column count matches is too
    # aggressive. Require matching headers, a suspicious header shape, or the
    # strong sparse-row continuation signal described above.
    if not (same_headers or suspicious_headers or continuation_column is not None):
        return False

    if old_headers != previous.table_headers:
        current.table_rows = _rekey_rows(
            current.table_rows, old_headers, previous.table_headers
        )
        current.table_headers = list(previous.table_headers)
    if not current.caption:
        current.caption = previous.caption

    if continuation_column is None or not previous.table_rows:
        return False

    if came_from_rows:
        current.table_rows.pop(0)
    target_header = previous.table_headers[continuation_column]
    previous.table_rows[-1][target_header] = _append_continuation(
        previous.table_rows[-1].get(target_header, ""),
        first_values[continuation_column],
    )
    # The merged row now has content from both pages, so its table block must
    # carry the expanded provenance used by downstream citations.
    previous.page_end = max(previous.page_end, current.page_start)
    return True


def _normalize_split_tables(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Normalize adjacent page-split tables before table-row chunking.

    Only consecutive table blocks in the same section are considered. This
    keeps the repair local and avoids joining same-width but unrelated tables
    separated by prose or a new section.
    """
    normalized: list[DocumentBlock] = []
    for block in blocks:
        if not isinstance(block, StructuredTableBlock):
            normalized.append(block)
            continue

        previous = normalized[-1] if normalized else None
        consumed = False
        if isinstance(previous, StructuredTableBlock):
            consumed = _repair_continued_table(previous, block)

        # A fragment containing only the consumed continuation cell contributes
        # no independent retrieval row. Its content and page are already on the
        # preceding table, while the full original Markdown remains on disk.
        if not (consumed and not block.table_rows):
            normalized.append(block)
    return normalized


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
    return _normalize_split_tables(blocks)


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
