"""Docling loading keeps table structure, provenance, and audit Markdown."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from docling_core.types.doc import DocItemLabel

from src.ingestion import loaders
from src.ingestion.loaders import (
    ContentBlock,
    StructuredTableBlock,
    StructuredTableCell,
    _clean,
    _extract_blocks,
    _headers_are_suspicious,
    _normalize_split_tables,
    _table_headers_and_rows,
    _unique_headers,
    load_pdf,
)


class FakeItem:
    def __init__(
        self,
        label,
        text="",
        pages=(1,),
        *,
        self_ref="",
        level=1,
        table_markdown="",
        caption="",
        columns=(),
        rows=(),
        captions=(),
    ):
        self.label = label
        self.text = text
        self.self_ref = self_ref
        self.level = level
        self.marker = "-"
        self.prov = [SimpleNamespace(page_no=page) for page in pages]
        self._table_markdown = table_markdown
        self._caption = caption
        self._frame = pd.DataFrame(list(rows), columns=list(columns))
        self.captions = [SimpleNamespace(cref=ref) for ref in captions]
        self.data = SimpleNamespace(
            table_cells=[
                SimpleNamespace(
                    text="ID",
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                    column_header=True,
                    row_header=False,
                    row_section=False,
                )
            ]
        )

    def export_to_markdown(self, document):
        return self._table_markdown

    def export_to_dataframe(self, document):
        return self._frame

    def caption_text(self, document):
        return self._caption


class FakeDocument:
    name = "Test document"
    pages = {1: object(), 2: object()}

    def __init__(self):
        self.items = [
            FakeItem(DocItemLabel.SECTION_HEADER, "Controls", pages=(1,), level=2),
            FakeItem(DocItemLabel.TEXT, "Apply the controls.", pages=(1,)),
            FakeItem(
                DocItemLabel.CAPTION,
                "Control mapping",
                pages=(2,),
                self_ref="#/texts/1",
            ),
            FakeItem(
                DocItemLabel.TABLE,
                pages=(2,),
                table_markdown="| ID | Action |\n| --- | --- |\n| PR.AA | Authorize |",
                caption="Control mapping",
                columns=("ID", "Action"),
                rows=(("PR.AA", "Authorize"),),
                captions=("#/texts/1",),
            ),
            FakeItem(DocItemLabel.PAGE_FOOTER, "2", pages=(2,)),
        ]

    def iterate_items(self):
        return iter((item, 1) for item in self.items)

    def export_to_markdown(self, **kwargs):
        return "## Controls\n\n| ID | Action |\n| --- | --- |\n| PR.AA | Authorize |"


def test_clean_preserves_markdown_table_rows():
    table = "| ID  | Action |\n| --- | --- |\n| PR.AA | Authorize |"
    assert _clean(table).splitlines() == [
        "| ID | Action |",
        "| --- | --- |",
        "| PR.AA | Authorize |",
    ]


def test_unique_headers_fills_blanks_and_deduplicates():
    assert _unique_headers([0, "Name", "Name", None]) == [
        "Column 1", "Name", "Name (2)", "Column 4"
    ]


def test_extract_blocks_builds_structured_table_and_removes_caption_duplicate():
    blocks = _extract_blocks(FakeDocument())
    assert [type(block) for block in blocks] == [
        ContentBlock,
        ContentBlock,
        StructuredTableBlock,
    ]

    table = blocks[-1]
    assert isinstance(table, StructuredTableBlock)
    assert table.caption == "Control mapping"
    assert table.table_headers == ["ID", "Action"]
    assert table.table_rows == [{"ID": "PR.AA", "Action": "Authorize"}]
    assert table.table_cells[0].is_column_header
    assert table.section == "Controls"
    assert (table.page_start, table.page_end) == (2, 2)


def test_load_pdf_always_saves_docling_markdown(monkeypatch, tmp_path: Path):
    document = FakeDocument()
    converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))
    monkeypatch.setattr(loaders, "_converter", lambda: converter)

    loaded = load_pdf(tmp_path / "source.pdf", markdown_dir=tmp_path / "markdown")

    assert loaded.markdown_path == tmp_path / "markdown" / "source.md"
    assert loaded.markdown_path.read_text(encoding="utf-8").startswith("## Controls")
    assert loaded.n_tables == 1


def _grid_cell(text: str, row: int, column: int, *, header: bool = False):
    return SimpleNamespace(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=column,
        end_col_offset_idx=column + 1,
        column_header=header,
    )


def _table_block_for_test(
    *,
    page: int,
    headers: list[str],
    rows: list[dict[str, str]],
    cells: list[StructuredTableCell] | None = None,
    caption: str = "",
) -> StructuredTableBlock:
    return StructuredTableBlock(
        text="table",
        page_start=page,
        page_end=page,
        section="Controls",
        caption=caption,
        table_headers=headers,
        table_rows=rows,
        table_cells=cells or [],
    )


def test_grid_header_detection_rejects_a_single_flagged_continuation_cell():
    grid = [
        [
            _grid_cell("", 0, 0),
            _grid_cell("", 0, 1),
            _grid_cell("continued paragraph", 0, 2, header=True),
        ],
        [
            _grid_cell("AC-2", 1, 0),
            _grid_cell("Account Management", 1, 1),
            _grid_cell("Manage accounts", 1, 2),
        ],
    ]
    item = SimpleNamespace(
        data=SimpleNamespace(grid=grid, num_cols=3),
        export_to_dataframe=lambda document: pd.DataFrame(
            columns=["Column 1", "Column 2", "continued paragraph"]
        ),
    )

    # Exercise the normal DataFrame path: it must notice the weak semantic
    # signal and fall back to the raw grid rather than losing row zero.
    headers, rows = _table_headers_and_rows(item, document=None)

    assert headers == ["Column 1", "Column 2", "Column 3"]
    assert rows[0] == {
        "Column 1": "",
        "Column 2": "",
        "Column 3": "continued paragraph",
    }
    assert rows[1]["Column 1"] == "AC-2"


def test_suspicious_header_detection_is_conservative_for_short_labels():
    paragraph = (
        "This is continuation text from a long description that was split across "
        "a page and should remain content instead of becoming a column heading."
    )

    assert _headers_are_suspicious(["Column 1", "Column 2", paragraph])
    assert not _headers_are_suspicious(["ID", "Control name", "Description"])


def test_normalization_merges_sparse_first_row_and_reuses_previous_headers():
    previous = _table_block_for_test(
        page=1,
        headers=["ID", "Control", "Description"],
        rows=[{"ID": "AC-1", "Control": "Policy", "Description": "First part."}],
        caption="Control catalog",
    )
    current = _table_block_for_test(
        page=2,
        headers=["Column 1", "Column 2", "Column 3"],
        rows=[
            {"Column 1": "", "Column 2": "", "Column 3": "Second part."},
            {
                "Column 1": "AC-2",
                "Column 2": "Accounts",
                "Column 3": "Manage accounts.",
            },
        ],
    )

    normalized = _normalize_split_tables([previous, current])

    assert normalized == [previous, current]
    assert previous.table_rows[-1]["Description"] == "First part. Second part."
    assert previous.page_end == 2
    assert current.table_headers == ["ID", "Control", "Description"]
    assert current.table_rows == [
        {"ID": "AC-2", "Control": "Accounts", "Description": "Manage accounts."}
    ]
    assert current.caption == "Control catalog"


def test_normalization_recovers_a_continuation_promoted_to_headers():
    continuation = (
        "continues with a paragraph-like explanation that belongs to the prior row "
        "and was incorrectly promoted by Docling"
    )
    previous = _table_block_for_test(
        page=4,
        headers=["ID", "Control", "Description"],
        rows=[{"ID": "IA-1", "Control": "Identify", "Description": "Start"}],
    )
    cells = [
        StructuredTableCell(
            text=continuation,
            row_start=0,
            row_end=1,
            column_start=2,
            column_end=3,
            is_column_header=True,
        )
    ]
    current_headers = ["Column 1", "Column 2", continuation]
    current = _table_block_for_test(
        page=5,
        headers=current_headers,
        rows=[
            {
                "Column 1": "IA-2",
                "Column 2": "Authenticate",
                continuation: "Verify identity",
            }
        ],
        cells=cells,
    )

    normalized = _normalize_split_tables([previous, current])

    assert len(normalized) == 2
    assert previous.table_rows[-1]["Description"] == f"Start {continuation}"
    assert current.table_headers == previous.table_headers
    assert current.table_rows == [
        {"ID": "IA-2", "Control": "Authenticate", "Description": "Verify identity"}
    ]


def test_normalization_recovers_continuation_appended_to_repeated_header():
    headers = ["CSF Element", "Description", "Priority", "Notes"]
    continuation = (
        "assist with maintaining cross-references between asset inventories and "
        "sources of vulnerability disclosures."
    )
    previous = _table_block_for_test(
        page=18,
        headers=headers,
        rows=[
            {
                "CSF Element": "ID.RA-08",
                "Description": "Vulnerability disclosures are handled",
                "Priority": "Medium",
                "Notes": "See [SP800-150] for data formats that may",
            }
        ],
    )
    cells = [
        StructuredTableCell(
            text=header if header != "Notes" else f"Notes {continuation}",
            row_start=0,
            row_end=1,
            column_start=column,
            column_end=column + 1,
            is_column_header=True,
        )
        for column, header in enumerate(headers)
    ]
    current = _table_block_for_test(
        page=19,
        headers=headers,
        rows=[
            {
                "CSF Element": "ID.RA-09",
                "Description": "Software integrity is assessed",
                "Priority": "Medium",
                "Notes": "See the notes for ID.RA.",
            }
        ],
        cells=cells,
    )

    normalized = _normalize_split_tables([previous, current])

    assert normalized == [previous, current]
    assert previous.page_end == 19
    assert previous.table_rows[-1]["Notes"] == (
        "See [SP800-150] for data formats that may " + continuation
    )
    assert current.table_rows[0]["CSF Element"] == "ID.RA-09"


def test_normalization_does_not_join_unrelated_same_width_tables():
    previous = _table_block_for_test(
        page=7,
        headers=["ID", "Control", "Description"],
        rows=[{"ID": "AC-1", "Control": "Policy", "Description": "Text"}],
    )
    current = _table_block_for_test(
        page=8,
        headers=["Code", "Meaning", "Owner"],
        rows=[{"Code": "X", "Meaning": "Example", "Owner": "Team"}],
    )

    _normalize_split_tables([previous, current])

    assert current.table_headers == ["Code", "Meaning", "Owner"]
    assert previous.table_rows[-1]["Description"] == "Text"
