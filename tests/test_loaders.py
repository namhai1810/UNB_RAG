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
    _clean,
    _extract_blocks,
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
