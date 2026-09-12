"""Semantic prose chunks and header:value table row chunks."""
from __future__ import annotations

import pytest

from src.ingestion import chunking
from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import ContentBlock, LoadedDoc, StructuredTableBlock


@pytest.fixture(autouse=True)
def word_tokenizer(monkeypatch):
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(text.split()))


@pytest.fixture(autouse=True)
def small_budget(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 30)
    monkeypatch.setattr(chunking.settings, "chunk_overlap", 12)


def _doc(blocks) -> LoadedDoc:
    return LoadedDoc(doc_id="doc1", source="test.pdf", title="Test", blocks=blocks)


def _text(text: str, page: int = 1, section: str = "S1") -> ContentBlock:
    return ContentBlock(text=text, page_start=page, page_end=page, section=section)


def _table(
    rows: list[dict[str, str]],
    page: int = 1,
    section: str = "S1",
    caption: str = "Technique mapping",
) -> StructuredTableBlock:
    headers = list(rows[0]) if rows else ["ID", "Action"]
    return StructuredTableBlock(
        text="| audit markdown |",
        caption=caption,
        table_headers=headers,
        table_rows=rows,
        page_start=page,
        page_end=page,
        section=section,
    )


def test_paragraph_chunks_respect_token_budget():
    blocks = [_text(" ".join(f"w{i}{j}" for j in range(10))) for i in range(12)]
    chunks = chunk_document(_doc(blocks))
    assert chunks
    assert all(c.n_tokens <= 40 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.chunk_type for c in chunks} == {"paragraph"}


def test_section_change_splits_paragraph_chunks():
    blocks = [
        _text("alpha beta", section="Containment"),
        _text("delta epsilon", page=2, section="Eradication"),
    ]
    chunks = chunk_document(_doc(blocks))
    assert [c.section for c in chunks] == ["Containment", "Eradication"]
    assert "delta" not in chunks[0].text


def test_overlap_is_carried_within_a_section():
    blocks = [_text(f"para{i} " + " ".join(f"w{j}" for j in range(9))) for i in range(6)]
    chunks = chunk_document(_doc(blocks))
    assert len(chunks) > 1
    tail = chunks[0].text.split("\n\n")[-1]
    assert tail in chunks[1].text


def test_oversized_paragraph_is_split_by_sentence():
    long_paragraph = " ".join(f"Sentence number {i} here." for i in range(30))
    chunks = chunk_document(_doc([_text(long_paragraph)]))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 40 for c in chunks)


def test_page_range_spans_contributing_blocks():
    chunks = chunk_document(
        _doc([_text("alpha beta", page=3), _text("gamma delta", page=4)])
    )
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 4
    assert chunks[0].citation() == "test.pdf, S1, pp.3-4"


def test_table_row_uses_header_value_retrieval_representation():
    table = _table([
        {
            "ID": "T1059",
            "Technique": "Command and Scripting Interpreter",
            "Tactic": "Execution",
        }
    ], section="Execution")
    chunks = chunk_document(_doc([_text("before table"), table, _text("after table")]))

    assert [chunk.chunk_type for chunk in chunks] == ["paragraph", "table", "paragraph"]
    table_chunk = chunks[1]
    assert "|" not in table_chunk.text
    assert "Section: Execution" in table_chunk.text
    assert "Table: Technique mapping" in table_chunk.text
    assert "ID: T1059" in table_chunk.text
    assert "Technique: Command and Scripting Interpreter" in table_chunk.text
    assert "Tactic: Execution" in table_chunk.text


def test_each_table_row_becomes_an_independently_retrievable_chunk():
    rows = [{"ID": f"C{i}", "Action": f"action {i}"} for i in range(8)]
    chunks = chunk_document(_doc([_table(rows, page=7, section="Controls")]))

    assert len(chunks) == 8
    assert all(chunk.chunk_type == "table" for chunk in chunks)
    assert [chunk.table_row_start for chunk in chunks] == list(range(1, 9))
    assert all(chunk.page_start == chunk.page_end == 7 for chunk in chunks)


def test_oversized_cell_splits_but_repeats_row_identity(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 14)
    description = " ".join(f"detail{i}" for i in range(30))
    chunks = chunk_document(_doc([_table([{"ID": "T1059", "Description": description}])]))

    assert len(chunks) > 1
    assert all("ID: T1059" in chunk.text for chunk in chunks)
    assert all("Description:" in chunk.text for chunk in chunks)
    assert all(chunk.n_tokens <= 14 for chunk in chunks)


def test_qdrant_payload_keeps_structured_row_metadata():
    chunk = chunk_document(_doc([_table([{"ID": "T1059", "Tactic": "Execution"}])]))[0]
    payload = chunk.to_dict()
    assert payload["table_headers"] == ["ID", "Tactic"]
    assert payload["table_rows"] == [{"ID": "T1059", "Tactic": "Execution"}]
    assert payload["table_row_start"] == payload["table_row_end"] == 1


def test_paragraph_payload_omits_empty_table_metadata():
    payload = chunk_document(_doc([_text("alpha beta")]))[0].to_dict()
    assert "table_headers" not in payload
    assert "table_rows" not in payload


def test_chunk_ids_are_unique_across_both_types():
    chunks = chunk_document(
        _doc([_text("alpha beta"), _table([{"A": "1", "B": "2"}]), _text("gamma delta")])
    )
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
