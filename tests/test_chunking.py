"""Chunking rules: budget respected, sections never straddled, overlap carried."""
from __future__ import annotations

import pytest

from src.ingestion import chunking
from src.ingestion.chunking import chunk_document
from src.ingestion.loaders import LoadedDoc, Page, normalize_heading


@pytest.fixture(autouse=True)
def word_tokenizer(monkeypatch):
    """Swap the BGE-M3 tokenizer for word counting - fast, and precise enough."""
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(text.split()))


@pytest.fixture(autouse=True)
def small_budget(monkeypatch):
    monkeypatch.setattr(chunking.settings, "chunk_size", 30)
    monkeypatch.setattr(chunking.settings, "chunk_overlap", 12)


def _doc(pages: list[Page], headings: list[str] | None = None) -> LoadedDoc:
    return LoadedDoc(
        doc_id="doc1",
        source="test.pdf",
        title="Test",
        pages=pages,
        headings={normalize_heading(h): h for h in (headings or [])},
    )


def _page(text: str, page: int = 1, section: str = "S1") -> Page:
    return Page(doc_id="doc1", source="test.pdf", title="Test", page=page,
                section=section, text=text)


def test_chunks_respect_the_token_budget():
    paras = "\n\n".join(" ".join(f"w{i}{j}" for j in range(10)) for i in range(12))
    chunks = chunk_document(_doc([_page(paras)]))
    assert chunks
    # A chunk may exceed the budget only by the unit that tipped it over.
    assert all(c.n_tokens <= 30 + 10 for c in chunks)


def test_detected_heading_splits_chunks():
    pages = [
        _page("Containment\nalpha beta gamma", page=1, section="Containment"),
        _page("Eradication\ndelta epsilon zeta", page=2, section="Eradication"),
    ]
    chunks = chunk_document(_doc(pages, headings=["Containment", "Eradication"]))
    assert [c.section for c in chunks] == ["Containment", "Eradication"]
    assert "delta" not in chunks[0].text


def test_section_starting_mid_page_does_not_relabel_the_text_above_it():
    """The regression this design exists for: page-level labels alone are wrong."""
    pages = [
        _page("alpha beta gamma", page=1, section="Containment"),
        # The loader maps all of page 2 to Eradication, but the first paragraph
        # is still the tail of Containment - the heading appears further down.
        _page("delta epsilon\n\nEradication\nzeta eta", page=2, section="Eradication"),
    ]
    chunks = chunk_document(_doc(pages, headings=["Containment", "Eradication"]))
    by_section = {c.section: c.text for c in chunks}
    assert "delta epsilon" in by_section["Containment"]
    assert "zeta eta" in by_section["Eradication"]


def test_page_map_is_adopted_when_a_heading_is_missed():
    """Backstop: two pages of disagreement means the heading was not extracted."""
    pages = [
        _page("alpha beta", page=1, section="Containment"),
        _page("gamma delta", page=2, section="Eradication"),
        _page("epsilon zeta", page=3, section="Eradication"),
    ]
    chunks = chunk_document(_doc(pages))       # no headings supplied
    assert "Eradication" in {c.section for c in chunks}


def test_overlap_is_carried_within_a_section():
    paras = "\n\n".join(f"para{i} " + " ".join(f"w{j}" for j in range(9)) for i in range(6))
    chunks = chunk_document(_doc([_page(paras)]))
    assert len(chunks) > 1
    # The tail of chunk n reappears at the head of chunk n+1.
    tail = chunks[0].text.split("\n\n")[-1]
    assert tail in chunks[1].text


def test_oversized_paragraph_is_split_by_sentence():
    long_para = " ".join(f"Sentence number {i} here." for i in range(30))
    chunks = chunk_document(_doc([_page(long_para)]))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 30 + 10 for c in chunks)


def test_page_range_spans_the_contributing_pages():
    pages = [_page("alpha beta", page=3), _page("gamma delta", page=4)]
    chunks = chunk_document(_doc(pages))
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 4
    assert chunks[0].citation() == "test.pdf, S1, pp.3-4"


def test_single_page_citation_format():
    chunks = chunk_document(_doc([_page("alpha beta", page=7)]))
    assert chunks[0].citation() == "test.pdf, S1, p.7"


def test_chunk_ids_are_unique():
    paras = "\n\n".join(" ".join(f"w{i}{j}" for j in range(10)) for i in range(10))
    chunks = chunk_document(_doc([_page(paras)]))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
