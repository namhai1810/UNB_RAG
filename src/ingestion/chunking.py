"""Token-aware chunking that respects document structure.

Chunks are built from paragraphs, never split mid-sentence when avoidable, and
never span a section boundary - a chunk that straddles "Containment" and
"Eradication" produces citations that are technically correct and practically
useless. Token counts come from the BGE-M3 tokenizer so the budget matches what
the embedding model actually sees.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from src.config import settings
from src.ingestion.loaders import LoadedDoc, Page, normalize_heading

log = logging.getLogger(__name__)

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    page_start: int
    page_end: int
    text: str
    n_tokens: int

    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        section = f", {self.section}" if self.section else ""
        return f"{self.source}{section}, {pages}"

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class _Unit:
    """One paragraph (or sentence, when a paragraph is oversized)."""

    text: str
    page: int
    section: str
    n_tokens: int


def _resolve_sections(pages: list[Page], headings: dict[str, str]):
    """Yield (paragraph, page, section), tracking the section at line level.

    The loader's page -> section map is only approximate: a section that starts
    halfway down a page mislabels everything above it. In these PDFs the heading
    is its own line but rarely its own paragraph, so detection happens line by
    line, and a detected heading also forces a paragraph break.

    Headings are authoritative; the page map is a backstop, adopted only after it
    has disagreed for two consecutive pages - which means a heading was missed
    and the section really has moved on.
    """
    current = pages[0].section if pages else ""
    stale_since: str | None = None
    stale_pages = 0

    for page in pages:
        if page.section != current:
            stale_pages = stale_pages + 1 if stale_since == page.section else 1
            stale_since = page.section
            if stale_pages >= 2:
                current, stale_since, stale_pages = page.section, None, 0
        else:
            stale_since, stale_pages = None, 0

        buffer: list[str] = []

        def flush():
            text = "\n".join(buffer).strip()
            buffer.clear()
            return text

        for line in page.text.splitlines():
            stripped = line.strip()
            if not stripped:
                if (text := flush()):
                    yield text, page.page, current
                continue

            matched = headings.get(normalize_heading(stripped))
            if matched:
                if (text := flush()):
                    yield text, page.page, current
                current, stale_since, stale_pages = matched, None, 0
                buffer.append(stripped)   # keep the heading atop its section
                continue

            buffer.append(stripped)

        if (text := flush()):
            yield text, page.page, current


def _split_units(
    pages: list[Page], max_tokens: int, headings: dict[str, str] | None = None
) -> list[_Unit]:
    units: list[_Unit] = []
    for para, page_no, section in _resolve_sections(pages, headings or {}):
        n = count_tokens(para)
        if n <= max_tokens:
            units.append(_Unit(para, page_no, section, n))
            continue

        # Oversized paragraph (tables, long procedure lists): fall back to
        # sentences so no single unit can exceed the chunk budget.
        buf, buf_tokens = [], 0
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            sn = count_tokens(sent)
            if buf and buf_tokens + sn > max_tokens:
                units.append(_Unit(" ".join(buf), page_no, section, buf_tokens))
                buf, buf_tokens = [], 0
            buf.append(sent)
            buf_tokens += sn
        if buf:
            units.append(_Unit(" ".join(buf), page_no, section, buf_tokens))
    return units


def _overlap_tail(units: list[_Unit], budget: int) -> list[_Unit]:
    """Trailing units of the previous chunk, up to `budget` tokens."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.n_tokens > budget:
            break
        tail.insert(0, unit)
        total += unit.n_tokens
    return tail


def chunk_document(doc: LoadedDoc) -> list[Chunk]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    units = _split_units(doc.pages, size, doc.headings)
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(u.text for u in current)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{idx:05d}",
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                section=current[0].section,
                page_start=min(u.page for u in current),
                page_end=max(u.page for u in current),
                text=text,
                n_tokens=current_tokens,
            )
        )
        tail = _overlap_tail(current, overlap)
        current = list(tail)
        current_tokens = sum(u.n_tokens for u in tail)

    for unit in units:
        section_changed = bool(current) and unit.section != current[-1].section
        if section_changed:
            flush()
            current, current_tokens = [], 0   # no overlap across a section boundary
        elif current and current_tokens + unit.n_tokens > size:
            flush()
        current.append(unit)
        current_tokens += unit.n_tokens

    flush()
    return chunks


def chunk_corpus(docs: list[LoadedDoc]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc)
        log.info("Chunked %s -> %d chunks", doc.source, len(chunks))
        all_chunks.extend(chunks)
    if all_chunks:
        sizes = [c.n_tokens for c in all_chunks]
        log.info(
            "Corpus: %d chunks, tokens min/mean/max = %d/%d/%d",
            len(all_chunks), min(sizes), sum(sizes) // len(sizes), max(sizes),
        )
    return all_chunks
