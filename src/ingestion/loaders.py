"""PDF loading for the cyber-security corpus.

Citations are only useful if they point somewhere a human can check, so every
page keeps its page number and - when the PDF carries bookmarks - the section
heading it falls under. Running headers/footers are stripped because they
otherwise appear in every chunk and drag retrieval toward boilerplate.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.config import settings

log = logging.getLogger(__name__)

# A line repeated on at least this share of pages is chrome, not content.
_CHROME_PAGE_RATIO = 0.5
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")
# Hyphen at end of line splitting a word across lines.
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
# Bookmark text and page text disagree on spacing, case, and trailing dots.
_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_heading(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


@dataclass
class Page:
    doc_id: str
    source: str          # file name, used verbatim in citations
    title: str           # document title (PDF metadata or file stem)
    page: int            # 1-indexed
    section: str
    text: str


@dataclass
class LoadedDoc:
    doc_id: str
    source: str
    title: str
    pages: list[Page] = field(default_factory=list)
    # normalised bookmark title -> original title. Lets the chunker recognise a
    # heading where it actually appears in the text, rather than trusting the
    # page-level map for a section that starts mid-page.
    headings: dict[str, str] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def _doc_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


def _clean(text: str) -> str:
    text = text.replace("­", "")            # soft hyphen
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _HYPHEN_RE.sub(r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _chrome_lines(page_texts: list[str]) -> set[str]:
    """Find headers/footers by looking at the first and last lines of each page."""
    counter: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[:2] + lines[-2:]:
            # Pure page numbers are handled by the digit check below.
            if 3 <= len(line) <= 120:
                counter[line] += 1
    threshold = max(2, int(len(page_texts) * _CHROME_PAGE_RATIO))
    return {line for line, n in counter.items() if n >= threshold}


def _strip_chrome(text: str, chrome: set[str]) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in chrome:
            continue
        if stripped.isdigit() and len(stripped) <= 4:   # bare page number
            continue
        kept.append(line)
    return "\n".join(kept)


def _section_map(doc: pymupdf.Document) -> tuple[dict[int, str], dict[str, str]]:
    """Return (page -> nearest preceding bookmark, normalised title -> title)."""
    toc = doc.get_toc(simple=True)
    if not toc:
        return {}, {}
    starts: list[tuple[int, str]] = []
    headings: dict[str, str] = {}
    for level, title, page in toc:
        title = title.strip()
        if page and page > 0 and level <= 3 and title:
            starts.append((page, title))
            headings[normalize_heading(title)] = title
    if not starts:
        return {}, headings
    starts.sort()

    mapping: dict[int, str] = {}
    current = starts[0][1]
    idx = 0
    for page in range(1, doc.page_count + 1):
        while idx < len(starts) and starts[idx][0] <= page:
            current = starts[idx][1]
            idx += 1
        mapping[page] = current
    return mapping, headings


def load_pdf(path: Path) -> LoadedDoc:
    doc = pymupdf.open(path)
    try:
        meta_title = (doc.metadata or {}).get("title") or ""
        title = meta_title.strip() or path.stem
        sections, headings = _section_map(doc)
        raw_pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
        chrome = _chrome_lines(raw_pages)

        loaded = LoadedDoc(
            doc_id=_doc_id(path), source=path.name, title=title, headings=headings
        )
        for i, raw in enumerate(raw_pages):
            text = _clean(_strip_chrome(raw, chrome))
            if len(text) < 40:  # cover pages, blank pages, pure figures
                continue
            loaded.pages.append(
                Page(
                    doc_id=loaded.doc_id,
                    source=path.name,
                    title=title,
                    page=i + 1,
                    section=sections.get(i + 1, ""),
                    text=text,
                )
            )
        return loaded
    finally:
        doc.close()


def load_corpus(raw_dir: Path | None = None) -> list[LoadedDoc]:
    raw_dir = raw_dir or settings.raw_dir
    paths = sorted(p for p in raw_dir.glob("**/*") if p.suffix.lower() == ".pdf")
    if not paths:
        raise FileNotFoundError(f"No PDFs found under {raw_dir}")

    docs = []
    for path in paths:
        loaded = load_pdf(path)
        log.info(
            "Loaded %s: %d pages kept, %d chars", path.name, len(loaded.pages), loaded.n_chars
        )
        docs.append(loaded)
    return docs
