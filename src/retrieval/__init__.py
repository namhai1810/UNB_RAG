"""Retrieval pipeline: hybrid search -> cross-encoder rerank -> top-k evidence."""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field

from src.config import settings

log = logging.getLogger(__name__)


class Evidence(BaseModel):
    """One retrieved chunk, carrying everything a citation needs."""

    chunk_id: str
    source: str
    title: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    text: str
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    # Which retrieval round produced it - useful when debugging rewrite loops.
    round: int = 0

    @property
    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        section = f", {self.section}" if self.section else ""
        return f"{self.source}{section}, {pages}"

    def render(self, index: int) -> str:
        """Numbered block the agents read. The [n] marker is what gets cited."""
        return f"[{index}] ({self.citation})\n{self.text}"


class RetrievalResult(BaseModel):
    query: str
    evidence: list[Evidence] = Field(default_factory=list)
    n_candidates: int = 0


class HybridRetriever:
    """Dense + BM25 fusion in Qdrant, then cross-encoder reranking."""

    def __init__(self, store=None, reranker=None) -> None:
        from src.ingestion.indexing import VectorStore
        from src.retrieval.reranker import get_reranker

        self.store = store or VectorStore()
        self._reranker = reranker or get_reranker()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        round_no: int = 0,
        source_filter: list[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.top_k_rerank
        candidates = self.store.hybrid_search(query, source_filter=source_filter)
        if not candidates:
            return RetrievalResult(query=query, evidence=[], n_candidates=0)

        scores = self._reranker.score(query, [c["text"] for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        kept = [
            c for c in candidates if c["rerank_score"] >= settings.rerank_score_threshold
        ][:top_k]

        evidence = [
            Evidence(
                chunk_id=c["chunk_id"],
                source=c["source"],
                title=c.get("title", ""),
                section=c.get("section", ""),
                page_start=c.get("page_start", 0),
                page_end=c.get("page_end", 0),
                text=c["text"],
                fusion_score=c.get("fusion_score", 0.0),
                rerank_score=c["rerank_score"],
                round=round_no,
            )
            for c in kept
        ]
        log.info(
            "Retrieved %d candidates -> %d kept for %r", len(candidates), len(evidence), query
        )
        return RetrievalResult(query=query, evidence=evidence, n_candidates=len(candidates))

    def close(self) -> None:
        self.store.close()


_active_retrievers: list[HybridRetriever] = []


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    retriever = HybridRetriever()
    _active_retrievers.append(retriever)
    return retriever


def get_active_retriever() -> HybridRetriever | None:
    """Return the shared retriever without initializing its GPU models."""
    return _active_retrievers[-1] if _active_retrievers else None


def format_evidence(evidence: list[Evidence]) -> str:
    """Render an evidence list into the numbered block agents are prompted on."""
    if not evidence:
        return "(no evidence retrieved)"
    return "\n\n".join(e.render(i + 1) for i, e in enumerate(evidence))


__all__ = [
    "Evidence",
    "RetrievalResult",
    "HybridRetriever",
    "get_active_retriever",
    "get_retriever",
    "format_evidence",
]
