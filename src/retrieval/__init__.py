"""Retrieval pipeline: hybrid search, reranking, and adjacent context expansion."""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field

from src.config import settings
from src.logging_utils import payload

log = logging.getLogger(__name__)


class Evidence(BaseModel):
    """One retrieved chunk, carrying everything a citation needs."""

    chunk_id: str
    source: str
    doc_id: str = ""
    chunk_index: int = -1
    title: str = ""
    section: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = "paragraph"
    table_caption: str = ""
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, str]] = Field(default_factory=list)
    table_row_start: int = 0
    table_row_end: int = 0
    text: str
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    is_neighbor: bool = False
    seed_chunk_id: str = ""
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
        context = (
            f"; adjacent context for {self.seed_chunk_id}"
            if self.is_neighbor and self.seed_chunk_id
            else ""
        )
        return f"[{index}] ({self.citation}{context})\n{self.text}"


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

        evidence = [self._to_evidence(c, round_no=round_no) for c in kept]
        log.info(
            "Retrieved %d candidates -> %d kept for %s",
            len(candidates),
            len(evidence),
            payload(query),
        )
        return RetrievalResult(query=query, evidence=evidence, n_candidates=len(candidates))

    @staticmethod
    def _payload_chunk_index(payload: dict) -> int:
        """Read the explicit index, with compatibility for indexes built earlier."""
        index = payload.get("chunk_index")
        if isinstance(index, int):
            return index

        doc_id = payload.get("doc_id", "")
        chunk_id = payload.get("chunk_id", "")
        prefix = f"{doc_id}-"
        suffix = chunk_id[len(prefix):] if doc_id and chunk_id.startswith(prefix) else ""
        return int(suffix) if suffix.isdigit() else -1

    @classmethod
    def _to_evidence(
        cls,
        payload: dict,
        *,
        round_no: int,
        is_neighbor: bool = False,
        seed_chunk_id: str = "",
        rerank_score: float | None = None,
    ) -> Evidence:
        return Evidence(
            chunk_id=payload["chunk_id"],
            source=payload["source"],
            doc_id=payload.get("doc_id", ""),
            chunk_index=cls._payload_chunk_index(payload),
            title=payload.get("title", ""),
            section=payload.get("section", ""),
            page_start=payload.get("page_start", 0),
            page_end=payload.get("page_end", 0),
            chunk_type=payload.get("chunk_type", "paragraph"),
            table_caption=payload.get("table_caption", ""),
            table_headers=payload.get("table_headers", []),
            table_rows=payload.get("table_rows", []),
            table_row_start=payload.get("table_row_start", 0),
            table_row_end=payload.get("table_row_end", 0),
            text=payload["text"],
            fusion_score=payload.get("fusion_score", 0.0),
            rerank_score=(
                payload.get("rerank_score", 0.0)
                if rerank_score is None
                else rerank_score
            ),
            is_neighbor=is_neighbor,
            seed_chunk_id=seed_chunk_id,
            round=round_no,
        )

    def expand_neighbors(
        self,
        seeds: list[Evidence],
        *,
        window: int | None = None,
        max_total: int | None = None,
        same_section_only: bool | None = None,
        exclude_chunk_ids: set[str] | None = None,
    ) -> list[Evidence]:
        """Append nearby chunks to reranked seeds, preserving seed rank and citations."""
        window = settings.neighbor_chunk_window if window is None else window
        max_total = settings.neighbor_max_total_chunks if max_total is None else max_total
        same_section_only = (
            settings.neighbor_same_section_only
            if same_section_only is None
            else same_section_only
        )
        if not seeds or window <= 0 or max_total <= len(seeds):
            return list(seeds)

        excluded = exclude_chunk_ids or set()
        seen = {seed.chunk_id for seed in seeds} | excluded
        requests: list[tuple[str, Evidence]] = []

        # Round-robin by distance gives every high-quality seed nearby context
        # before a larger window consumes the remaining prompt budget.
        for distance in range(1, window + 1):
            for seed in seeds:
                if not seed.doc_id or seed.chunk_index < 0:
                    continue
                for index in (seed.chunk_index - distance, seed.chunk_index + distance):
                    if index < 0:
                        continue
                    chunk_id = f"{seed.doc_id}-{index:05d}"
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    requests.append((chunk_id, seed))

        payloads = self.store.get_chunks([chunk_id for chunk_id, _ in requests])
        by_id = {item["chunk_id"]: item for item in payloads}
        expanded = list(seeds)
        for chunk_id, seed in requests:
            payload = by_id.get(chunk_id)
            if payload is None:
                continue
            if same_section_only and payload.get("section", "") != seed.section:
                continue
            expanded.append(
                self._to_evidence(
                    payload,
                    round_no=seed.round,
                    is_neighbor=True,
                    seed_chunk_id=seed.chunk_id,
                    # Neighbors are context, not cross-encoder-ranked hits.
                    rerank_score=0.0,
                )
            )
            if len(expanded) >= max_total:
                break
        return expanded

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
