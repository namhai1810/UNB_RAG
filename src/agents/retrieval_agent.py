"""Agent 2 - Retrieval.

Runs every query from the plan through hybrid search, then merges the result
lists with reciprocal rank fusion. Multi-query matters here: "eradication steps"
and "removing attacker persistence" surface different chunks of the same
document, and fusing them beats picking one phrasing and hoping.

This agent is deliberately LLM-free. Query formulation already happened in
triage (or in the verifier's rewrite), so adding another generation step here
would only add latency and a second place for the wording to drift.
"""
from __future__ import annotations

import logging

from src.config import settings
from src.logging_utils import log_event, payload
from src.retrieval import Evidence, HybridRetriever, get_retriever

log = logging.getLogger(__name__)

# Standard RRF constant; damps the influence of any single list's top hit.
RRF_K = 60


def _fuse(ranked_lists: list[list[Evidence]], top_k: int) -> list[Evidence]:
    """Reciprocal rank fusion across per-query result lists, keyed by chunk."""
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}

    for evidence_list in ranked_lists:
        for rank, evidence in enumerate(evidence_list):
            scores[evidence.chunk_id] = scores.get(evidence.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank + 1
            )
            # Keep the copy with the strongest cross-encoder score.
            if (
                evidence.chunk_id not in best
                or evidence.rerank_score > best[evidence.chunk_id].rerank_score
            ):
                best[evidence.chunk_id] = evidence

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [best[cid] for cid in ordered[:top_k]]


def retrieve(
    queries: list[str],
    round_no: int = 0,
    top_k: int | None = None,
    retriever: HybridRetriever | None = None,
    exclude_chunk_ids: set[str] | None = None,
) -> list[Evidence]:
    """Retrieve for each query and fuse.

    `exclude_chunk_ids` lets a retry round skip chunks the verifier already
    rejected, so a rewrite that lands on similar wording still returns something
    new instead of the same insufficient evidence.
    """
    retriever = retriever or get_retriever()
    top_k = top_k or settings.top_k_rerank
    exclude = exclude_chunk_ids or set()

    log_event(
        log,
        "retrieval.start",
        round=round_no,
        queries=payload(queries),
        excluded_chunk_ids=sorted(exclude),
        top_k=top_k,
    )
    ranked_lists: list[list[Evidence]] = []
    for query in queries:
        result = retriever.retrieve(query, top_k=settings.top_k_fused, round_no=round_no)
        filtered = [e for e in result.evidence if e.chunk_id not in exclude]
        log_event(
            log,
            "retrieval.query.output",
            round=round_no,
            query=payload(query),
            candidates=result.n_candidates,
            returned=len(result.evidence),
            after_exclusion=len(filtered),
            chunk_ids=[e.chunk_id for e in filtered],
        )
        if filtered:
            ranked_lists.append(filtered)

    if not ranked_lists:
        log.warning(
            "retrieval.empty | round=%d queries=%s", round_no, payload(queries)
        )
        return []

    fused = _fuse(ranked_lists, top_k)
    log_event(
        log,
        "retrieval.end",
        round=round_no,
        query_count=len(queries),
        unique_chunks=len({e.chunk_id for items in ranked_lists for e in items}),
        evidence=[
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "rerank_score": round(item.rerank_score, 4),
            }
            for item in fused
        ],
    )
    return fused
