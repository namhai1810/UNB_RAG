"""Shared state passed between graph nodes.

LangGraph merges node returns into this dict, so a node only returns the keys it
changes. Lists that grow across retry rounds (`evidence`, `tried_queries`) are
merged explicitly inside the node rather than by a reducer, because the merge is
semantic - dedup by chunk id, keep the best-scoring copy - not a plain append.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from src.agents.answer_agent import AnswerPayload
from src.agents.triage_agent import TriageDecision
from src.agents.verifier_agent import VerificationVerdict
from src.retrieval import Evidence

Status = Literal[
    "pending",              # still running
    "rejected",             # out of scope
    "needs_clarification",  # ambiguous
    "answered",             # grounded answer produced
    "answered_partial",     # answered after the retry budget ran out
    "insufficient",         # no usable evidence at all
    "error",
]


class TraceEvent(TypedDict, total=False):
    node: str
    round: int
    detail: str
    payload: dict


class RAGState(TypedDict, total=False):
    # --- input
    query: str

    # --- agent 1
    triage: TriageDecision

    # --- agent 2
    search_queries: list[str]     # queries for the current round
    tried_queries: list[str]      # every query used so far
    evidence: list[Evidence]      # current working set
    rejected_chunk_ids: list[str] # chunks the verifier already dismissed
    round: int

    # --- agent 3
    verdict: VerificationVerdict

    # --- agent 4
    answer: AnswerPayload

    # --- output
    status: Status
    response: str                 # user-facing text
    trace: list[TraceEvent]
    error: str


def initial_state(query: str) -> RAGState:
    return RAGState(
        query=query,
        search_queries=[],
        tried_queries=[],
        evidence=[],
        rejected_chunk_ids=[],
        round=0,
        status="pending",
        response="",
        trace=[],
    )


def add_trace(state: RAGState, node: str, detail: str, **payload) -> list[TraceEvent]:
    """Return the trace list with one event appended (never mutates in place)."""
    event: TraceEvent = {
        "node": node,
        "round": state.get("round", 0),
        "detail": detail,
        "payload": payload,
    }
    return [*state.get("trace", []), event]
