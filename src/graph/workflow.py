"""LangGraph wiring of the four agents.

                       User query
                           |
                      [1] triage
              /            |            \\
      out_of_scope     ambiguous      in_scope
           |               |               |
        reject          clarify      [2] retrieve  <---------+
                                           |                 |
                                     [3] verify              |
                                    /            \\           |
                            insufficient        enough   rewrite +
                                 |                 |      retry (budget)
                                 +-----------------|-----------+
                                                   |
                                            [4] answer
                                                   |
                                    grounded answer + citations
"""
from __future__ import annotations

import logging
from time import perf_counter

from langgraph.graph import END, START, StateGraph

from src.agents import answer_agent, retrieval_agent, triage_agent, verifier_agent
from src.config import settings
from src.graph.state import RAGState, add_trace, initial_state
from src.logging_utils import (
    log_event,
    logged_node,
    new_request_id,
    payload,
    reset_request_id,
    set_request_id,
    state_summary,
)
from src.retrieval import Evidence

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@logged_node(log, "triage")
def triage_node(state: RAGState) -> dict:
    decision = triage_agent.triage(state["query"])
    return {
        "triage": decision,
        "search_queries": decision.search_queries,
        "trace": add_trace(
            state, "triage", decision.reasoning, category=decision.category
        ),
    }


@logged_node(log, "reject")
def reject_node(state: RAGState) -> dict:
    decision = state["triage"]
    reason = decision.rejection_reason or (
        "This question falls outside the cyber-security corpus this system indexes."
    )
    return {
        "status": "rejected",
        "response": reason,
        "trace": add_trace(state, "reject", reason),
    }


@logged_node(log, "clarify")
def clarify_node(state: RAGState) -> dict:
    decision = state["triage"]
    question = decision.clarifying_question
    return {
        "status": "needs_clarification",
        "response": question,
        "trace": add_trace(state, "clarify", question),
    }


def _merge_evidence(existing: list[Evidence], fresh: list[Evidence]) -> list[Evidence]:
    """Union by chunk ID, preferring direct hits and then stronger scores."""
    by_id: dict[str, Evidence] = {e.chunk_id: e for e in existing}
    for item in fresh:
        current = by_id.get(item.chunk_id)
        if (
            current is None
            or (current.is_neighbor and not item.is_neighbor)
            or (
                current.is_neighbor == item.is_neighbor
                and item.rerank_score > current.rerank_score
            )
        ):
            by_id[item.chunk_id] = item
    return sorted(
        by_id.values(),
        key=lambda evidence: (evidence.is_neighbor, -evidence.rerank_score),
    )


@logged_node(log, "retrieve")
def retrieve_node(state: RAGState) -> dict:
    round_no = state.get("round", 0) + 1
    queries = state.get("search_queries") or [state["query"]]

    fresh = retrieval_agent.retrieve(
        queries,
        round_no=round_no,
        exclude_chunk_ids=set(state.get("rejected_chunk_ids", [])),
    )

    # On a retry, the previous round's evidence was judged insufficient - keep it
    # only alongside the new material, ranked together, capped at the budget.
    working_set_limit = max(
        settings.top_k_rerank,
        settings.neighbor_max_total_chunks,
    )
    merged = _merge_evidence(state.get("evidence", []), fresh)[:working_set_limit]

    return {
        "round": round_no,
        "evidence": merged,
        "tried_queries": [*state.get("tried_queries", []), *queries],
        "trace": add_trace(
            state, "retrieve", f"{len(fresh)} new, {len(merged)} in working set",
            queries=queries,
        ),
    }


@logged_node(log, "verify")
def verify_node(state: RAGState) -> dict:
    evidence = state.get("evidence", [])
    verdict = verifier_agent.verify(
        state["query"], evidence, tried_queries=state.get("tried_queries", [])
    )

    update: dict = {
        "verdict": verdict,
        "trace": add_trace(
            state, "verify", verdict.reasoning,
            sufficient=verdict.sufficient,
            supporting=verdict.supporting_indices,
        ),
    }

    if verdict.sufficient:
        # Narrow the working set to what the verifier actually endorsed, so the
        # answer agent is not tempted by passages already judged irrelevant.
        supported = [evidence[i - 1] for i in verdict.supporting_indices]
        update["evidence"] = supported or evidence
    else:
        # Remember what failed so the next round does not re-serve it.
        supporting = {evidence[i - 1].chunk_id for i in verdict.supporting_indices}
        rejected = [e.chunk_id for e in evidence if e.chunk_id not in supporting]
        update["rejected_chunk_ids"] = [
            *state.get("rejected_chunk_ids", []), *rejected
        ]
        if verdict.rewritten_queries:
            update["search_queries"] = verdict.rewritten_queries
        # Keep only endorsed passages in the working set for the retry.
        update["evidence"] = [evidence[i - 1] for i in verdict.supporting_indices]

    return update


@logged_node(log, "answer")
def answer_node(state: RAGState) -> dict:
    verdict = state.get("verdict")
    evidence = state.get("evidence", [])
    exhausted = verdict is not None and not verdict.sufficient

    if exhausted and not evidence:
        response = (
            "I could not find sufficient supporting material in the indexed documents "
            "to answer this reliably."
        )
        missing = getattr(verdict, "missing_information", "")
        if missing:
            response += f"\n\nWhat is missing: {missing}"
        return {
            "status": "insufficient",
            "response": response,
            "trace": add_trace(state, "answer", "aborted - no supporting evidence"),
        }

    caveat = ""
    if exhausted:
        caveat = (
            f"Retrieval was retried {state.get('round', 0)} times without fully covering "
            "the question; this answer reflects only the passages that were found."
        )

    payload = answer_agent.generate(state["query"], evidence, caveat_prefix=caveat)
    return {
        "answer": payload,
        "status": "answered_partial" if exhausted else "answered",
        "response": payload.answer,
        "trace": add_trace(
            state, "answer", f"confidence={payload.confidence}",
            citations=len(payload.citations),
        ),
    }


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_triage(state: RAGState) -> str:
    category = state["triage"].category
    target = {"out_of_scope": "reject", "ambiguous": "clarify"}.get(category, "retrieve")
    log_event(log, "state.route", router="triage", decision=category, target=target)
    return target


def route_verify(state: RAGState) -> str:
    verdict = state["verdict"]
    reason = "evidence_sufficient"
    target = "answer"
    if not verdict.sufficient:
        if state.get("round", 0) >= settings.max_retrieval_rounds:
            reason = "retrieval_budget_exhausted"
        elif not verdict.rewritten_queries:
            reason = "no_rewritten_queries"
        else:
            reason = "retry_with_rewrite"
            target = "retrieve"

    log_event(
        log,
        "state.route",
        router="verify",
        target=target,
        reason=reason,
        round=state.get("round", 0),
        rewritten_queries=payload(verdict.rewritten_queries),
    )
    return target


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("triage", triage_node)
    graph.add_node("reject", reject_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify", verify_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_triage,
        {"reject": "reject", "clarify": "clarify", "retrieve": "retrieve"},
    )
    graph.add_edge("reject", END)
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "verify")
    graph.add_conditional_edges(
        "verify", route_verify, {"retrieve": "retrieve", "answer": "answer"}
    )
    graph.add_edge("answer", END)

    return graph.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run(query: str) -> RAGState:
    """Run the full pipeline for one query."""
    request_id = new_request_id()
    token = set_request_id(request_id)
    started = perf_counter()
    recursion_limit = 4 * settings.max_retrieval_rounds + 10
    log_event(
        log,
        "pipeline.start",
        query=payload(query),
        recursion_limit=recursion_limit,
        max_retrieval_rounds=settings.max_retrieval_rounds,
    )
    try:
        # recursion_limit guards against a routing bug turning the retry loop
        # into a runaway; each round costs roughly three node visits.
        state = get_graph().invoke(
            initial_state(query),
            config={"recursion_limit": recursion_limit},
        )
    except Exception:
        log.exception(
            "pipeline.error | request_id=%s duration_ms=%.2f",
            request_id,
            (perf_counter() - started) * 1000,
        )
        raise
    else:
        log_event(
            log,
            "pipeline.end",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            final_state=state_summary(state),
        )
        return state
    finally:
        reset_request_id(token)
