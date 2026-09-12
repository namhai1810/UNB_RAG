"""Graph-level behaviour: routing, the retry loop, and how it degrades."""
from __future__ import annotations

import pytest

from src.agents.triage_agent import TriageDecision
from src.graph import workflow
from src.graph.state import initial_state
from src.retrieval import Evidence


@pytest.fixture
def no_retrieval(monkeypatch, evidence_factory):
    """Replace the retriever with a counter that returns fresh evidence per round."""
    calls: list[dict] = []

    def fake_retrieve(queries, round_no=0, top_k=None, retriever=None,
                      exclude_chunk_ids=None):
        calls.append({"queries": list(queries), "round": round_no})
        return evidence_factory(3, prefix=f"r{round_no}c")

    monkeypatch.setattr(workflow.retrieval_agent, "retrieve", fake_retrieve)
    return calls


def run(query: str = "How do we contain an incident?"):
    return workflow.build_graph().invoke(initial_state(query))


def test_merge_prefers_a_direct_hit_over_the_same_neighbor_chunk():
    neighbor = Evidence(
        chunk_id="shared", source="guide.pdf", text="Context",
        rerank_score=0.0, is_neighbor=True, seed_chunk_id="seed",
    )
    direct = Evidence(
        chunk_id="shared", source="guide.pdf", text="Direct hit",
        rerank_score=0.7,
    )

    merged = workflow._merge_evidence([neighbor], [direct])

    assert merged == [direct]
    assert merged[0].is_neighbor is False


# ----------------------------------------------------------------- routing
def test_out_of_scope_is_rejected_without_retrieval(fake_llm, no_retrieval):
    fake_llm.queue(
        TriageDecision(
            category="out_of_scope",
            reasoning="Not cyber-security.",
            search_queries=[],
            clarifying_question="",
            rejection_reason="This system only covers cyber-security documents.",
        )
    )
    state = run("What is the best pho recipe?")
    assert state["status"] == "rejected"
    assert "cyber-security" in state["response"]
    assert no_retrieval == []          # never touched the index


def test_ambiguous_asks_for_clarification(fake_llm, no_retrieval):
    fake_llm.queue(
        TriageDecision(
            category="ambiguous",
            reasoning="No referent.",
            search_queries=[],
            clarifying_question="Which system was affected?",
            rejection_reason="",
        )
    )
    state = run("We got hit. What now?")
    assert state["status"] == "needs_clarification"
    assert state["response"] == "Which system was affected?"
    assert no_retrieval == []


def test_in_scope_uses_the_rewritten_queries(
    fake_llm, no_retrieval, triage_in_scope, verdict_sufficient, answer_ok
):
    fake_llm.queue(triage_in_scope, verdict_sufficient, answer_ok)
    state = run()
    assert state["status"] == "answered"
    assert no_retrieval[0]["queries"] == ["incident containment", "containment steps"]


# -------------------------------------------------------------- retry loop
def test_insufficient_evidence_triggers_a_rewrite_round(
    fake_llm, no_retrieval, triage_in_scope, verdict_insufficient,
    verdict_sufficient, answer_ok,
):
    fake_llm.queue(triage_in_scope, answer_ok)
    fake_llm.queue(verdict_insufficient, verdict_sufficient)
    state = run()
    assert state["status"] == "answered"
    assert len(no_retrieval) == 2
    # Round two searches what the verifier asked for, not the original queries.
    assert no_retrieval[1]["queries"] == verdict_insufficient.rewritten_queries


def test_retry_budget_is_respected(
    fake_llm, no_retrieval, triage_in_scope, verdict_insufficient, answer_ok
):
    fake_llm.queue(triage_in_scope, verdict_insufficient, answer_ok)
    state = run()
    assert len(no_retrieval) == workflow.settings.max_retrieval_rounds
    assert state["status"] == "answered_partial"
    assert "retried" in state["answer"].caveats.lower()


def test_loop_stops_when_the_verifier_offers_no_rewrite(
    fake_llm, no_retrieval, triage_in_scope, answer_ok
):
    from src.agents.verifier_agent import VerificationVerdict

    fake_llm.queue(
        triage_in_scope,
        VerificationVerdict(
            sufficient=False, reasoning="Nothing relevant.", supporting_indices=[1],
            missing_information="Not in corpus.", rewritten_queries=[],
        ),
        answer_ok,
    )
    state = run()
    assert len(no_retrieval) == 1      # no point retrying the same query
    assert state["status"] == "answered_partial"


def test_no_supporting_evidence_at_all_declines_to_answer(
    fake_llm, no_retrieval, triage_in_scope
):
    from src.agents.verifier_agent import VerificationVerdict

    fake_llm.queue(
        triage_in_scope,
        VerificationVerdict(
            sufficient=False, reasoning="Nothing supports this.", supporting_indices=[],
            missing_information="The corpus does not discuss this.", rewritten_queries=[],
        ),
    )
    state = run("How many rounds of loss estimation does the appendix mandate?")
    assert state["status"] == "insufficient"
    assert "does not discuss" in state["response"]


# ------------------------------------------------------------------ traces
def test_trace_records_every_node(
    fake_llm, no_retrieval, triage_in_scope, verdict_sufficient, answer_ok
):
    fake_llm.queue(triage_in_scope, verdict_sufficient, answer_ok)
    nodes = [event["node"] for event in run()["trace"]]
    assert nodes == ["triage", "retrieve", "verify", "answer"]
