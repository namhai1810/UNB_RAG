"""A scripted stand-in for the LLM, so graph behaviour can be tested offline.

The agent layer only ever asks the provider for a validated Pydantic object, so a
fake that returns pre-built objects exercises the real routing, retry, and
citation logic without a model, an API key, or a GPU.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.agents.answer_agent import GeneratedAnswer
from src.agents.triage_agent import TriageDecision
from src.agents.verifier_agent import VerificationVerdict
from src.retrieval import Evidence


class FakeLLM:
    """Returns queued responses, matched by schema."""

    name = "fake"

    def __init__(self) -> None:
        self.queues: dict[str, list[BaseModel]] = {}
        self.calls: list[tuple[str, str]] = []

    def queue(self, *responses: BaseModel) -> "FakeLLM":
        for response in responses:
            self.queues.setdefault(type(response).__name__, []).append(response)
        return self

    def structured(self, system, user, schema, *, effort="medium"):
        self.calls.append((schema.__name__, user))
        queue = self.queues.get(schema.__name__)
        if not queue:
            raise AssertionError(f"FakeLLM has no queued {schema.__name__} response")
        # Repeat the last response once the queue is down to one, so a retry loop
        # does not need every round scripted.
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def complete(self, system, user, *, effort="medium"):
        return "fake text"


@pytest.fixture
def fake_llm(monkeypatch):
    from src.agents import answer_agent, triage_agent, verifier_agent

    llm = FakeLLM()
    for module in (triage_agent, verifier_agent, answer_agent):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
    return llm


@pytest.fixture
def evidence_factory():
    def make(n: int, prefix: str = "c") -> list[Evidence]:
        return [
            Evidence(
                chunk_id=f"{prefix}{i}",
                source="NIST.SP.800-61r3.pdf",
                section="2.1. Incident Response Life Cycle Model",
                page_start=10 + i,
                page_end=10 + i,
                text=f"Passage {i} about incident response.",
                rerank_score=1.0 - i * 0.1,
            )
            for i in range(1, n + 1)
        ]

    return make


@pytest.fixture
def triage_in_scope():
    return TriageDecision(
        category="in_scope",
        reasoning="Cyber-security question.",
        search_queries=["incident containment", "containment steps"],
        clarifying_question="",
        rejection_reason="",
    )


@pytest.fixture
def verdict_sufficient():
    return VerificationVerdict(
        sufficient=True,
        reasoning="Covered.",
        supporting_indices=[1, 2],
        missing_information="",
        rewritten_queries=[],
    )


@pytest.fixture
def verdict_insufficient():
    return VerificationVerdict(
        sufficient=False,
        reasoning="Only tangential.",
        supporting_indices=[1],
        missing_information="No containment procedure.",
        rewritten_queries=["eradication procedure", "attacker persistence removal"],
    )


@pytest.fixture
def answer_ok():
    return GeneratedAnswer(
        answer="Containment limits the damage [1] and precedes eradication [2].",
        used_indices=[1, 2],
        confidence="high",
        caveats="",
    )
