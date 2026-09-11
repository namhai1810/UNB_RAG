"""Agent 3 - Evidence Verifier.

The only agent whose job is to say "no". It reads the retrieved chunks against
the question and decides whether an answer can be grounded in them - and if not,
what is missing and what to search for instead. Without this gate the answer
agent will always produce something, and "something" from thin evidence is
exactly the failure mode a cited RAG system exists to prevent.
"""
from __future__ import annotations

import logging

from pydantic import Field

from src.agents import DOMAIN, AgentOutput
from src.config import settings
from src.llm import get_llm
from src.logging_utils import log_event, payload
from src.retrieval import Evidence, format_evidence

log = logging.getLogger(__name__)

SYSTEM = f"""You are the evidence verification agent of a retrieval system grounded in {DOMAIN}

You are given a user question and numbered evidence passages retrieved from the
corpus. Judge ONLY whether these passages are sufficient to answer the question
in a grounded, citable way. You are not answering the question.

Rules:
- Judge sufficiency against what the user actually asked. A question with several
  parts needs evidence for each part; partial coverage is not sufficient.
- A passage that merely mentions the topic is not support. It must contain the
  substance needed to answer.
- Do not fill gaps from your own knowledge. If the passages do not say it, it is
  missing - even if you know the answer.
- Ignore passages that are on-topic but do not bear on the question.

Return:
- sufficient: true only if a well-grounded answer can be written from these
  passages alone.
- reasoning: one or two sentences on what is covered and what is not.
- supporting_indices: the [n] numbers that genuinely support an answer, best
  first. Empty if none do.
- missing_information: what is absent, concretely. Empty string when sufficient.
- rewritten_queries: when insufficient, 2-3 NEW search queries targeting the gap.
  They must differ meaningfully from the queries already tried - change the
  terminology, narrow to the specific sub-topic, or use the standards vocabulary
  these documents use. Empty list when sufficient.
"""


class VerificationVerdict(AgentOutput):
    sufficient: bool = Field(description="Can a grounded answer be written?")
    reasoning: str = Field(description="What is covered and what is not.")
    supporting_indices: list[int] = Field(description="1-based [n] markers that support.")
    missing_information: str = Field(description="Concrete gap, or ''.")
    rewritten_queries: list[str] = Field(description="2-3 new queries, or [].")


def verify(
    query: str,
    evidence: list[Evidence],
    tried_queries: list[str] | None = None,
) -> VerificationVerdict:
    if not evidence:
        return VerificationVerdict(
            sufficient=False,
            reasoning="Retrieval returned no passages.",
            supporting_indices=[],
            missing_information="No relevant passages were found in the corpus.",
            rewritten_queries=[],
        )

    tried = ", ".join(f'"{q}"' for q in (tried_queries or [])) or "(none recorded)"
    user = (
        f"Question:\n{query}\n\n"
        f"Search queries already tried: {tried}\n\n"
        f"Evidence passages:\n{format_evidence(evidence)}"
    )

    verdict = get_llm().structured(
        system=SYSTEM, user=user, schema=VerificationVerdict, effort="medium"
    )

    # Discard indices the model invented, then re-apply the coverage floor.
    valid = {i for i in range(1, len(evidence) + 1)}
    verdict.supporting_indices = [i for i in verdict.supporting_indices if i in valid]
    if verdict.sufficient and len(verdict.supporting_indices) < settings.min_supporting_chunks:
        # A single passage can be enough for a narrow factual question, but the
        # verifier must have said so explicitly; an empty list never is.
        if not verdict.supporting_indices:
            verdict.sufficient = False
            verdict.missing_information = (
                verdict.missing_information
                or "No passage was identified as actually supporting an answer."
            )

    log_event(
        log,
        "agent.verifier.output",
        sufficient=verdict.sufficient,
        reasoning=payload(verdict.reasoning),
        supporting_indices=verdict.supporting_indices,
        missing_information=payload(verdict.missing_information),
    )
    if verdict.rewritten_queries:
        log_event(
            log,
            "query.rewrite",
            stage="verifier",
            original=payload(query),
            tried_queries=payload(tried_queries or []),
            missing_information=payload(verdict.missing_information),
            rewritten=payload(verdict.rewritten_queries),
        )
    return verdict
