"""Agent 1 - Query Triage / Analyzer.

Gatekeeper for the whole graph. It answers three questions in one call: does
this belong to the corpus at all, is it answerable as phrased, and what should
actually be sent to the retriever. That last part matters more than it looks -
a user asking "we got hit, what now?" needs to become "ransomware incident
containment steps" before BM25 has anything to bite on.
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field

from src.agents import DOMAIN, AgentOutput
from src.llm import get_llm
from src.logging_utils import log_event, payload

log = logging.getLogger(__name__)

Category = Literal["in_scope", "ambiguous", "out_of_scope"]

SYSTEM = f"""You are the triage agent of a retrieval system grounded in {DOMAIN}

Classify the user's query into exactly one category:

- "in_scope": a cyber-security question the corpus can plausibly answer. Prefer
  this whenever the topic is cyber-security and the intent is clear, even if you
  are unsure the corpus contains the specific detail - retrieval decides that,
  not you.
- "ambiguous": the topic is cyber-security but the question cannot be retrieved
  against as written - a critical referent is missing (which system, which
  framework, which phase), or it is so broad that any answer would be arbitrary.
  Do NOT use this merely because the question is short.
- "out_of_scope": not a cyber-security question (cooking, sport, general coding
  help, personal chat), or it asks for something the corpus cannot supply -
  operational attack assistance, malware authoring, or help compromising a
  specific target.

Defensive and analytical security questions are IN SCOPE, including how attacks
work, how ransomware propagates, and how adversaries behave. Understanding
attacker behaviour is the basis of defence. Reserve "out_of_scope" for requests
seeking operational capability to attack, not for requests seeking understanding.

Then produce the retrieval plan:

- search_queries: for "in_scope", 2-3 self-contained search queries in English.
  Rewrite pronouns and context into explicit terms, and use the vocabulary these
  documents use (e.g. "containment", "eradication", "recovery", "playbook",
  "indicators of compromise"). Vary them - one close to the user's wording, one
  using standards terminology. Empty list for other categories.
- clarifying_question: for "ambiguous", one specific question that would unblock
  retrieval. Empty string otherwise.
- rejection_reason: for "out_of_scope", one sentence the user will read,
  explaining what this system does cover. Empty string otherwise.
"""


class TriageDecision(AgentOutput):
    category: Category = Field(description="in_scope | ambiguous | out_of_scope")
    reasoning: str = Field(description="One or two sentences justifying the category.")
    search_queries: list[str] = Field(description="2-3 retrieval queries, or [].")
    clarifying_question: str = Field(description="Question for the user, or ''.")
    rejection_reason: str = Field(description="Why it was rejected, or ''.")


def triage(query: str) -> TriageDecision:
    decision = get_llm().structured(
        system=SYSTEM,
        user=f"User query:\n{query}",
        schema=TriageDecision,
        effort="low",
    )

    # Guard against a model that classifies in_scope but forgets the plan.
    if decision.category == "in_scope" and not decision.search_queries:
        decision.search_queries = [query]
    if decision.category == "ambiguous" and not decision.clarifying_question:
        decision.clarifying_question = (
            "Could you narrow the question - which system, phase, or framework "
            "are you asking about?"
        )
    decision.search_queries = [q.strip() for q in decision.search_queries if q.strip()][:3]

    log_event(
        log,
        "agent.triage.output",
        category=decision.category,
        reasoning=payload(decision.reasoning),
        clarifying_question=payload(decision.clarifying_question),
        rejection_reason=payload(decision.rejection_reason),
    )
    if decision.search_queries:
        log_event(
            log,
            "query.rewrite",
            stage="triage",
            original=payload(query),
            rewritten=payload(decision.search_queries),
        )
    return decision
