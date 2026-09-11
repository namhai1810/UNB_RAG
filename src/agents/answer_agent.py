"""Agent 4 - Answer Generator.

Writes the final answer using only the verified passages, with inline [n]
markers. Every marker is checked against the evidence list afterwards: a citation
the system cannot resolve is worse than no citation, because it looks verifiable
and is not.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import Field

from src.agents import DOMAIN, AgentOutput
from src.llm import get_llm
from src.retrieval import Evidence, format_evidence

log = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d+)\]")

SYSTEM = f"""You are the answer agent of a retrieval system grounded in {DOMAIN}

Write an answer to the user's question using ONLY the numbered evidence passages
provided. This is a grounded-answering task, not a knowledge task.

Grounding rules - these are absolute:
- Every factual claim must be traceable to a passage, and must carry the marker
  of that passage, written as [n], immediately after the claim.
- Never state anything the passages do not support, however confident you are.
- Never cite a number that is not in the evidence list.
- If the passages only partially cover the question, answer the covered part and
  say plainly which part is not covered by the sources.
- When sources use precise standards language (phase names, control identifiers,
  document numbers), keep their wording rather than paraphrasing it away.

Style:
- Answer directly, then support. No preamble, no restating the question.
- Use short paragraphs; use a list only when the source material is itself a
  sequence of steps or items.
- Match the depth of the question - a definition needs a sentence, a "how do we
  respond" question needs the actual procedure.

Return:
- answer: the answer text with inline [n] markers.
- used_indices: every marker number used in `answer`.
- confidence: "high" when the passages directly and completely answer the
  question; "medium" when the answer requires stitching or mild inference from
  what is stated; "low" when coverage is thin.
- caveats: limitations a reader should know - uncovered sub-questions, source
  scope, or dated guidance. Empty string if there are none.
"""


class GeneratedAnswer(AgentOutput):
    answer: str = Field(description="Answer text with inline [n] citation markers.")
    used_indices: list[int] = Field(description="Marker numbers used in the answer.")
    confidence: Literal["high", "medium", "low"] = Field(description="Grounding confidence.")
    caveats: str = Field(description="Limitations, or ''.")


class Citation(AgentOutput):
    marker: int
    source: str
    section: str
    pages: str
    chunk_id: str
    quote: str = Field(description="Opening of the cited passage, for spot-checking.")


class AnswerPayload(AgentOutput):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    caveats: str
    dropped_markers: list[int] = Field(
        default_factory=list, description="Markers removed because they resolved to nothing."
    )


def _build_citations(evidence: list[Evidence], markers: list[int]) -> list[Citation]:
    citations = []
    for marker in sorted(set(markers)):
        item = evidence[marker - 1]
        pages = (
            f"p.{item.page_start}"
            if item.page_start == item.page_end
            else f"pp.{item.page_start}-{item.page_end}"
        )
        quote = " ".join(item.text.split())[:220]
        citations.append(
            Citation(
                marker=marker,
                source=item.source,
                section=item.section,
                pages=pages,
                chunk_id=item.chunk_id,
                quote=quote + ("..." if len(quote) == 220 else ""),
            )
        )
    return citations


def generate(
    query: str,
    evidence: list[Evidence],
    caveat_prefix: str = "",
) -> AnswerPayload:
    """Generate a grounded answer; `caveat_prefix` is prepended to the caveats.

    The prefix is how the graph reports a degraded run - e.g. the retrieval
    budget ran out - without the answer agent having to know about the loop.
    """
    if not evidence:
        return AnswerPayload(
            answer=(
                "I could not find anything in the indexed documents that answers this "
                "question, so I have nothing to ground an answer in."
            ),
            citations=[],
            confidence="low",
            caveats=caveat_prefix,
        )

    user = f"Question:\n{query}\n\nEvidence passages:\n{format_evidence(evidence)}"
    result = get_llm().structured(
        system=SYSTEM, user=user, schema=GeneratedAnswer, effort="high"
    )

    # Trust the text, not the model's self-report: markers actually present in
    # the answer are what readers follow.
    in_text = [int(m) for m in _MARKER_RE.findall(result.answer)]
    valid_range = range(1, len(evidence) + 1)
    dropped = sorted({m for m in in_text if m not in valid_range})

    if dropped:
        log.warning("Dropping unresolvable citation markers: %s", dropped)
        result.answer = _MARKER_RE.sub(
            lambda m: "" if int(m.group(1)) in dropped else m.group(0), result.answer
        )
        result.answer = re.sub(r" +([.,;:])", r"\1", result.answer)
        result.answer = re.sub(r" {2,}", " ", result.answer)

    markers = [m for m in in_text if m in valid_range]
    # Fall back to the model's own list if it wrote no inline markers at all.
    if not markers:
        markers = [m for m in result.used_indices if m in valid_range]

    caveats = " ".join(x for x in (caveat_prefix, result.caveats) if x).strip()
    confidence = result.confidence
    if not markers:
        confidence = "low"
        caveats = (caveats + " The answer could not be tied to specific passages.").strip()

    return AnswerPayload(
        answer=result.answer.strip(),
        citations=_build_citations(evidence, markers),
        confidence=confidence,
        caveats=caveats,
        dropped_markers=dropped,
    )
