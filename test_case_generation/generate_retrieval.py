"""Generate schema-2 grounded cases split into single- and multi-gold sets.

Generation is deliberately two-stage. The LLM first extracts one atomic fact
from each candidate chunk. The complete corpus is then searched for other
chunks that may independently support the same reference answer. Exact copies
are accepted deterministically; semantic candidates are validated by the LLM
and must return their own verbatim evidence spans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import settings
from src.llm import get_llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated_retrieval_test_cases.json"
DEFAULT_METHODOLOGY = Path(__file__).resolve().parent / "README.md"
QUESTION_STYLES = ("lexical", "semantic", "natural")
MULTIPLICITIES = ("single_gold", "multi_gold")


class StructuredLLM(Protocol):
    def structured(self, system: str, user: str, schema, *, effort: str = "medium"):
        """Return a response validated against ``schema``."""


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paraphrase_type: Literal["lexical", "semantic", "natural"]
    query: str

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return " ".join(value.strip().split())


class GeneratedChunkCases(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    reference_answer: str
    evidence_spans: list[str] = Field(min_length=1, max_length=3)
    questions: list[GeneratedQuestion] = Field(min_length=3, max_length=3)

    @field_validator("reference_answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        return " ".join(value.strip().split())

    @field_validator("evidence_spans")
    @classmethod
    def clean_spans(cls, values: list[str]) -> list[str]:
        return [" ".join(value.strip().split()) for value in values if value.strip()]


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GeneratedChunkCases]


class SupportJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    support_level: Literal["full", "partial", "none"]
    evidence_spans: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("evidence_spans")
    @classmethod
    def clean_spans(cls, values: list[str]) -> list[str]:
        return [" ".join(value.strip().split()) for value in values if value.strip()]


class SupportBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SupportJudgment]


SYSTEM_PROMPT = """You create a high-quality retrieval benchmark for a cyber-security RAG system.

For every supplied source chunk, identify ONE useful, atomic, self-contained
fact or procedure explicitly stated in that chunk. Return a concise reference
answer, one to three short VERBATIM evidence spans, and exactly three questions
asking for that same answer.

Question styles:
- lexical: retains important source terminology without copying a sentence;
- semantic: changes wording and sentence structure substantially;
- natural: sounds like a realistic user question.

Questions must be self-contained, atomic, unique, and answerable completely by
the reference answer. Do not refer to an unspecified text, passage, chunk,
document, it, or this. Preserve official identifiers and negation. Evidence
spans must jointly support the complete answer and contain at least eight words.
Use only the supplied chunk IDs and return one item per chunk."""


SUPPORT_SYSTEM_PROMPT = """You validate alternative gold chunks for a retrieval benchmark.

For every request, decide whether the candidate chunk ALONE supports the entire
reference answer:
- full: enough information to answer the same atomic question completely;
- partial: supports only part of the answer;
- none: merely related or irrelevant.

Do not use outside knowledge and do not prefer the primary wording. For full or
partial support, copy one to three short VERBATIM evidence spans from the
candidate chunk. Return every request_id exactly once. Use an empty span list
for none."""


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "what", "when", "which", "with",
}


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value)} - _STOPWORDS


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def load_chunks(path: Path = DEFAULT_CHUNKS) -> list[dict]:
    chunks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("chunk_id") or not item.get("text"):
            raise ValueError(f"Invalid chunk at {path}:{line_number}")
        chunks.append(item)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def is_eligible_chunk(chunk: dict, *, min_tokens: int = 100) -> bool:
    text = " ".join(str(chunk.get("text", "")).split())
    section = str(chunk.get("section", "")).casefold()
    excluded_sections = (
        "table of contents", "contents", "acknowledg", "keywords", "references",
        "how to cite", "change log", "change record", "revision history",
        "reports on computer systems technology", "disclaimer",
        "incident response recommendations and considerations for cybersecurity risk management",
    )
    return (
        int(chunk.get("n_tokens", 0)) >= min_tokens
        and len(text) >= 300
        and not any(term in section for term in excluded_sections)
    )


def select_chunks(
    chunks: list[dict], count: int, *, seed: int = 43, min_tokens: int = 100
) -> list[dict]:
    """Deterministically sample across source and paragraph/table strata."""
    eligible = [chunk for chunk in chunks if is_eligible_chunk(chunk, min_tokens=min_tokens)]
    if count < 1 or count > len(eligible):
        raise ValueError(f"Requested {count} candidates; {len(eligible)} are eligible")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in eligible:
        strata[(chunk.get("source", "unknown"), chunk.get("chunk_type", "paragraph"))].append(chunk)

    rng = random.Random(seed)
    queues: dict[tuple[str, str], deque[dict]] = {}
    for key, values in strata.items():
        shuffled = list(values)
        rng.shuffle(shuffled)
        queues[key] = deque(shuffled)
    keys = sorted(queues)
    rng.shuffle(keys)

    selected: list[dict] = []
    while len(selected) < count and keys:
        remaining = []
        for key in keys:
            if queues[key] and len(selected) < count:
                selected.append(queues[key].popleft())
            if queues[key]:
                remaining.append(key)
        keys = remaining
    return selected


def question_counts(total: int) -> list[int]:
    """Return 2-3 questions per group that sum to exactly ``total``."""
    if total < 2:
        raise ValueError("Each multiplicity needs at least two questions")
    groups = math.ceil(total / 3)
    while groups * 2 > total:
        groups -= 1
    counts = [2] * groups
    for index in range(total - 2 * groups):
        counts[index] += 1
    if not counts or any(value not in (2, 3) for value in counts):
        raise ValueError(f"Cannot distribute {total} questions in groups of 2-3")
    return counts


def _request_generation_batch(
    llm: StructuredLLM, chunks: list[dict], attempt: int
) -> GeneratedBatch:
    inputs = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk.get("source", ""),
            "section": chunk.get("section", ""),
            "text": chunk["text"],
        }
        for chunk in chunks
    ]
    user = (
        f"Generation attempt: {attempt}. Return exactly {len(inputs)} items.\n"
        "Input chunks:\n" + json.dumps(inputs, ensure_ascii=False, indent=2)
    )
    return llm.structured(system=SYSTEM_PROMPT, user=user, schema=GeneratedBatch, effort="medium")


def _validate_generated(
    item: GeneratedChunkCases, chunk: dict, used_queries: set[str]
) -> bool:
    if item.chunk_id != chunk["chunk_id"] or len(item.questions) != 3:
        return False
    if len(item.reference_answer) < 12:
        return False
    if {question.paraphrase_type for question in item.questions} != set(QUESTION_STYLES):
        return False
    text = _normalise(chunk["text"])
    if not item.evidence_spans or any(
        len(_normalise(span)) < 25 or _normalise(span) not in text
        for span in item.evidence_spans
    ):
        return False
    queries = [_normalise(question.query) for question in item.questions]
    if any(len(query) < 12 or not query.endswith("?") for query in queries):
        return False
    if len(set(queries)) != 3 or any(query in used_queries for query in queries):
        return False
    forbidden = ("the chunk", "the passage", "the text", "the document")
    return not any(term in query for query in queries for term in forbidden)


def _generate_candidates(
    selected: list[dict], llm: StructuredLLM, *, batch_size: int, max_attempts: int
) -> dict[str, GeneratedChunkCases]:
    by_id = {chunk["chunk_id"]: chunk for chunk in selected}
    pending = list(by_id)
    accepted: dict[str, GeneratedChunkCases] = {}
    used_queries: set[str] = set()
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        next_pending: list[str] = []
        for start in range(0, len(pending), batch_size):
            ids = pending[start : start + batch_size]
            try:
                response = _request_generation_batch(llm, [by_id[item] for item in ids], attempt)
            except Exception:
                if attempt == max_attempts:
                    raise
                next_pending.extend(ids)
                continue
            returned = {item.chunk_id: item for item in response.items}
            for chunk_id in ids:
                item = returned.get(chunk_id)
                if item is None or not _validate_generated(item, by_id[chunk_id], used_queries):
                    next_pending.append(chunk_id)
                    continue
                accepted[chunk_id] = item
                used_queries.update(_normalise(question.query) for question in item.questions)
        pending = next_pending
    if pending:
        print(f"Skipping {len(pending)} candidates that failed generation validation")
    return accepted


def _candidate_pairs(
    items: dict[str, GeneratedChunkCases],
    chunks: list[dict],
    *,
    semantic_candidates: int,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, np.ndarray]]:
    """Return candidate IDs/reasons and embeddings used for the audit."""
    from src.retrieval.dense import get_dense_encoder

    encoder = get_dense_encoder()
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_vectors = encoder.encode_passages(chunk_texts, show_progress=True)
    item_ids = list(items)
    answer_vectors = encoder.encode_queries([items[item_id].reference_answer for item_id in item_ids])
    similarities = answer_vectors @ chunk_vectors.T
    chunk_index = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}

    candidates: dict[str, dict[str, set[str]]] = {}
    for row, item_id in enumerate(item_ids):
        item = items[item_id]
        primary_index = chunk_index[item_id]
        primary_spans = [_normalise(span) for span in item.evidence_spans]
        fact_tokens = _tokens(item.reference_answer + " " + " ".join(item.evidence_spans))
        found: dict[str, set[str]] = defaultdict(set)
        for chunk in chunks:
            candidate_id = chunk["chunk_id"]
            if candidate_id == item_id:
                continue
            text = _normalise(chunk["text"])
            if any(span in text for span in primary_spans):
                found[candidate_id].add("exact_span")
            candidate_tokens = _tokens(chunk["text"])
            if fact_tokens and len(fact_tokens & candidate_tokens) / len(fact_tokens) >= 0.72:
                found[candidate_id].add("lexical")

        scores = similarities[row].copy()
        scores[primary_index] = -1.0
        top_indices = np.argsort(scores)[::-1][:semantic_candidates]
        for index in top_indices:
            found[chunks[int(index)]["chunk_id"]].add("semantic")
        candidates[item_id] = dict(found)
    return candidates, {"chunks": chunk_vectors, "answers": answer_vectors}


def _request_support_batch(
    llm: StructuredLLM, requests: list[dict]
) -> SupportBatch:
    return llm.structured(
        system=SUPPORT_SYSTEM_PROMPT,
        user="Validation requests:\n" + json.dumps(requests, ensure_ascii=False, indent=2),
        schema=SupportBatch,
        effort="medium",
    )


def _valid_judgment(judgment: SupportJudgment, candidate: dict) -> bool:
    if judgment.support_level == "none":
        return not judgment.evidence_spans
    text = _normalise(candidate["text"])
    return bool(judgment.evidence_spans) and all(
        len(_normalise(span)) >= 25 and _normalise(span) in text
        for span in judgment.evidence_spans
    )


def _audit_support(
    items: dict[str, GeneratedChunkCases],
    chunks: list[dict],
    candidate_reasons: dict[str, dict[str, set[str]]],
    llm: StructuredLLM,
    *,
    validation_batch_size: int,
) -> dict[str, dict]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    audits: dict[str, dict] = {
        item_id: {"full": [], "partial": []} for item_id in items
    }
    requests: list[dict] = []
    request_meta: dict[str, tuple[str, str]] = {}

    for item_id, reasons_by_candidate in candidate_reasons.items():
        item = items[item_id]
        spans = [_normalise(span) for span in item.evidence_spans]
        for candidate_id, reasons in reasons_by_candidate.items():
            candidate = by_id[candidate_id]
            candidate_text = _normalise(candidate["text"])
            if spans and all(span in candidate_text for span in spans):
                audits[item_id]["full"].append(
                    {
                        "chunk_id": candidate_id,
                        "support_level": "full",
                        "match_kind": "exact_overlap",
                        "evidence_spans": item.evidence_spans,
                    }
                )
                continue
            request_id = f"support_{len(requests) + 1:05d}"
            request_meta[request_id] = (item_id, candidate_id)
            requests.append(
                {
                    "request_id": request_id,
                    "reference_answer": item.reference_answer,
                    "primary_evidence_spans": item.evidence_spans,
                    "candidate_chunk_id": candidate_id,
                    "candidate_source": candidate.get("source", ""),
                    "candidate_section": candidate.get("section", ""),
                    "candidate_text": candidate["text"],
                    "discovery_reasons": sorted(reasons),
                }
            )

    for start in range(0, len(requests), validation_batch_size):
        batch = requests[start : start + validation_batch_size]
        response = _request_support_batch(llm, batch)
        returned = {item.request_id: item for item in response.items}
        for request in batch:
            request_id = request["request_id"]
            judgment = returned.get(request_id)
            item_id, candidate_id = request_meta[request_id]
            if judgment is None or not _valid_judgment(judgment, by_id[candidate_id]):
                continue
            if judgment.support_level in {"full", "partial"}:
                audits[item_id][judgment.support_level].append(
                    {
                        "chunk_id": candidate_id,
                        "support_level": judgment.support_level,
                        "match_kind": "semantic_equivalent",
                        "evidence_spans": judgment.evidence_spans,
                    }
                )
    return audits


def _multiplicity_scope(primary: dict, full_alternatives: list[dict], by_id: dict[str, dict]) -> str:
    if not full_alternatives:
        return "unique"
    doc_ids = {primary.get("doc_id", "")}
    doc_ids.update(by_id[item["chunk_id"]].get("doc_id", "") for item in full_alternatives)
    if len(doc_ids) == 1:
        return "same_document"
    counts = Counter(
        [primary.get("doc_id", ""), *[by_id[item["chunk_id"]].get("doc_id", "") for item in full_alternatives]]
    )
    return "mixed" if any(count > 1 for count in counts.values()) else "cross_document"


def _select_balanced_groups(
    classified: list[dict], multiplicity: str, count: int, *, seed: int
) -> list[dict]:
    eligible = [item for item in classified if item["gold_multiplicity"] == multiplicity]
    if len(eligible) < count:
        raise RuntimeError(
            f"Need {count} {multiplicity} groups but discovered only {len(eligible)}. "
            "Increase --candidate-chunks or --semantic-candidates."
        )
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in eligible:
        chunk = item["primary_chunk"]
        strata[(chunk.get("source", "unknown"), chunk.get("chunk_type", "paragraph"))].append(item)
    rng = random.Random(seed)
    queues = {}
    for key, values in strata.items():
        values = list(values)
        rng.shuffle(values)
        queues[key] = deque(values)
    keys = sorted(queues)
    rng.shuffle(keys)
    selected = []
    while len(selected) < count:
        remaining = []
        for key in keys:
            if queues[key] and len(selected) < count:
                selected.append(queues[key].popleft())
            if queues[key]:
                remaining.append(key)
        keys = remaining
    return selected


def _gold_record(chunk: dict, *, support_level: str, match_kind: str, evidence_spans: list[str]) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk.get("doc_id", ""),
        "source": chunk.get("source", ""),
        "section": chunk.get("section", ""),
        "page_start": chunk.get("page_start", 0),
        "page_end": chunk.get("page_end", 0),
        "chunk_type": chunk.get("chunk_type", "paragraph"),
        "support_level": support_level,
        "match_kind": match_kind,
        "evidence_spans": evidence_spans,
    }


def _build_cases(
    selected_by_type: dict[str, list[dict]], questions_by_type: dict[str, int], by_id: dict[str, dict]
) -> list[dict]:
    cases = []
    case_number = 0
    for multiplicity in MULTIPLICITIES:
        if not selected_by_type[multiplicity]:
            continue
        counts = question_counts(questions_by_type[multiplicity])
        for group_index, (entry, question_count) in enumerate(
            zip(selected_by_type[multiplicity], counts), 1
        ):
            primary = entry["primary_chunk"]
            item = entry["generated"]
            gold_chunks = [
                _gold_record(
                    primary,
                    support_level="full",
                    match_kind="primary",
                    evidence_spans=item.evidence_spans,
                )
            ]
            for alternative in entry["audit"]["full"]:
                gold_chunks.append(
                    _gold_record(
                        by_id[alternative["chunk_id"]],
                        support_level="full",
                        match_kind=alternative["match_kind"],
                        evidence_spans=alternative["evidence_spans"],
                    )
                )
            partial_chunks = [
                _gold_record(
                    by_id[partial["chunk_id"]],
                    support_level="partial",
                    match_kind=partial["match_kind"],
                    evidence_spans=partial["evidence_spans"],
                )
                for partial in entry["audit"]["partial"]
            ]
            questions = item.questions if question_count == 3 else [
                question for question in item.questions if question.paraphrase_type != "lexical"
            ]
            for question in questions:
                case_number += 1
                cases.append(
                    {
                        "id": f"ret_{case_number:03d}",
                        "group_id": f"{multiplicity}_group_{group_index:03d}",
                        "type": "retrieval_grounded",
                        "gold_multiplicity": multiplicity,
                        "duplication_scope": entry["duplication_scope"],
                        "query": question.query,
                        "paraphrase_type": question.paraphrase_type,
                        "reference_answer": item.reference_answer,
                        "primary_gold_chunk_id": primary["chunk_id"],
                        "acceptable_gold_chunk_ids": [gold["chunk_id"] for gold in gold_chunks],
                        "partial_support_chunk_ids": [chunk["chunk_id"] for chunk in partial_chunks],
                        "gold_chunks": gold_chunks,
                        "partial_support_chunks": partial_chunks,
                        "gold": {
                            "doc_id": primary.get("doc_id", ""),
                            "source": primary.get("source", ""),
                            "section": primary.get("section", ""),
                            "page_start": primary.get("page_start", 0),
                            "page_end": primary.get("page_end", 0),
                            "chunk_type": primary.get("chunk_type", "paragraph"),
                            "content_sha256": _content_hash(primary["text"]),
                            "evidence_spans": item.evidence_spans,
                        },
                    }
                )
    return cases


def generate_retrieval_dataset(
    *,
    single_questions: int = 50,
    multi_questions: int = 50,
    candidate_chunks: int = 100,
    chunks: list[dict] | None = None,
    chunks_path: Path = DEFAULT_CHUNKS,
    llm: StructuredLLM | None = None,
    batch_size: int = 4,
    validation_batch_size: int = 8,
    semantic_candidates: int = 5,
    max_attempts: int = 3,
    seed: int = 43,
    min_tokens: int = 100,
) -> dict:
    if min(single_questions, multi_questions) < 2:
        raise ValueError("Each gold multiplicity must contain at least two questions")
    if min(batch_size, validation_batch_size, semantic_candidates, max_attempts) < 1:
        raise ValueError("Batch sizes, semantic candidates, and attempts must be positive")

    all_chunks = chunks if chunks is not None else load_chunks(chunks_path)
    selected = select_chunks(all_chunks, candidate_chunks, seed=seed, min_tokens=min_tokens)
    backend = llm or get_llm()
    generated = _generate_candidates(selected, backend, batch_size=batch_size, max_attempts=max_attempts)
    selected = [chunk for chunk in selected if chunk["chunk_id"] in generated]
    candidate_reasons, _ = _candidate_pairs(
        generated, all_chunks, semantic_candidates=semantic_candidates
    )
    audits = _audit_support(
        generated,
        all_chunks,
        candidate_reasons,
        backend,
        validation_batch_size=validation_batch_size,
    )
    by_id = {chunk["chunk_id"]: chunk for chunk in all_chunks}
    classified = []
    for primary in selected:
        item_id = primary["chunk_id"]
        audit = audits[item_id]
        multiplicity = "multi_gold" if audit["full"] else "single_gold"
        classified.append(
            {
                "primary_chunk": primary,
                "generated": generated[item_id],
                "audit": audit,
                "gold_multiplicity": multiplicity,
                "duplication_scope": _multiplicity_scope(primary, audit["full"], by_id),
            }
        )

    questions_by_type = {
        "single_gold": single_questions,
        "multi_gold": multi_questions,
    }
    selected_by_type = {
        multiplicity: _select_balanced_groups(
            classified,
            multiplicity,
            len(question_counts(questions_by_type[multiplicity])),
            seed=seed + index,
        )
        for index, multiplicity in enumerate(MULTIPLICITIES)
    }
    cases = _build_cases(selected_by_type, questions_by_type, by_id)

    corpus_bytes = (
        chunks_path.read_bytes()
        if chunks is None and chunks_path.exists()
        else "\n".join(json.dumps(chunk, sort_keys=True) for chunk in all_chunks).encode()
    )
    scope_counts = Counter(case["duplication_scope"] for case in cases)
    group_scope_counts = Counter(
        entry["duplication_scope"]
        for values in selected_by_type.values()
        for entry in values
    )
    return {
        "metadata": {
            "schema_version": "2.0",
            "type": "retrieval_grounded",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": "en",
            "generator_model": getattr(backend, "model", getattr(backend, "name", "unknown")),
            "embedding_model_for_candidate_discovery": settings.embedding_model,
            "corpus_sha256": _sha256_bytes(corpus_bytes),
            "seed": seed,
            "case_count": len(cases),
            "group_count": sum(len(values) for values in selected_by_type.values()),
            "candidate_chunk_count": len(selected),
            "semantic_candidates_per_fact": semantic_candidates,
            "questions_by_gold_multiplicity": questions_by_type,
            "groups_by_gold_multiplicity": {
                multiplicity: len(selected_by_type[multiplicity]) for multiplicity in MULTIPLICITIES
            },
            "questions_by_duplication_scope": dict(sorted(scope_counts.items())),
            "groups_by_duplication_scope": dict(sorted(group_scope_counts.items())),
            "questions_per_group": "2-3",
        },
        "cases": cases,
    }


def _count_table(counts: Counter | dict) -> str:
    lines = ["| Group | Count |", "|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    return "\n".join(lines)


def render_methodology(dataset: dict) -> str:
    metadata = dataset["metadata"]
    cases = dataset["cases"]
    source_counts = Counter(case["gold"]["source"] for case in cases)
    chunk_counts = Counter(case["gold"]["chunk_type"] for case in cases)
    style_counts = Counter(case["paraphrase_type"] for case in cases)
    gold_counts = Counter(len(case["acceptable_gold_chunk_ids"]) for case in cases)
    return f"""# Grounded retrieval dataset construction report

Generated at: `{metadata['created_at']}`

## Dataset identity

- Schema: **{metadata['schema_version']}**
- Cases: **{metadata['case_count']}** from **{metadata['group_count']}** atomic facts
- Candidate chunks audited: **{metadata['candidate_chunk_count']}**
- Corpus SHA-256: `{metadata['corpus_sha256']}`
- Fact/support validator: `{metadata['generator_model']}`
- Semantic candidate discovery: `{metadata['embedding_model_for_candidate_discovery']}`
- Random seed: **{metadata['seed']}**

## Reproduce the dataset

Prerequisites: the processed corpus must exist at `data/processed/chunks.jsonl`,
and the configured LLM plus BGE-M3 model must be available. From the repository
root, run:

```bash
python -m test_case_generation.generate_retrieval
```

By default this rewrites `generated_retrieval_test_cases.json` and this README
with the corpus fingerprint and measured distribution. Then regenerate the
combined result report:

```bash
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_end_to_end
# If a full-flow run was interrupted:
python -m evaluation.evaluate_end_to_end --resume
```

Run `python -m test_case_generation.generate_retrieval --help` to change quotas,
seed, candidate count, input chunks, or output paths. Generation is model-assisted,
so review the resulting evidence labels before treating a regenerated set as a
benchmark baseline.

## Gold multiplicity definitions

- `single_gold`: exactly one chunk in the audited corpus independently supports
  the complete reference answer.
- `multi_gold`: two or more chunks independently support the complete reference
  answer.
- `same_document`: all full-support chunks belong to one document.
- `cross_document`: full-support chunks occur in different documents, with one
  occurrence per represented document.
- `mixed`: full-support chunks span documents and at least one document contains
  multiple occurrences.

`primary_gold_chunk_id` records which chunk generated the fact. All entries in
`acceptable_gold_chunk_ids` have equal full relevance; the primary is not more
correct than a validated alternative.

## Construction procedure

1. Load the complete processed corpus and reject metadata/front-matter fragments.
2. Deterministically sample eligible candidate chunks across `(source, chunk_type)` strata.
3. Generate one atomic reference answer, verbatim primary evidence, and lexical,
   semantic, and natural questions for every candidate chunk.
4. Search the complete corpus for alternative support using exact evidence-span
   matches, lexical fact-token coverage, and the top
   {metadata['semantic_candidates_per_fact']} BGE-M3 semantic candidates.
5. Accept exact copies only when every primary evidence span occurs in the
   candidate. Send other candidates to the support validator as `full`, `partial`,
   or `none`; full/partial judgments must provide candidate-local verbatim evidence.
6. Classify facts by the number and document scope of independently full-supporting chunks.
7. Sample balanced single- and multi-gold groups, then assign two or three
   paraphrases per group to reach the requested question totals.

Candidate discovery is intentionally high-recall and validation is independent
of retrieval rank. “Unique” is therefore an operational label under this audit,
not a mathematical proof that no human could find another equivalent passage.

## Final distribution

### Questions by gold multiplicity

{_count_table(metadata['questions_by_gold_multiplicity'])}

### Source groups

{_count_table(source_counts)}

### Primary chunk types

{_count_table(chunk_counts)}

### Paraphrase styles

{_count_table(style_counts)}

### Duplication scope

{_count_table(metadata['questions_by_duplication_scope'])}

### Number of acceptable gold chunks per question

{_count_table({str(key): value for key, value in gold_counts.items()})}

## Schema and scoring contract

Every case stores `gold_multiplicity`, `duplication_scope`, a provenance-only
`primary_gold_chunk_id`, all full-support `gold_chunks`, and separately labelled
`partial_support_chunks`. The compatibility field `acceptable_gold_chunk_ids`
contains only full-support IDs.

For single-gold cases, Recall@K is equivalent to Hit@K. For multi-gold cases,
the evaluator reports both Any-Gold Hit@K (answerability) and Gold Coverage@K
(how many repeated occurrences were found). MRR uses the first full gold. nDCG
assigns grade 2 to every full gold, grade 1 to partial support, and grade 0 to all
other chunks.

Citation correctness accepts a citation to any full gold chunk. It does not
require citing every repeated occurrence and does not by itself establish
claim-level entailment.

## Quality controls and limitations

- Primary and alternative evidence spans must occur verbatim in their labelled chunks.
- Every multi-gold case has at least two full-support chunks; every single-gold
  case has exactly one after the audit.
- Two to three paraphrases from one fact are correlated, so reports include both
  question and group counts and keep multiplicity results separate.
- Synthetic questions may be cleaner than production traffic.
- Semantic candidate discovery uses an embedding model only to propose passages;
  it never automatically assigns semantic gold relevance.
- Support validation is model-assisted. High-impact benchmark releases should
  still receive human review, particularly all multi-gold alternatives and a
  sample of operationally unique cases.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate schema-2 single/multi-gold cases")
    parser.add_argument("--single-questions", type=int, default=50)
    parser.add_argument("--multi-questions", type=int, default=50)
    parser.add_argument("--candidate-chunks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--semantic-candidates", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--methodology-report", type=Path, default=DEFAULT_METHODOLOGY)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = generate_retrieval_dataset(
        single_questions=args.single_questions,
        multi_questions=args.multi_questions,
        candidate_chunks=args.candidate_chunks,
        chunks_path=args.chunks,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        semantic_candidates=args.semantic_candidates,
        max_attempts=args.max_attempts,
        seed=args.seed,
        min_tokens=args.min_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.methodology_report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.methodology_report.write_text(render_methodology(dataset), encoding="utf-8")
    print(
        f"Wrote {dataset['metadata']['case_count']} schema-2 cases to {args.output}\n"
        f"Wrote dataset construction report to {args.methodology_report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
