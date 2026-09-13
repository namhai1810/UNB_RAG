"""Generate grounded retrieval test cases from the indexed corpus.

Each selected source chunk yields two or three English paraphrases of one
answerable question.  The resulting cases carry stable chunk-level relevance
labels and can be evaluated by ``evaluation.evaluate_retrieval``.

Example:
    python -m test_case_generation.generate_retrieval --count 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import settings
from src.llm import get_llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated_retrieval_test_cases.json"
QUESTION_STYLES = ("lexical", "semantic", "natural")


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
    questions: list[GeneratedQuestion] = Field(min_length=2, max_length=3)

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


SYSTEM_PROMPT = """You create a high-quality retrieval benchmark for a cyber-security RAG system.

For every supplied source chunk, identify ONE useful, self-contained fact or
procedure that is explicitly stated in that chunk. Then return:
- a concise English reference answer supported completely by that chunk;
- one to three short VERBATIM English evidence spans copied from the chunk;
- exactly the requested English questions, all asking for the same answer.

Question styles:
- lexical: retains important source terminology but does not copy a source sentence;
- semantic: changes wording and sentence structure substantially;
- natural: sounds like a real user's question and may naturally mix standard English cyber terms.

Every question must be self-contained. Do not refer to "the text", "the chunk",
"above", or an unspecified "it/this". Do not ask for information absent from the
chunk. Keep questions atomic: no multi-part or multi-hop questions. Preserve
official names, identifiers, version numbers, and negation exactly when material.
The reference answer must fully answer EVERY generated question; never make a question broader than the answer. Evidence spans must contain at least eight words and jointly support the complete answer, not merely a label or heading.
Use only the provided chunk IDs and return one item per requested chunk."""


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


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


def is_eligible_chunk(chunk: dict, *, min_tokens: int = 80) -> bool:
    """Reject front matter and fragments that produce misleading easy questions."""
    text = " ".join(str(chunk.get("text", "")).split())
    section = str(chunk.get("section", "")).casefold()
    excluded_sections = (
        "table of contents",
        "contents",
        "acknowledg",
        "keywords",
        "references",
        "how to cite",
        "change log",
        "change record",
        "revision history",
        "reports on computer systems technology",
        "disclaimer",
        "incident response recommendations and considerations for cybersecurity risk management",
    )
    return (
        int(chunk.get("n_tokens", 0)) >= min_tokens
        and len(text) >= 300
        and not any(term in section for term in excluded_sections)
    )


def select_chunks(
    chunks: list[dict], count: int, *, seed: int = 42, min_tokens: int = 80
) -> list[dict]:
    """Deterministically sample across source and paragraph/table strata."""
    if count < 1:
        raise ValueError("count must be at least 1")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        if is_eligible_chunk(chunk, min_tokens=min_tokens):
            key = (chunk.get("source", "unknown"), chunk.get("chunk_type", "paragraph"))
            strata[key].append(chunk)

    rng = random.Random(seed)
    queues: dict[tuple[str, str], deque[dict]] = {}
    for key, values in strata.items():
        values = list(values)
        rng.shuffle(values)
        queues[key] = deque(values)

    keys = sorted(queues)
    rng.shuffle(keys)
    selected: list[dict] = []
    while len(selected) < count and keys:
        remaining = []
        for key in keys:
            queue = queues[key]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(key)
        keys = remaining

    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} eligible chunks are available; requested {count}"
        )
    return selected


def question_counts(total: int, per_chunk: int = 3) -> list[int]:
    """Return 2-3 questions per chunk that sum to exactly ``total``."""
    if total < 2:
        raise ValueError("total must be at least 2")
    if per_chunk not in (2, 3):
        raise ValueError("per_chunk must be 2 or 3")
    groups = math.ceil(total / per_chunk)
    while groups * 2 > total:
        groups -= 1
    counts = [2] * groups
    for index in range(total - 2 * groups):
        counts[index] += 1
    if not counts or any(value not in (2, 3) for value in counts):
        raise ValueError(f"Cannot distribute {total} questions in groups of 2-3")
    return counts


def _expected_styles(count: int) -> set[str]:
    return {"semantic", "natural"} if count == 2 else set(QUESTION_STYLES)


def _request_batch(
    llm: StructuredLLM,
    chunks: list[dict],
    expected_counts: dict[str, int],
    attempt: int,
) -> GeneratedBatch:
    inputs = []
    for chunk in chunks:
        count = expected_counts[chunk["chunk_id"]]
        styles = [style for style in QUESTION_STYLES if style in _expected_styles(count)]
        inputs.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "requested_question_count": count,
                "required_styles": styles,
                "text": chunk["text"],
            }
        )
    user = (
        f"Generation attempt: {attempt}. Return exactly {len(inputs)} items.\n"
        "Input chunks:\n"
        + json.dumps(inputs, ensure_ascii=False, indent=2)
    )
    return llm.structured(
        system=SYSTEM_PROMPT,
        user=user,
        schema=GeneratedBatch,
        effort="medium",
    )


def _validate_generated(
    item: GeneratedChunkCases,
    chunk: dict,
    expected_count: int,
    used_queries: set[str],
) -> bool:
    if item.chunk_id != chunk["chunk_id"] or len(item.questions) != expected_count:
        return False
    if not item.reference_answer or len(item.reference_answer) < 12:
        return False
    styles = {question.paraphrase_type for question in item.questions}
    if styles != _expected_styles(expected_count):
        return False

    chunk_text = _normalise(chunk["text"])
    if not item.evidence_spans or any(
        len(_normalise(span)) < 25 or _normalise(span) not in chunk_text
        for span in item.evidence_spans
    ):
        return False

    queries = [_normalise(question.query) for question in item.questions]
    if any(len(query) < 12 or not query.endswith("?") for query in queries):
        return False
    if len(set(queries)) != len(queries) or any(query in used_queries for query in queries):
        return False
    forbidden = ("the chunk", "the passage", "the text", "the document")
    return not any(term in query for query in queries for term in forbidden)


def _acceptable_chunk_ids(item: GeneratedChunkCases, chunks: list[dict]) -> list[str]:
    """Include overlap copies containing the same verbatim supporting span."""
    spans = [_normalise(span) for span in item.evidence_spans]
    acceptable = {item.chunk_id}
    for chunk in chunks:
        text = _normalise(chunk["text"])
        if any(span in text for span in spans):
            acceptable.add(chunk["chunk_id"])
    return [item.chunk_id, *sorted(acceptable - {item.chunk_id})]


def generate_retrieval_dataset(
    *,
    total: int = 100,
    chunks: list[dict] | None = None,
    chunks_path: Path = DEFAULT_CHUNKS,
    llm: StructuredLLM | None = None,
    per_chunk: int = 3,
    batch_size: int = 4,
    max_attempts: int = 3,
    seed: int = 42,
    min_tokens: int = 80,
) -> dict:
    if batch_size < 1 or max_attempts < 1:
        raise ValueError("batch_size and max_attempts must be at least 1")

    all_chunks = chunks if chunks is not None else load_chunks(chunks_path)
    counts = question_counts(total, per_chunk)
    selected = select_chunks(all_chunks, len(counts), seed=seed, min_tokens=min_tokens)
    expected_counts = {
        chunk["chunk_id"]: count for chunk, count in zip(selected, counts)
    }
    by_id = {chunk["chunk_id"]: chunk for chunk in selected}
    pending = list(by_id)
    accepted: dict[str, GeneratedChunkCases] = {}
    used_queries: set[str] = set()
    backend = llm or get_llm()

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        next_pending: list[str] = []
        for start in range(0, len(pending), batch_size):
            ids = pending[start : start + batch_size]
            batch_chunks = [by_id[chunk_id] for chunk_id in ids]
            try:
                response = _request_batch(backend, batch_chunks, expected_counts, attempt)
            except Exception:
                if attempt == max_attempts:
                    raise
                next_pending.extend(ids)
                continue

            returned = {item.chunk_id: item for item in response.items}
            for chunk_id in ids:
                item = returned.get(chunk_id)
                if item is None or not _validate_generated(
                    item, by_id[chunk_id], expected_counts[chunk_id], used_queries
                ):
                    next_pending.append(chunk_id)
                    continue
                accepted[chunk_id] = item
                used_queries.update(_normalise(q.query) for q in item.questions)
        pending = next_pending

    if pending:
        raise RuntimeError(
            f"Could not generate valid cases for {len(pending)} chunks after "
            f"{max_attempts} attempts: {', '.join(pending)}"
        )

    cases = []
    group_number = 0
    case_number = 0
    for chunk in selected:
        group_number += 1
        item = accepted[chunk["chunk_id"]]
        acceptable = _acceptable_chunk_ids(item, all_chunks)
        for question in item.questions:
            case_number += 1
            cases.append(
                {
                    "id": f"ret_{case_number:03d}",
                    "group_id": f"ret_group_{group_number:03d}",
                    "type": "retrieval_grounded",
                    "query": question.query,
                    "paraphrase_type": question.paraphrase_type,
                    "reference_answer": item.reference_answer,
                    "primary_gold_chunk_id": item.chunk_id,
                    "acceptable_gold_chunk_ids": acceptable,
                    "gold": {
                        "doc_id": chunk.get("doc_id", ""),
                        "source": chunk.get("source", ""),
                        "section": chunk.get("section", ""),
                        "page_start": chunk.get("page_start", 0),
                        "page_end": chunk.get("page_end", 0),
                        "chunk_type": chunk.get("chunk_type", "paragraph"),
                        "content_sha256": _content_hash(chunk["text"]),
                        "evidence_spans": item.evidence_spans,
                    },
                }
            )

    if len(cases) != total:
        raise AssertionError(f"Generated {len(cases)} cases, expected {total}")

    corpus_bytes = (
        chunks_path.read_bytes()
        if chunks is None and chunks_path.exists()
        else "\n".join(json.dumps(c, sort_keys=True) for c in all_chunks).encode()
    )
    return {
        "metadata": {
            "schema_version": "1.0",
            "type": "retrieval_grounded",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": "en",
            "generator_model": getattr(backend, "model", getattr(backend, "name", "unknown")),
            "corpus_sha256": _sha256_bytes(corpus_bytes),
            "seed": seed,
            "case_count": len(cases),
            "group_count": len(selected),
            "questions_per_group": "2-3",
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate chunk-grounded retrieval cases.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--per-chunk", type=int, choices=(2, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=80)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = generate_retrieval_dataset(
        total=args.count,
        chunks_path=args.chunks,
        per_chunk=args.per_chunk,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        seed=args.seed,
        min_tokens=args.min_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {dataset['metadata']['case_count']} cases from "
        f"{dataset['metadata']['group_count']} chunks to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
