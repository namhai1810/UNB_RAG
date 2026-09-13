from __future__ import annotations

import json

from test_case_generation.generate_retrieval import (
    GeneratedBatch,
    GeneratedChunkCases,
    GeneratedQuestion,
    generate_retrieval_dataset,
    question_counts,
)


def _chunk(index: int, source: str = "guide.pdf", chunk_type: str = "paragraph") -> dict:
    evidence = f"Organizations must preserve audit logs for incident {index} before recovery begins."
    return {
        "chunk_id": f"doc-{index:05d}",
        "doc_id": "doc",
        "source": source,
        "section": f"Section {index}",
        "page_start": index,
        "page_end": index,
        "chunk_type": chunk_type,
        "n_tokens": 100,
        "text": (evidence + " Additional grounded cybersecurity guidance.") * 5,
    }


class FakeLLM:
    model = "fake-model"

    def structured(self, system, user, schema, *, effort="medium"):
        inputs = json.loads(user.split("Input chunks:\n", 1)[1])
        items = []
        for value in inputs:
            chunk_id = value["chunk_id"]
            count = value["requested_question_count"]
            index = int(chunk_id.rsplit("-", 1)[1])
            styles = ["semantic", "natural"] if count == 2 else ["lexical", "semantic", "natural"]
            items.append(
                GeneratedChunkCases(
                    chunk_id=chunk_id,
                    reference_answer=f"Audit logs for incident {index} must be preserved before recovery.",
                    evidence_spans=[
                        f"Organizations must preserve audit logs for incident {index} before recovery begins."
                    ],
                    questions=[
                        GeneratedQuestion(
                            paraphrase_type=style,
                            query=f"How should audit logs for incident {index} be handled before recovery ({style})?",
                        )
                        for style in styles
                    ],
                )
            )
        return GeneratedBatch(items=items)


def test_question_counts_make_exact_groups_of_two_or_three():
    counts = question_counts(100, 3)
    assert sum(counts) == 100
    assert len(counts) == 34
    assert set(counts) == {2, 3}


def test_generate_retrieval_dataset_is_flat_and_grounded():
    chunks = [_chunk(1), _chunk(2, chunk_type="table")]
    dataset = generate_retrieval_dataset(
        total=4, chunks=chunks, llm=FakeLLM(), batch_size=2, min_tokens=10
    )
    assert dataset["metadata"]["case_count"] == 4
    assert dataset["metadata"]["group_count"] == 2
    assert len(dataset["cases"]) == 4
    assert {case["paraphrase_type"] for case in dataset["cases"]} == {"semantic", "natural"}
    assert all(case["primary_gold_chunk_id"] in case["acceptable_gold_chunk_ids"] for case in dataset["cases"])
    assert all(case["reference_answer"] for case in dataset["cases"])
