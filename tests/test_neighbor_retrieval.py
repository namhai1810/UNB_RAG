"""Adjacent-chunk expansion after seed retrieval and reranking."""
from __future__ import annotations

from src.retrieval import Evidence, HybridRetriever


class FakeStore:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = {item["chunk_id"]: item for item in payloads}
        self.requested: list[str] = []

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        self.requested = list(chunk_ids)
        return [self.payloads[item] for item in reversed(chunk_ids) if item in self.payloads]


class FakeReranker:
    pass


def _payload(index: int, section: str = "Containment") -> dict:
    return {
        "chunk_id": f"doc-00001-{index:05d}",
        "doc_id": "doc-00001",
        "chunk_index": index,
        "source": "guide.pdf",
        "section": section,
        "page_start": index + 1,
        "page_end": index + 1,
        "text": f"Passage {index}",
    }


def _seed(index: int, score: float = 0.8, section: str = "Containment") -> Evidence:
    return Evidence(
        chunk_id=f"doc-00001-{index:05d}",
        doc_id="doc-00001",
        chunk_index=index,
        source="guide.pdf",
        section=section,
        text=f"Seed {index}",
        rerank_score=score,
        round=2,
    )


def _retriever(payloads: list[dict]) -> tuple[HybridRetriever, FakeStore]:
    store = FakeStore(payloads)
    return HybridRetriever(store=store, reranker=FakeReranker()), store


def test_neighbors_are_appended_without_displacing_ranked_seeds():
    retriever, _ = _retriever([_payload(0), _payload(2)])
    result = retriever.expand_neighbors([_seed(1)], window=1, max_total=3)

    assert [item.chunk_index for item in result] == [1, 0, 2]
    assert result[0].is_neighbor is False
    assert all(item.is_neighbor for item in result[1:])
    assert all(item.seed_chunk_id == result[0].chunk_id for item in result[1:])
    assert all(item.round == 2 for item in result)


def test_adjacent_seeds_do_not_duplicate_each_other():
    retriever, store = _retriever([_payload(0), _payload(3)])
    result = retriever.expand_neighbors([_seed(1), _seed(2)], window=1, max_total=6)

    assert [item.chunk_index for item in result] == [1, 2, 0, 3]
    assert store.requested == ["doc-00001-00000", "doc-00001-00003"]


def test_document_start_and_missing_end_are_safe():
    retriever, store = _retriever([_payload(1)])
    result = retriever.expand_neighbors([_seed(0)], window=1, max_total=3)

    assert [item.chunk_index for item in result] == [0, 1]
    assert store.requested == ["doc-00001-00001"]


def test_section_boundary_and_excluded_chunks_are_not_added():
    retriever, store = _retriever([_payload(0, section="Preparation"), _payload(2)])
    result = retriever.expand_neighbors(
        [_seed(1)],
        window=1,
        max_total=3,
        same_section_only=True,
        exclude_chunk_ids={"doc-00001-00002"},
    )

    assert [item.chunk_index for item in result] == [1]
    assert store.requested == ["doc-00001-00000"]


def test_zero_window_skips_the_store_lookup():
    retriever, store = _retriever([_payload(0), _payload(2)])
    seed = _seed(1)

    assert retriever.expand_neighbors([seed], window=0) == [seed]
    assert store.requested == []


def test_legacy_payload_derives_index_from_stable_chunk_id():
    payload = _payload(7)
    payload.pop("chunk_index")

    evidence = HybridRetriever._to_evidence(payload, round_no=1)

    assert evidence.chunk_index == 7
