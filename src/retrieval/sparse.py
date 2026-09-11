"""Sparse (BM25) retrieval encoder.

Lexical matching is not optional for this corpus: queries carry exact artifacts -
CVE IDs, "SP 800-61r3", registry paths, tool names - that a dense encoder happily
smooths away. FastEmbed's BM25 produces Qdrant-native sparse vectors, so both
branches live in one collection and are fused server-side.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

BM25_MODEL = "Qdrant/bm25"


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


class SparseEncoder:
    def __init__(self, model_name: str = BM25_MODEL) -> None:
        from fastembed import SparseTextEmbedding

        log.info("Loading sparse encoder %s", model_name)
        self.model_name = model_name
        self.model = SparseTextEmbedding(model_name=model_name)

    @staticmethod
    def _to_vector(emb) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in emb.indices],
            values=[float(v) for v in emb.values],
        )

    def encode_passages(self, passages: list[str]) -> list[SparseVector]:
        return [self._to_vector(e) for e in self.model.embed(passages)]

    def encode_query(self, query: str) -> SparseVector:
        # query_embed skips document-frequency weighting - correct for BM25 queries.
        return self._to_vector(next(iter(self.model.query_embed(query))))


@lru_cache(maxsize=1)
def get_sparse_encoder() -> SparseEncoder:
    return SparseEncoder()
