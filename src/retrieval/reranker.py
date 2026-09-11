"""Cross-encoder reranking (BGE-reranker-v2-m3).

Fusion gives recall; the cross-encoder gives precision. It reads the query and
the chunk together, so it is the only stage that can tell "how to contain
ransomware" from "how ransomware spreads" - a distinction the bi-encoder blurs
and the answer agent cannot recover from.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.config import settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.device
        log.info("Loading reranker %s on %s", self.model_name, self.device)
        self.model = CrossEncoder(self.model_name, device=self.device, max_length=1024)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(
            pairs, batch_size=settings.embedding_batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
