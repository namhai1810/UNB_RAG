"""Dense retrieval encoder.

Symmetric encoders such as BGE-M3 need no prompt. Instruction-aware encoders
such as Qwen3-Embedding can select their model-provided query prompt through
``EMBEDDING_QUERY_PROMPT_NAME``. Queries and passages retain separate length
budgets because a question is short and a corpus chunk is not.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.config import settings

log = logging.getLogger(__name__)


class DenseEncoder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        log.info("Loading dense encoder %s on %s", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(
        self,
        texts: list[str],
        max_length: int,
        show_progress: bool,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        self.model.max_seq_length = max_length
        prompt_kwargs = {"prompt_name": prompt_name} if prompt_name else {}
        return self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,       # cosine == dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            **prompt_kwargs,
        )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(
            queries,
            settings.query_max_length,
            show_progress=False,
            prompt_name=settings.embedding_query_prompt_name,
        )

    def encode_passages(self, passages: list[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(passages, settings.passage_max_length, show_progress)

    def encode_query(self, query: str) -> list[float]:
        return self.encode_queries([query])[0].tolist()


@lru_cache(maxsize=1)
def get_dense_encoder() -> DenseEncoder:
    return DenseEncoder()
