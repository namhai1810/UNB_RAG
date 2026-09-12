"""Qdrant index: build it, and run the hybrid query against it.

One collection holds both a dense vector and a BM25 sparse vector per chunk, so
fusion happens inside Qdrant (prefetch both branches -> reciprocal rank fusion)
rather than in Python over two disconnected result lists.

Local mode keeps everything in `data/processed/qdrant` - no server, no Docker -
at the cost of a single-writer file lock, which is fine for one pipeline.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.config import settings
from src.ingestion.chunking import Chunk
from src.retrieval.dense import get_dense_encoder
from src.retrieval.sparse import SparseVector, get_sparse_encoder

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("6f1e4a1c-2c9a-4f4e-9f1a-1c0e5b7d3a21")


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, path: Path | None = None, collection: str | None = None) -> None:
        self.path = path or settings.qdrant_path
        self.collection = collection or settings.collection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    # ------------------------------------------------------------- lifecycle
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def recreate(self, dim: int) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF is applied by Qdrant at query time, which is what turns raw
                # term frequencies from FastEmbed into real BM25 scoring.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        log.info("Created collection %r (dim=%d)", self.collection, dim)

    # ---------------------------------------------------------------- upsert
    def upsert(
        self,
        chunks: list[Chunk],
        dense_vectors,
        sparse_vectors: list[SparseVector],
        batch_size: int = 128,
    ) -> None:
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    DENSE: dense.tolist() if hasattr(dense, "tolist") else list(dense),
                    SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=chunk.to_dict(),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ]
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[i : i + batch_size])
        log.info("Upserted %d points into %r", len(points), self.collection)

    # ---------------------------------------------------------------- search
    def hybrid_search(
        self,
        query: str,
        limit: int | None = None,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """Dense + BM25 prefetch, fused with RRF. Returns payloads plus a score."""
        limit = limit or settings.top_k_fused
        dense_vec = get_dense_encoder().encode_query(query)
        sparse_vec = get_sparse_encoder().encode_query(query)

        query_filter = (
            models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchAny(any=source_filter))]
            )
            if source_filter
            else None
        )

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using=DENSE,
                    limit=settings.top_k_dense,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values
                    ),
                    using=SPARSE,
                    limit=settings.top_k_sparse,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [{**point.payload, "fusion_score": point.score} for point in response.points]

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Load chunk payloads by their stable IDs without running a vector search."""
        if not chunk_ids:
            return []

        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
        by_chunk_id = {
            record.payload["chunk_id"]: record.payload
            for record in records
            if record.payload and record.payload.get("chunk_id")
        }
        return [by_chunk_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_chunk_id]


def build_index(chunks: list[Chunk], store: VectorStore | None = None) -> VectorStore:
    dense_encoder = get_dense_encoder()
    sparse_encoder = get_sparse_encoder()
    store = store or VectorStore()

    texts = [c.text for c in chunks]
    log.info("Embedding %d chunks (dense)...", len(texts))
    dense_vectors = dense_encoder.encode_passages(texts)
    log.info("Embedding %d chunks (sparse/BM25)...", len(texts))
    sparse_vectors = sparse_encoder.encode_passages(texts)

    store.recreate(dim=dense_encoder.dim)
    store.upsert(chunks, dense_vectors, sparse_vectors)
    return store
