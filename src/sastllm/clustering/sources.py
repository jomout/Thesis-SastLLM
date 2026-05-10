from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from sastllm.db import EmbeddingsManager

EmbeddingBatch = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class EmbeddingRecord:
    functionality_id: int
    vector: np.ndarray


class EmbeddingSource:
    """
    Re-iterable source for functionality embeddings.

    The factory must return a fresh iterable every time because fitting/searching
    may need to scan the same source multiple times.
    """

    def __init__(self, factory):
        self._factory = factory

    def records(self) -> Iterator[EmbeddingRecord]:
        for functionality_id, vector in self._factory():
            yield EmbeddingRecord(
                functionality_id=int(functionality_id),
                vector=np.asarray(vector, dtype=np.float32),
            )

    def batches(self, batch_size: int) -> Iterator[EmbeddingBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        ids: list[int] = []
        vectors: list[np.ndarray] = []

        for record in self.records():
            ids.append(record.functionality_id)
            vectors.append(record.vector)

            if len(vectors) == batch_size:
                yield np.asarray(ids, dtype=np.int64), np.asarray(vectors, dtype=np.float32)
                ids, vectors = [], []

        if vectors:
            yield np.asarray(ids, dtype=np.int64), np.asarray(vectors, dtype=np.float32)


class QdrantEmbeddingRepository:
    """Builds embedding sources backed by Qdrant."""

    def __init__(self, *, collection_name: str, embeddings_manager: EmbeddingsManager | None = None) -> None:
        self.collection_name = collection_name
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()

    def for_split(self, split: str) -> EmbeddingSource:
        return EmbeddingSource(
            lambda: self.embeddings_manager.get_embeddings_by_payload_field(
                collection_name=self.collection_name,
                field="split",
                values=[split],
            )
        )

    def first_n(self, n: int) -> EmbeddingSource:
        if n <= 0:
            raise ValueError("n must be > 0.")
        return EmbeddingSource(lambda: self.embeddings_manager.get_n_embeddings(collection_name=self.collection_name, n=n))

    def count_for_split(self, split: str) -> int:
        return self.embeddings_manager.count_embeddings_by_payload_field(
            collection_name=self.collection_name,
            field="split",
            values=[split],
        )


def normalize_batches(source: EmbeddingSource, batch_size: int) -> Iterable[EmbeddingBatch]:
    for ids, vectors in source.batches(batch_size):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        yield ids, vectors
