from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sastllm.configs import get_logger

from .sources import EmbeddingSource

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmbeddingSample:
    functionality_ids: np.ndarray
    vectors: np.ndarray
    population_size: int

    def __len__(self) -> int:
        return int(self.vectors.shape[0])


class ReservoirSampler:
    def __init__(self, *, sample_size: int, random_state: int) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0.")
        self.sample_size = sample_size
        self.rng = np.random.default_rng(random_state)
        self.ids: list[int] = []
        self.vectors: list[np.ndarray] = []
        self.seen = 0

    def add_batch(self, functionality_ids: np.ndarray, vectors: np.ndarray) -> None:
        for functionality_id, vector in zip(functionality_ids, vectors):
            if self.seen < self.sample_size:
                self.ids.append(int(functionality_id))
                self.vectors.append(np.asarray(vector, dtype=np.float32).copy())
            else:
                replacement_index = int(self.rng.integers(0, self.seen + 1))
                if replacement_index < self.sample_size:
                    self.ids[replacement_index] = int(functionality_id)
                    self.vectors[replacement_index] = np.asarray(vector, dtype=np.float32).copy()
            self.seen += 1

    def result(self) -> EmbeddingSample:
        if not self.vectors:
            logger.error("Reservoir sampling produced no vectors", population_size=self.seen, requested_size=self.sample_size)
            raise RuntimeError("Cannot sample an empty embedding source.")
        matrix = np.asarray(self.vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        sample = EmbeddingSample(
            functionality_ids=np.asarray(self.ids, dtype=np.int64),
            vectors=matrix,
            population_size=self.seen,
        )
        logger.info(
            "Completed reservoir sampling",
            population_size=sample.population_size,
            sample_size=len(sample),
            vector_dim=int(matrix.shape[1]),
        )
        return sample


def reservoir_sample(
    source: EmbeddingSource,
    *,
    sample_size: int,
    random_state: int,
) -> EmbeddingSample:
    """Return a uniform, seeded reservoir sample from a re-iterable source."""
    logger.info("Starting reservoir sampling", requested_size=sample_size, random_state=random_state)
    sampler = ReservoirSampler(sample_size=sample_size, random_state=random_state)
    for functionality_ids, vectors in source.batches(batch_size=1000):
        sampler.add_batch(functionality_ids, vectors)
    return sampler.result()
