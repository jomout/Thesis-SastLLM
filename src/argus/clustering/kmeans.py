from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from argus.configs import get_logger

from .artifacts import export_clusterer_bundle, load_clusterer_bundle
from .sources import EmbeddingSource, normalize_batches

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusterAssignmentBatch:
    functionality_ids: np.ndarray
    cluster_ids: np.ndarray


class MiniBatchKMeansClusterer:
    """
    Streaming MiniBatchKMeans trainer with joblib restoration and ONNX export.

    It avoids materializing the full embedding matrix in memory during fit and
    prediction. The legacy `predict()` array API is retained for compatibility.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.kmeans: MiniBatchKMeans | None = None
        self.n_clusters: int | None = None
        self.embedding_dimension: int | None = None

    def fit(
        self,
        source: EmbeddingSource,
        *,
        k: int,
        batch_size: int = 1000,
        initialization_vectors: np.ndarray | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be > 0.")

        logger.info("Fitting MiniBatchKMeans", k=k, batch_size=batch_size)
        self.kmeans = self._fit_new_model(
            source,
            k=k,
            batch_size=batch_size,
            initialization_vectors=initialization_vectors,
        )
        self.n_clusters = k
        self.embedding_dimension = int(self.kmeans.cluster_centers_.shape[1])
        logger.info("MiniBatchKMeans training completed", k=k)

    def predict_batches(self, source: EmbeddingSource, *, batch_size: int = 1000) -> Iterator[ClusterAssignmentBatch]:
        if self.kmeans is None:
            raise RuntimeError("Model must be fit or loaded before prediction.")

        total = 0
        for ids, vectors in normalize_batches(source, batch_size):
            if self.embedding_dimension is not None and vectors.shape[1] != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: model expects {self.embedding_dimension}, got {vectors.shape[1]}.")

            labels = np.asarray(self.kmeans.predict(vectors), dtype=np.int64).reshape(-1)
            if len(labels) != len(ids):
                raise RuntimeError(f"Clusterer returned {len(labels)} labels for {len(ids)} embeddings.")
            total += len(ids)
            logger.debug("Predicted clustering batch", batch_size=len(ids), total=total)
            yield ClusterAssignmentBatch(functionality_ids=ids, cluster_ids=labels)
        logger.info("Cluster prediction completed", samples=total)

    def predict(self, source: EmbeddingSource, *, batch_size: int = 1000) -> np.ndarray:
        batches = [np.column_stack((batch.functionality_ids, batch.cluster_ids)) for batch in self.predict_batches(source, batch_size=batch_size)]
        if not batches:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(batches, axis=0)

    def save_model(self, model_dir: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        if self.kmeans is None or self.n_clusters is None:
            raise RuntimeError("Model must be fit before saving.")

        export_clusterer_bundle(
            model=self.kmeans,
            model_dir=model_dir,
            metadata=metadata,
            random_state=self.random_state,
        )

    def load_model(self, model_dir: str | Path, *, expected_metadata: Mapping[str, Any] | None = None) -> None:
        loaded = load_clusterer_bundle(model_dir, expected_metadata=expected_metadata)
        self.kmeans = loaded.model
        self.n_clusters = loaded.n_clusters
        self.embedding_dimension = loaded.embedding_dimension
        if loaded.random_state is not None:
            self.random_state = loaded.random_state
        logger.info(
            "Loaded joblib clustering model",
            model_dir=str(model_dir),
            n_clusters=self.n_clusters,
            embedding_dimension=self.embedding_dimension,
        )

    def _fit_new_model(
        self,
        source: EmbeddingSource,
        *,
        k: int,
        batch_size: int,
        initialization_vectors: np.ndarray | None = None,
    ) -> MiniBatchKMeans:
        start = perf_counter()
        model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=self.random_state)
        warmup: list[np.ndarray] = []
        warmup_size = 0
        initialized = initialization_vectors is not None
        if initialization_vectors is not None:
            if initialization_vectors.ndim != 2 or len(initialization_vectors) < k:
                raise ValueError(f"Initialization vectors must have shape (N, D) with N >= k={k}.")
            model.fit(initialization_vectors)
            logger.debug("Initialized MiniBatchKMeans from seeded reservoir", initialization_samples=len(initialization_vectors), k=k)
        batches = 0
        samples = 0
        vector_dim: int | None = None

        for _, vectors in normalize_batches(source, batch_size):
            batches += 1
            samples += len(vectors)
            vector_dim = int(vectors.shape[1]) if vectors.ndim == 2 else None
            if not initialized:
                warmup.append(vectors)
                warmup_size += len(vectors)
                if warmup_size >= k:
                    model.fit(np.concatenate(warmup, axis=0))
                    initialized = True
                    warmup = []
                    logger.debug("Initialized MiniBatchKMeans centers", warmup_samples=warmup_size, k=k)
            else:
                model.partial_fit(vectors)

        if not initialized:
            if not warmup:
                raise RuntimeError("Embedding source is empty.")
            warmup_matrix = np.concatenate(warmup, axis=0)
            if len(warmup_matrix) < k:
                raise ValueError(f"Cannot fit k={k}; only {len(warmup_matrix)} embeddings are available.")
            model.fit(warmup_matrix)

        logger.info(
            "MiniBatchKMeans fit pass completed",
            k=k,
            batches=batches,
            samples=samples,
            vector_dim=vector_dim,
            elapsed_seconds=round(perf_counter() - start, 3),
        )
        return model
