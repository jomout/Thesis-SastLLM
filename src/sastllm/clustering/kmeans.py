from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from sastllm.configs import get_logger

from .sources import EmbeddingSource, normalize_batches

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusterAssignmentBatch:
    functionality_ids: np.ndarray
    cluster_ids: np.ndarray


class MiniBatchKMeansClusterer:
    """
    Streaming MiniBatchKMeans wrapper for functionality embeddings.

    It avoids materializing the full embedding matrix in memory during fit and
    prediction. The legacy `predict()` array API is retained for compatibility.
    """

    MODEL_KEY = "minibatch_kmeans_model"
    CLUSTERS_KEY = "n_clusters"

    def __init__(self, *, random_state: int = 42, plots_dir: str | Path = "cluster_plots") -> None:
        self.random_state = random_state
        self.plots_dir = Path(plots_dir)
        self.kmeans: MiniBatchKMeans | None = None
        self.n_clusters: int | None = None

    def find_optimal_k(
        self,
        source: EmbeddingSource,
        *,
        n: int,
        batch_size: int = 1000,
        min_samples_per_cluster: int = 20,
        num_candidates: int = 30,
        early_stop_patience: int = 4,
    ) -> int:
        if n <= 2:
            raise ValueError("n must be greater than 2 to search for k.")
        if min_samples_per_cluster <= 0:
            raise ValueError("min_samples_per_cluster must be > 0.")

        k_values = self._candidate_k_values(
            n=n,
            min_samples_per_cluster=min_samples_per_cluster,
            num_candidates=num_candidates,
        )
        inertias: list[float] = []
        stable_knee_steps = 0
        last_knee: int | None = None

        for k in tqdm(k_values, desc="Finding optimal k"):
            try:
                model = self._fit_new_model(source, k=k, batch_size=batch_size)
            except ValueError as e:
                logger.warning("Stopping k search at k=%s: %s", k, e)
                break
            inertias.append(float(model.inertia_))

            if len(inertias) >= 3:
                current_knee = self._locate_knee(k_values[: len(inertias)], inertias)
                stable_knee_steps = stable_knee_steps + 1 if current_knee is not None and current_knee == last_knee else 0
                last_knee = current_knee
                if stable_knee_steps >= early_stop_patience:
                    logger.info("Stopping k search early at stable knee=%s.", current_knee)
                    break

        if not inertias:
            raise RuntimeError("Unable to fit any candidate k values during clustering search.")

        optimal_k = self._locate_knee(k_values[: len(inertias)], inertias)
        if optimal_k is None:
            optimal_k = int(k_values[int(np.argmin(inertias))])
            logger.warning("KneeLocator did not find an elbow; using lowest-inertia candidate k=%s.", optimal_k)

        self.n_clusters = optimal_k
        self._plot_inertia(k_values[: len(inertias)], inertias, optimal_k, n=n)
        return optimal_k

    def fit(self, source: EmbeddingSource, *, k: int, batch_size: int = 1000) -> None:
        if k <= 0:
            raise ValueError("k must be > 0.")

        logger.info("Fitting MiniBatchKMeans with k=%s.", k)
        self.kmeans = self._fit_new_model(source, k=k, batch_size=batch_size)
        self.n_clusters = k
        logger.info("MiniBatchKMeans training completed.")

    def predict_batches(self, source: EmbeddingSource, *, batch_size: int = 1000) -> Iterator[ClusterAssignmentBatch]:
        if self.kmeans is None:
            raise RuntimeError("Model must be fit or loaded before prediction.")

        for ids, vectors in normalize_batches(source, batch_size):
            labels = self.kmeans.predict(vectors)
            yield ClusterAssignmentBatch(functionality_ids=ids, cluster_ids=labels.astype(np.int64))

    def predict(self, source: EmbeddingSource, *, batch_size: int = 1000) -> np.ndarray:
        batches = [
            np.column_stack((batch.functionality_ids, batch.cluster_ids))
            for batch in self.predict_batches(source, batch_size=batch_size)
        ]
        if not batches:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(batches, axis=0)

    def save_model(self, path: str | Path) -> None:
        if self.kmeans is None or self.n_clusters is None:
            raise RuntimeError("Model must be fit before saving.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                self.MODEL_KEY: self.kmeans,
                self.CLUSTERS_KEY: self.n_clusters,
                "metadata": {
                    "clusterer": self.__class__.__name__,
                    "random_state": self.random_state,
                },
            },
            model_path,
        )
        logger.info("Saved clustering model to %s.", model_path)

    def load_model(self, path: str | Path) -> None:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        payload: Any = joblib.load(model_path)
        if isinstance(payload, dict):
            model = payload.get(self.MODEL_KEY) or payload.get("model")
            n_clusters = payload.get(self.CLUSTERS_KEY) or payload.get("n_clusters")
        else:
            model = payload
            n_clusters = getattr(payload, "n_clusters", None)

        if not isinstance(model, MiniBatchKMeans):
            raise ValueError(f"Unsupported clustering artifact in {model_path}.")
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError(f"Missing or invalid cluster count in {model_path}.")

        self.kmeans = model
        self.n_clusters = n_clusters
        logger.info("Loaded clustering model from %s.", model_path)

    def _fit_new_model(self, source: EmbeddingSource, *, k: int, batch_size: int) -> MiniBatchKMeans:
        model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=self.random_state)
        warmup: list[np.ndarray] = []
        warmup_size = 0
        initialized = False

        for _, vectors in normalize_batches(source, batch_size):
            if not initialized:
                warmup.append(vectors)
                warmup_size += len(vectors)
                if warmup_size >= k:
                    model.fit(np.concatenate(warmup, axis=0))
                    initialized = True
                    warmup = []
            else:
                model.partial_fit(vectors)

        if not initialized:
            if not warmup:
                raise RuntimeError("Embedding source is empty.")
            warmup_matrix = np.concatenate(warmup, axis=0)
            if len(warmup_matrix) < k:
                raise ValueError(f"Cannot fit k={k}; only {len(warmup_matrix)} embeddings are available.")
            model.fit(warmup_matrix)

        return model

    @staticmethod
    def _candidate_k_values(*, n: int, min_samples_per_cluster: int, num_candidates: int) -> np.ndarray:
        k_max = max(2, n // min_samples_per_cluster)
        if k_max < 2:
            raise ValueError("Not enough samples to search for k.")
        if k_max == 2:
            return np.array([2], dtype=np.int64)
        return np.unique(np.logspace(np.log10(2), np.log10(k_max), num=num_candidates).astype(np.int64))

    @staticmethod
    def _locate_knee(k_values: np.ndarray, inertias: list[float]) -> int | None:
        if len(k_values) < 3 or len(inertias) < 3:
            return None
        knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing").knee
        return int(knee) if knee is not None else None

    def _plot_inertia(self, k_values: np.ndarray, inertias: list[float], elbow: int, *, n: int) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(k_values, inertias, marker="o", label="Inertia")
        plt.vlines(elbow, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", label="Selected k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.legend()
        plt.savefig(self.plots_dir / f"n_{n}_k_{elbow}.png")
        plt.close()
