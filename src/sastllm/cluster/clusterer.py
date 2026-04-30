from pathlib import Path
from typing import Iterator, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from sastllm.configs import get_logger
from sastllm.db.batch_datasource import BatchDataSource

logger = get_logger()


class Clusterer:
    def __init__(self, *, plots_dir: str = "cluster_plots"):
        logger.debug("Initializing Clusterer.")
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.n_clusters: Optional[int] = None
        self.plots_dir = Path(plots_dir)
        logger.debug("Clusterer initialized.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        return normalize(X, norm="l2", axis=1)

    def _iter_batches(
        self,
        data_source: BatchDataSource,
        batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yields (ids_batch, X_batch) without ever holding the full dataset in RAM.
        Each X_batch is already L2-normalised.
        """
        ids_buf: list[int] = []
        vecs_buf: list[np.ndarray] = []

        for ident, vec in data_source.iter():
            ids_buf.append(ident)
            vecs_buf.append(vec)

            if len(vecs_buf) == batch_size:
                X = self._preprocess(np.array(vecs_buf, dtype=np.float32))
                yield np.array(ids_buf, dtype=np.int64), X
                ids_buf, vecs_buf = [], []

        # Flush remainder
        if vecs_buf:
            X = self._preprocess(np.array(vecs_buf, dtype=np.float32))
            yield np.array(ids_buf, dtype=np.int64), X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_optimal_k(
        self,
        data_source: BatchDataSource,
        n: int,
        batch_size: int = 100,
        m_min: int = 10,
    ) -> int:
        optimal_k = self._find_optimal_k(data_source=data_source, n=n, batch_size=batch_size, m_min=m_min)
        if not optimal_k:
            raise RuntimeError("Failed to determine optimal k.")
        self.n_clusters = optimal_k
        return optimal_k

    def fit(
        self,
        data_source: BatchDataSource,
        k: Optional[int] = None,
    ) -> None:
        logger.info("Fitting Clusterer model.")

        if k is not None:
            self.n_clusters = k

        if self.n_clusters is None:
            raise ValueError("Number of clusters (n_clusters) must be set before fitting.")

        self._fit(data_source=data_source, k=self.n_clusters, batch_size=1000)
        logger.debug("Clusterer model fitted.")

    def predict(self, data_source: BatchDataSource) -> np.ndarray:
        logger.debug("Predicting clusters.")
        if not self.kmeans or not self.n_clusters:
            raise RuntimeError("Model must be fit before prediction.")
        return self._predict(data_source=data_source)

    def save_model(self, path: str | Path) -> None:
        logger.debug(f"Saving model to {path}")
        if self.kmeans is None:
            raise RuntimeError("Model must be fit before saving.")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"minibatch_kmeans_model": self.kmeans, "n_clusters": self.n_clusters}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path) -> None:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        logger.info(f"Loading model from {path}")
        payload = joblib.load(path)
        self.kmeans = payload["minibatch_kmeans_model"]
        self.n_clusters = payload["n_clusters"]
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _find_optimal_k(
        self,
        data_source: BatchDataSource,
        n: int,
        batch_size: int = 100,
        m_min: int = 10,
    ) -> Optional[int]:
        logger.debug("Finding optimal k via Elbow method (streaming).")

        k_max = n // m_min
        num_ks = 30
        K_range = np.unique(np.logspace(np.log10(2), np.log10(k_max), num=num_ks).astype(int))

        inertias: list[float] = []
        early_stop_patience = 4
        unchanged_knee_steps = 0
        last_knee: Optional[int] = None

        for k in tqdm(K_range, desc="Finding optimal k"):
            mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)

            warmup: list[np.ndarray] = []
            warmup_size = 0
            warmup_done = False

            for _, X_batch in self._iter_batches(data_source, batch_size):
                if not warmup_done:
                    warmup.append(X_batch)
                    warmup_size += len(X_batch)
                    if warmup_size >= k:
                        mbk.fit(np.concatenate(warmup))
                        warmup_done = True
                        warmup = []
                else:
                    mbk.partial_fit(X_batch)

            if not warmup_done and warmup:
                mbk.fit(np.concatenate(warmup))  # small k, whole dataset fits in warmup

            inertias.append(mbk.inertia_)

            if len(inertias) >= 3:
                kl = KneeLocator(
                    K_range[: len(inertias)],
                    inertias,
                    curve="convex",
                    direction="decreasing",
                )
                current_knee = kl.knee
                unchanged_knee_steps = unchanged_knee_steps + 1 if current_knee == last_knee and current_knee is not None else 0
                last_knee = current_knee

                if unchanged_knee_steps >= early_stop_patience:
                    logger.info(f"Early stopping at knee={current_knee}.")
                    break

        kl = KneeLocator(
            K_range[: len(inertias)],
            inertias,
            curve="convex",
            direction="decreasing",
        )
        logger.info(f"Elbow at k={kl.knee}")
        self._plot_inertia(K_range[: len(inertias)], inertias, kl.knee, n=n)
        return kl.knee

    def _fit(
        self,
        data_source: BatchDataSource,
        k: int,
        batch_size: int = 1000,
    ) -> None:
        logger.debug(f"Fitting MiniBatchKMeans (streaming, batch_size={batch_size})")

        if self.kmeans is None:
            self.kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch_size,
                random_state=42,
            )

        warmup: list[np.ndarray] = []
        warmup_size = 0
        warmup_done = False

        for _, X_batch in self._iter_batches(data_source, batch_size):
            if not warmup_done:
                warmup.append(X_batch)
                warmup_size += len(X_batch)

                if warmup_size >= k:
                    X_warm = np.concatenate(warmup)
                    self.kmeans.fit(X_warm)  # initialises centres
                    warmup_done = True
                    warmup = []
            else:
                self.kmeans.partial_fit(X_batch)

        # Edge case: entire dataset was consumed into warmup but never >= k
        if not warmup_done:
            if warmup:
                X_warm = np.concatenate(warmup)
                if len(X_warm) < k:
                    raise ValueError(f"Dataset has only {len(X_warm)} samples but k={k}. Reduce k or provide more data.")
                self.kmeans.fit(X_warm)
            else:
                raise RuntimeError("Data source was empty.")

        logger.info("MiniBatchKMeans training completed.")

    def _predict(self, data_source: BatchDataSource) -> np.ndarray:
        logger.debug("Predicting clusters (streaming).")
        if not self.kmeans:
            raise RuntimeError("Model must be fit before prediction.")

        all_ids: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for ids_batch, X_batch in self._iter_batches(data_source, batch_size=1000):
            all_ids.append(ids_batch)
            all_labels.append(self.kmeans.predict(X_batch))

        ids = np.concatenate(all_ids)
        labels = np.concatenate(all_labels)
        logger.debug("Cluster prediction completed.")
        return np.column_stack((ids, labels))

    def _plot_inertia(
        self,
        K_range: np.ndarray,
        inertias: List[float],
        elbow: Optional[int],
        n: int,
    ) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(K_range, inertias, marker="o", label="Inertia")
        if elbow is not None:
            plt.vlines(elbow, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", label="Elbow k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.legend()
        plt.savefig(self.plots_dir / f"n_{n}_k_{elbow}.png")
        plt.close()

    def _log_silhouette(self, X: np.ndarray, labels: np.ndarray) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        score = silhouette_score(X, labels)
        logger.info(f"Silhouette Score: {score:.4f}")

        with open(self.plots_dir / "silhouette_scores.log", "a") as f:
            f.write(f"Number of Samples: {X.shape[0]}, Number of Clusters: {self.n_clusters}, Silhouette Score: {score:.4f}\n")
