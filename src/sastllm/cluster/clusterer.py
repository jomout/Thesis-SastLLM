from pathlib import Path
from typing import List, Optional

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
    """
    Handles embedding generation, preprocessing, and clustering of functionality tags.

    This class supports fetching tagged functionalities from a database, encoding
    them into embeddings, reducing dimensionality, and performing clustering
    using HDBSCAN to estimate the number of clusters, followed by KMeans for final
    label assignment. The results are then stored back in the database.
    """

    def __init__(
        self,
        *,
        plots_dir: str = "cluster_plots",
    ):
        """
        Initializes the FunctionalityClusterer.

        Args:
            plots_dir (str): Directory to save clustering plots.
        """
        logger.debug("Initializing Clusterer.")

        self.kmeans = None
        self.n_clusters = None
        self.plots_dir = Path(plots_dir)

        logger.debug("Clusterer initialized.")

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Applies L2 normalization to the input feature matrix.

        Args:
            X (np.ndarray): The input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed and normalized feature matrix.
        """
        return normalize(X, norm="l2", axis=1)

    def find_optimal_k(
        self, data_source: BatchDataSource, n: int, batch_size: int = 100, m_min: int = 10
    ) -> int:
        """
        Wrapper that embeds, preprocesses, and finds the optimal number of clusters.
        """
        optimal_k = self._find_optimal_k(
            data_source=data_source, n=n, batch_size=batch_size, m_min=m_min
        )
        if not optimal_k:
            logger.error("Failed to determine optimal k.")
            raise RuntimeError("Failed to determine optimal k.")
        self.n_clusters = optimal_k
        return optimal_k

    def fit(self, data_source: BatchDataSource, n: int, k: Optional[int] = None) -> None:
        """
        Fits the clustering model to the provided functionality tags.

        Args:
            X (np.ndarray): The input feature matrix.
            k (int): The number of clusters to form.

        Returns:
            List[int]: A list of cluster labels assigned to each functionality tag.
        """
        logger.info("Fitting Clusterer model.")

        if k is not None:
            self.n_clusters = k

        if self.n_clusters is None:
            if k is None:
                logger.error("Number of clusters (n_clusters) must be set before fitting.")
                raise ValueError("Number of clusters (n_clusters) must be set before fitting.")
            else:
                self.n_clusters = k
                logger.info(f"Number of clusters to form: {self.n_clusters}")

        # Fitting KMeans
        self._fit(data_source=data_source, n=n, k=self.n_clusters, batch_size=1000)

        logger.debug("Clusterer model fitted.")

    def _consume_embeddings(
        self,
        data_source: BatchDataSource,
        n: int,
    ):
        """
        Loads all ids and items from the data_source generator into numpy arrays.
        """
        gen = iter(data_source.iter())
        first_item = next(gen)

        if not isinstance(first_item, tuple):
            raise RuntimeError("Data source items are not tuples of (id, vector).")

        first_vec = first_item[1]
        dim = first_vec.shape[0]

        logger.info(f"Consuming {n} items with dim {dim} from data source.")

        X = np.empty((n, dim), dtype=np.float32)
        ids = np.empty((n,), dtype=np.int32)

        gen = data_source.iter()  # fresh generator
        i = 0

        for item in gen:
            if i >= n:
                logger.warning("Data source overflowed the expected size.")
                break  # safety valve

            ident, vec = item
            ids[i] = ident  # type: ignore
            X[i] = vec

            i += 1

        if i < n:
            raise RuntimeError(f"Data source underflowed: expected {n}, got only {i}")

        return ids, self._preprocess(X)

    def _find_optimal_k(
        self, data_source: BatchDataSource, n: int, batch_size: int = 100, m_min: int = 10
    ) -> Optional[int]:
        logger.debug("Finding optimal number of clusters (k) using the Elbow method.")

        # Load all data
        ids, X = self._consume_embeddings(data_source, n=n)
        del ids

        logger.info(f"Data loaded for optimal k search: {X.shape[0]} samples.")
        n = X.shape[0]

        # Determine search bounds
        k_max = n // m_min
        logger.debug(f"Max k based on m_min={m_min}: {k_max}")

        num_ks = 30
        K_range = np.unique(np.logspace(np.log10(2), np.log10(k_max), num=num_ks).astype(int))

        inertias = []
        early_stop_patience = 4
        unchanged_knee_steps = 0
        last_knee = None

        for k in tqdm(K_range, desc="Finding optimal k"):
            mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42).fit(X)
            inertias.append(mbk.inertia_)

            # Recompute knee every iteration
            if len(inertias) >= 3:  # needs minimum points
                kl = KneeLocator(
                    K_range[: len(inertias)],
                    inertias,
                    curve="convex",
                    direction="decreasing",
                )
                current_knee = kl.knee

                # Early stopping condition
                if current_knee == last_knee and current_knee is not None:
                    unchanged_knee_steps += 1
                else:
                    unchanged_knee_steps = 0

                last_knee = current_knee

                if unchanged_knee_steps >= early_stop_patience:
                    logger.info(
                        f"Early stopping: knee unchanged for {early_stop_patience} steps "
                        f"(knee={current_knee})."
                    )
                    break

        del X

        # Final knee detection with full collected data
        kl = KneeLocator(
            K_range[: len(inertias)],
            inertias,
            curve="convex",
            direction="decreasing",
        )
        logger.info(f"Elbow at k = {kl.knee}")

        # Plot partial search results
        self._plot_inertia(K_range[: len(inertias)], inertias, kl.knee, n=n)

        return kl.knee

    def _plot_inertia(
        self, K_range: np.ndarray, inertias: List[float], elbow: Optional[int], n: int
    ) -> None:
        """
        Plots the inertia values against the number of clusters and marks the elbow point.

        Args:
            K_range (np.ndarray): The range of k values tested.
            inertias (List[float]): The corresponding inertia values for each k.
            elbow (Optional[int]): The identified elbow point (optimal k).
            n (int): The number of samples in the dataset.
            score (float): The silhouette score for the clustering.
        """
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = self.plots_dir / f"n_{n}_k_{elbow}.png"
        plt.figure()
        plt.plot(K_range, inertias, marker="o", label="Inertia")
        if elbow is not None:
            plt.vlines(
                elbow,
                plt.ylim()[0],
                plt.ylim()[1],
                linestyles="dashed",
                label="Elbow k",
            )
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.legend()
        plt.savefig(fig_dir)
        plt.close()

    def _log_silhouette(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Computes and logs the silhouette score for the clustering.

        Args:
            X (np.ndarray): The input feature matrix.
            labels (np.ndarray): The cluster labels assigned to each sample.
        """
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        score = silhouette_score(X, labels)
        logger.info(f"Silhouette Score: {score:.4f}")

        silhouette_dir = self.plots_dir / "silhouette_scores.log"
        with open(silhouette_dir, "a") as f:
            f.write(
                f"Number of Samples: {X.shape[0]}, Number of Clusters: {self.n_clusters}, \
                    Silhouette Score: {score:.4f}\n"
            )

    def _fit(
        self,
        data_source: BatchDataSource,
        n: int,
        k: int | None = None,
        batch_size: int = 1000,
    ):
        """
        Performs MiniBatchKMeans clustering using partial_fit with streamed batches.
        X is never fully loaded in RAM.
        """

        logger.debug(f"Clustering with MiniBatchKMeans (streaming, batch_size={batch_size})")

        # Create model if needed
        if not self.kmeans:
            if k is None:
                raise ValueError("k must be provided when initializing kmeans.")
            self.kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch_size,
                random_state=42,
            )
        else:
            logger.info(
                f"Using existing MiniBatchKMeans model for fitting with k={self.n_clusters}"
            )

        _, X = self._consume_embeddings(data_source, n)
        logger.info(f"Fitting MiniBatchKMeans with k={k}")
        self.kmeans.fit(X)
        logger.info("MiniBatchKMeans training completed.")

    def _predict(self, data_source: BatchDataSource, n: int) -> np.ndarray:
        """
        Predicts cluster labels for new functionality tags using the trained model.
        Args:
            data_source (BatchDataSource): The data source providing functionality tag embeddings.
        Returns:
            np.ndarray: An array of shape (n_samples, 2) with columns [id, label].
        """

        logger.debug("Predicting clusters (streaming).")

        if not self.kmeans:
            raise RuntimeError("Model must be fit before prediction.")

        # Consume datasource at once
        ids, X = self._consume_embeddings(data_source, n=n)
        labels = self.kmeans.predict(X)

        logger.debug("Cluster prediction completed.")

        return np.column_stack((ids, labels))

    def predict(self, data_source: BatchDataSource, n: int) -> np.ndarray:
        """
        Predicts cluster labels for new functionality tags using the trained model.
        Args:
            data_source (BatchDataSource): The data source providing functionality tag embeddings.
        Returns:
            np.ndarray: An array of shape (n_samples, 2) with columns [id, label].
        """
        logger.debug("Predicting clusters.")

        if not self.kmeans or not self.n_clusters:
            logger.error("Model must be fit before prediction.")
            raise RuntimeError("Model must be fit before prediction.")

        # Predict cluster labels
        result = self._predict(data_source=data_source, n=n)

        logger.debug("Cluster prediction completed.")

        return result

    def save_model(self, path: str | Path) -> None:
        """
        Saves the trained PCA and KMeans models to a file using joblib.

        Args:
            path (str | Path): Path to save the model file.
        """
        logger.debug(f"Saving model to {path}")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.kmeans is None:
            logger.error("Model must be fit before saving.")
            raise RuntimeError("Model must be fit before saving.")

        payload = {
            "minibatch_kmeans_model": self.kmeans,
            "n_clusters": self.n_clusters,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path) -> None:
        """
        Loads a PCA + KMeans model bundle from file.

        Args:
            path (str | Path): Path to the saved model file.

        Returns:
            Dict: A dictionary containing 'pca' and 'kmeans' keys with their respective
                model instances.
        """
        model_path = Path(path)
        if not model_path.exists():
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")

        logger.info(f"Loading model from {path}")
        payload = joblib.load(path)
        self.kmeans = payload["minibatch_kmeans_model"]
        self.n_clusters = payload["n_clusters"]

        logger.info("Model loaded successfully.")
