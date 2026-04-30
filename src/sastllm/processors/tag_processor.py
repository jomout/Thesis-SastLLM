from typing import Any, Literal

import numpy as np

from sastllm.cluster import Clusterer
from sastllm.configs import get_logger
from sastllm.db import EmbeddingsManager, FunctionalityManager
from sastllm.db.batch_datasource import BatchDataSource
from sastllm.dtos.update_dtos import UpdateFunctionalityDto
from scripts.utils import load_yaml

logger = get_logger(__name__)


class TagProcessor:
    """
    Orchestrates the clustering of functionality tags and assigns cluster IDs
    to functionalities in the database.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        config_path: str = "configs/clustering.yaml",
    ) -> None:
        logger.debug("Initializing TagProcessor.")

        self.functionality_db = FunctionalityManager()
        self.embeddings_manager = EmbeddingsManager()
        self.collection_name = collection_name
        self.cfg = self._load_config(config_path)

        logger.debug("TagProcessor initialized.")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        try:
            cfg = load_yaml(config_path).get("clustering", {})
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}.")

        if not cfg:
            raise ValueError(f"No 'clustering' section found in {config_path}.")

        return cfg

    def _cfg(self, *keys: str, default=None) -> Any:
        """Safely traverses nested config keys."""
        node = self.cfg
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _embeddings_by_split(self, split: str) -> BatchDataSource:
        return BatchDataSource(
            lambda: self.embeddings_manager.get_embeddings_by_payload_field(
                collection_name=self.collection_name,
                field="split",
                values=[split],
            )
        )

    def _count_by_split(self, split: str) -> int:
        return self.embeddings_manager.count_embeddings_by_payload_field(
            collection_name=self.collection_name,
            field="split",
            values=[split],
        )

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def _search(self) -> None:
        logger.info("Search Mode: Searching for optimal k.")

        grid_ns = self._cfg("search", "grid_search", default=[])
        if not isinstance(grid_ns, list) or not all(isinstance(x, int) for x in grid_ns):
            raise ValueError("'search.grid_search' must be a list of ints.")

        plot_dir = self._cfg("search", "save_plots_dir", default="plots")
        save_dir = self._cfg("search", "save_model_dir", default="models/clustering/searching_models")

        for n in grid_ns:
            logger.info(f"Search Mode: n={n} functionalities.")
            clusterer = Clusterer(plots_dir=plot_dir)

            source = BatchDataSource(lambda n=n: self.embeddings_manager.get_n_embeddings(collection_name=self.collection_name, n=n))

            optimal_k = clusterer.find_optimal_k(source, n=n, batch_size=1000, m_min=20)
            logger.info(f"Optimal k for n={n}: {optimal_k}")

            clusterer.fit(source, k=optimal_k)
            clusterer.save_model(f"{save_dir}/clusterer_n_{n}_k_{optimal_k}.joblib")

    def _train(self) -> None:
        logger.info("Training Mode: Fitting clustering model.")

        k = self._cfg("train", "k")
        if k is None:
            raise ValueError("'train.k' must be specified in config.")

        save_dir = self._cfg("train", "save_model_dir", default="models/clustering/trained_models")

        source = self._embeddings_by_split("train")
        clusterer = Clusterer()

        clusterer.fit(source, k=k)
        result = clusterer.predict(source)

        clusterer.save_model(f"{save_dir}/clusterer_k_{k}.joblib")
        self._store_labels(result)

    def _test(self) -> None:
        logger.info("Testing Mode: Predicting with existing model.")

        model_file = self._cfg("test", "load_model_file")
        if model_file is None:
            raise ValueError("'test.load_model_file' must be specified in config.")

        clusterer = Clusterer()
        clusterer.load_model(model_file)

        source = self._embeddings_by_split("test")
        result = clusterer.predict(source)
        self._store_labels(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, mode: Literal["search", "train", "test"]) -> None:
        """
        Runs the tag processing pipeline.

        Args:
            mode: 'search' finds optimal k, 'train' fits a model,
                  'test' predicts with a saved model.
        """
        logger.info(f"Starting tag processing in '{mode}' mode.")

        dispatch = {
            "search": self._search,
            "train": self._train,
            "test": self._test,
        }

        try:
            dispatch[mode]()
        except KeyError:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(dispatch)}")
        except Exception as e:
            logger.error(f"Tag processing failed: {e}")
            raise RuntimeError(f"Tag processing failed: {e}") from e

        logger.info("Tag processing completed successfully.")

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_labels(self, result: np.ndarray) -> None:
        logger.debug("Storing cluster labels in the database.")
        try:
            dtos = [UpdateFunctionalityDto(functionality_id=int(func_id), cluster_id=int(cluster_id)) for func_id, cluster_id in result]
            self.functionality_db.update_bulk_functionalities(dtos)
        except Exception as e:
            logger.error(f"Failed to store cluster labels: {e}")
            raise RuntimeError(f"Failed to store cluster labels: {e}") from e

        logger.debug(f"Stored {len(result)} cluster labels.")
