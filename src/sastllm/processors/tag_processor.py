from typing import Literal

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
    Orchestrates the clustering of functionality tags and assigns cluster IDs to functionalities
    in the database.
    This class fetches functionality tags from the database, generates embeddings,
    reduces dimensionality if needed, performs clustering, and updates the database with cluster
    assignments.
    """

    def __init__(
        self,
        *,
        batch_size: int = 50,
        collection_name: str,
        config_path: str = "configs/clustering.yaml",
    ) -> None:
        """
        Initialize the TagProcessor with its dependencies and configuration.

        """
        logger.debug("Initializing TagProcessor.")

        self.batch_size = batch_size
        self.functionality_db = FunctionalityManager()
        self.embeddings_manager = EmbeddingsManager()

        self.collection_name = collection_name

        try:
            cfg = load_yaml(config_path).get("clustering", {})
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}.")
            cfg = {}

        if not cfg:
            logger.error(f"No 'clustering' section found in {config_path}.")
            raise ValueError("No clustering configuration found.")

        self.cfg = cfg

        logger.debug("TagProcessor initialized.")

    def _search(self) -> None:
        """
        Perform grid search to find the optimal number of clusters (k) for functionality tags.
        """
        logger.info("Search Mode: Searching for k.")

        # Load from YAML if available
        grid_search_functionalities = self.cfg.get("search", {}).get("grid_search", [])

        if not isinstance(grid_search_functionalities, list) or not all(
            isinstance(x, int) for x in grid_search_functionalities
        ):
            logger.error("Invalid 'search.grid_search' in YAML.")
            raise ValueError("Invalid 'search.grid_search' in YAML.")

        plot_dir = self.cfg.get("search", {}).get("save_plots_dir", "plots")

        # Grid search for optimal k
        for n in grid_search_functionalities:
            logger.info(f"Search Mode: Processing n={n} functionalities.")
            clusterer = Clusterer(plots_dir=plot_dir)

            # Fetch n embeddings from db
            embedding_records = BatchDataSource(
                lambda: self.embeddings_manager.get_n_embeddings(
                    collection_name=self.collection_name, n=n
                )
            )
            # Find optimal k
            logger.info(f"Finding optimal k for n={n}.")
            optimal_k = clusterer.find_optimal_k(embedding_records, n=n, batch_size=1000, m_min=20)
            clusterer.fit(embedding_records, n=n, k=optimal_k)

            # Save model
            logger.info(f"Optimal k for n={n} is {optimal_k}.")
            save_dir = self.cfg.get("search", {}).get(
                "save_model_dir", "models/clustering/searching_models"
            )
            clusterer.save_model(f"{save_dir}/clusterer_n:{n}_k:{optimal_k}.joblib")

    def _train(self) -> None:
        """
        Perform training of clustering models on functionality tags and assign cluster IDs.
        """
        logger.info("Training Mode: Fitting clustering models.")

        k = self.cfg.get("train", {}).get("k", None)

        if k is None:
            logger.error("In 'train' mode, 'k' must be specified.")
            raise ValueError("In 'train' mode, 'k' must be specified.")

        clusterer = Clusterer()

        logger.info(f"Fetching training embeddings for k={k}.")
        embeddings = BatchDataSource(
            lambda: self.embeddings_manager.get_embeddings_by_payload_field(
                collection_name=self.collection_name,
                field="split",
                values=["train"],
            )
        )

        # Get number of training embeddings
        n_embeddings = self.embeddings_manager.count_embeddings_by_payload_field(
            collection_name=self.collection_name,
            field="split",
            values=["train"],
        )

        # Train clusterer
        logger.info("Training clustering model.")
        clusterer.fit(embeddings, n=n_embeddings, k=k)

        # Predict clusters
        logger.info("Predicting clusters for training data.")
        result = clusterer.predict(embeddings, n=n_embeddings)

        # Save model
        save_dir = self.cfg.get("train", {}).get(
            "save_model_dir", "models/clustering/trained_models"
        )
        clusterer.save_model(f"{save_dir}/clusterer_k_{k}.joblib")

        self._store_labels(result)

    def _test(self) -> None:
        logger.info("Testing Mode: Using existing clustering models.")
        k = self.cfg.get("test", {}).get("k", None)

        if k is None:
            logger.error("In 'test' mode, 'k' must be specified.")
            raise ValueError("In 'test' mode, 'k' must be specified.")

        clusterer = Clusterer()

        model_file = self.cfg.get("test", {}).get("load_model_file", None)
        if model_file is None:
            logger.error("In 'test' mode, 'load_file' must be specified.")
            raise ValueError("In 'test' mode, 'load_file' must be specified.")

        # Get model with k clusters
        clusterer.load_model(model_file)

        logger.info(f"Fetching testing embeddings for k={k}.")
        embeddings = BatchDataSource(
            lambda: self.embeddings_manager.get_embeddings_by_payload_field(
                collection_name=self.collection_name,
                field="split",
                values=["test"],
            )
        )

        # Get number of testing embeddings
        n_embeddings = self.embeddings_manager.count_embeddings_by_payload_field(
            collection_name=self.collection_name,
            field="split",
            values=["test"],
        )

        # Predict clusters
        logger.info("Predicting clusters for testing data.")
        result = clusterer.predict(embeddings, n=n_embeddings)

        self._store_labels(result)

    def run(self, mode: Literal["search", "train", "test"], k: int | None = None) -> None:
        """
        Runs the tag processing pipeline to cluster functionality tags and assign cluster IDs.

        Steps:
            1. Fetch all functionality tags from the database.
            2. Generate embeddings for the tags.
            3. Cluster the embeddings using MiniBatchKMeans.
            4. Assign cluster IDs back to the functionalities in the database.

        Args:
            mode (Literal['search', 'train', 'test']): Mode of operation. In 'train' mode, fits new
            clustering models. In 'test' mode, uses existing models.
            verbose (bool, optional): If True, prints detailed information about tags and their
            assigned clusters. Defaults to False.
        """
        logger.info(f"Starting tag processing in '{mode}' mode.")
        try:
            if mode == "search":
                self._search()

            elif mode == "train":
                self._train()

            else:
                self._test()

        except Exception as e:
            logger.error(f"Tag processing failed: {e}")
            raise RuntimeError(f"Tag processing failed: {e}") from e

        logger.info("Tag processing completed successfully.")

    def _store_labels(self, result: np.ndarray) -> None:
        """
        Stores the cluster labels in the database.
        Args:
            result (np.ndarray): Array of tuples (functionality_id, cluster_id).
        """
        logger.debug("Storing cluster labels in the database.")

        try:
            functionalities = []

            for func_id, cluster_id in result:
                functionalities.append(
                    UpdateFunctionalityDto(functionality_id=func_id, cluster_id=cluster_id)
                )
            self.functionality_db.update_bulk_functionalities(functionalities)
        except Exception as e:
            logger.error(f"Failed to store cluster labels: {e}")
            raise RuntimeError(f"Failed to store cluster labels: {e}") from e

        logger.debug(f"Stored cluster labels for {len(result)} functionalities.")
