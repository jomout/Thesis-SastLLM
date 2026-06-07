from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sastllm.configs import get_logger
from sastllm.db import FunctionalityManager
from sastllm.dtos.update_dtos import UpdateFunctionalityDto
from sastllm.utils.observability import log_duration

from .config import ClusteringConfig, ClusteringMode
from .kmeans import ClusterAssignmentBatch, MiniBatchKMeansClusterer
from .sources import QdrantEmbeddingRepository

logger = get_logger(__name__)


class FunctionalityClusteringService:
    """
    Orchestrates the clustering phase without owning low-level model or storage logic.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        config: ClusteringConfig | None = None,
        config_path: str | Path = "configs/clustering.yaml",
        embedding_repository: QdrantEmbeddingRepository | None = None,
        functionality_manager: FunctionalityManager | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.config = config or ClusteringConfig.from_yaml(config_path)
        self.embedding_repository = embedding_repository or QdrantEmbeddingRepository(collection_name=collection_name)
        self.functionality_manager = functionality_manager or FunctionalityManager()
        logger.info("Initialized functionality clustering service", collection_name=collection_name)

    def run(self, mode: ClusteringMode) -> None:
        dispatch = {
            "search": self.search,
            "train": self.train,
            "test": self.test,
        }
        try:
            dispatch[mode]()
        except KeyError as e:
            raise ValueError(f"Unknown clustering mode '{mode}'. Choose from: {list(dispatch)}.") from e

    def search(self) -> None:
        cfg = self.config.search
        logger.info(
            "Searching for clustering k",
            collection_name=self.collection_name,
            grid_search=cfg.grid_search,
            batch_size=cfg.batch_size,
            min_samples_per_cluster=cfg.min_samples_per_cluster,
        )

        for n in cfg.grid_search:
            source = self.embedding_repository.first_n(n)
            clusterer = MiniBatchKMeansClusterer(random_state=cfg.random_state, plots_dir=cfg.save_plots_dir)
            with log_duration(logger, "cluster_search_sample", n=n):
                optimal_k = clusterer.find_optimal_k(
                    source,
                    n=n,
                    batch_size=cfg.batch_size,
                    min_samples_per_cluster=cfg.min_samples_per_cluster,
                    num_candidates=cfg.num_k_candidates,
                    early_stop_patience=cfg.early_stop_patience,
                )
                clusterer.fit(source, k=optimal_k, batch_size=cfg.batch_size)
                clusterer.save_model(cfg.save_model_dir / f"clusterer_n_{n}_k_{optimal_k}.joblib")

    def train(self) -> None:
        cfg = self.config.train
        source = self.embedding_repository.for_split("train")
        train_count = self.embedding_repository.count_for_split("train")
        logger.info("Training clustering model", collection_name=self.collection_name, split="train", embeddings=train_count, k=cfg.k)

        clusterer = MiniBatchKMeansClusterer(random_state=cfg.random_state)
        with log_duration(logger, "cluster_train_fit", split="train", embeddings=train_count, k=cfg.k):
            clusterer.fit(source, k=cfg.k, batch_size=cfg.batch_size)
            clusterer.save_model(cfg.save_model_dir / f"clusterer_k_{cfg.k}.joblib")

        with log_duration(logger, "cluster_train_assignments", split="train"):
            self._store_assignments(clusterer.predict_batches(source, batch_size=cfg.batch_size))

    def test(self) -> None:
        cfg = self.config.test
        source = self.embedding_repository.for_split("test")
        test_count = self.embedding_repository.count_for_split("test")
        logger.info(
            "Predicting clustering assignments",
            collection_name=self.collection_name,
            split="test",
            embeddings=test_count,
            model_file=str(cfg.load_model_file),
        )

        clusterer = MiniBatchKMeansClusterer()
        clusterer.load_model(cfg.load_model_file)

        with log_duration(logger, "cluster_test_assignments", split="test", embeddings=test_count):
            self._store_assignments(clusterer.predict_batches(source, batch_size=cfg.batch_size))

    def _store_assignments(self, assignment_batches: Iterable[ClusterAssignmentBatch]) -> None:
        total = 0
        for batch in assignment_batches:
            updates = [UpdateFunctionalityDto(functionality_id=int(functionality_id), cluster_id=int(cluster_id)) for functionality_id, cluster_id in zip(batch.functionality_ids, batch.cluster_ids)]
            self.functionality_manager.update_bulk_functionalities(updates)
            total += len(updates)
            logger.debug("Stored cluster assignment batch", batch_size=len(updates), total=total)

        logger.info("Stored %s functionality cluster assignments.", total)
