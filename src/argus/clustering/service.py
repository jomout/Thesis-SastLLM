from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from argus.configs import get_logger
from argus.db import FunctionalityManager
from argus.dtos.update_dtos import UpdateFunctionalityDto
from argus.utils.observability import log_duration

from .artifacts import search_artifacts, training_artifacts
from .config import ClusteringConfig, ClusteringMode
from .evaluation import ClusterQualityEvaluator
from .kmeans import ClusterAssignmentBatch, MiniBatchKMeansClusterer
from .sampling import reservoir_sample
from .selection import ClusterCountSelector, save_quality_report, save_selection_report
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
        evaluation_cfg = self.config.evaluation
        logger.info(
            "Searching for clustering k",
            collection_name=self.collection_name,
            grid_search=cfg.grid_search,
            batch_size=cfg.batch_size,
            min_samples_per_cluster=cfg.min_samples_per_cluster,
        )

        for n in cfg.grid_search:
            source = self.embedding_repository.first_n(n)
            evaluator = self._quality_evaluator()
            with log_duration(logger, "cluster_search_sample", n=n):
                evaluation_sample = reservoir_sample(
                    source,
                    sample_size=min(evaluation_cfg.sample_size, n),
                    random_state=evaluation_cfg.random_state,
                )
                selector = ClusterCountSelector(
                    evaluator=evaluator,
                    random_state=cfg.random_state,
                    elbow_window_factor=evaluation_cfg.elbow_window_factor,
                    max_silhouette_singleton_fraction=evaluation_cfg.max_silhouette_singleton_fraction,
                )
                selection = selector.search(
                    source,
                    evaluation_sample,
                    n=n,
                    batch_size=cfg.batch_size,
                    min_samples_per_cluster=cfg.min_samples_per_cluster,
                    num_candidates=cfg.num_k_candidates,
                )
                artifacts = search_artifacts(
                    cfg.save_model_dir,
                    n=n,
                    k=selection.selected_k,
                )
                report_paths = save_selection_report(
                    selection,
                    output_dir=artifacts.directory,
                    name=f"{artifacts.stem}_selection",
                )
                logger.info(
                    "Selected clustering K",
                    n=n,
                    selected_k=selection.selected_k,
                    reason=selection.selection_reason,
                    elbow_k=selection.elbow_k,
                    silhouette_best_k=selection.silhouette_best_k,
                    calinski_harabasz_best_k=selection.calinski_harabasz_best_k,
                    reports=[str(path) for path in report_paths],
                )

                clusterer = MiniBatchKMeansClusterer(random_state=cfg.random_state)
                initialization_vectors = evaluation_sample.vectors if len(evaluation_sample) >= selection.selected_k else None
                clusterer.fit(
                    source,
                    k=selection.selected_k,
                    batch_size=cfg.batch_size,
                    initialization_vectors=initialization_vectors,
                )
                if clusterer.kmeans is None:
                    raise RuntimeError("Selected clusterer did not produce a fitted model.")
                final_quality = evaluator.evaluate_streaming(
                    clusterer.kmeans,
                    source,
                    batch_size=cfg.batch_size,
                    silhouette_sample=evaluation_sample,
                    min_cluster_size=cfg.min_samples_per_cluster,
                )
                clusterer.save_model(artifacts.model)
                quality_path = save_quality_report(final_quality, artifacts.quality_report)
                logger.info(
                    "Saved clustering search artifacts",
                    artifact_dir=str(artifacts.directory),
                    model_file=str(artifacts.model),
                    quality_report=str(quality_path),
                )

    def train(self) -> None:
        cfg = self.config.train
        evaluation_cfg = self.config.evaluation
        source = self.embedding_repository.for_split("train")
        train_count = self.embedding_repository.count_for_split("train")
        logger.info("Training clustering model", collection_name=self.collection_name, split="train", embeddings=train_count, k=cfg.k)
        if train_count < cfg.k:
            raise ValueError(f"Cannot fit k={cfg.k}; only {train_count} training embeddings are available.")

        with log_duration(logger, "cluster_train_sample", split="train", embeddings=train_count, k=cfg.k):
            evaluation_sample = reservoir_sample(
                source,
                sample_size=min(max(evaluation_cfg.sample_size, cfg.k), train_count),
                random_state=evaluation_cfg.random_state,
            )

        clusterer = MiniBatchKMeansClusterer(random_state=cfg.random_state)
        artifacts = training_artifacts(cfg.save_model_dir, k=cfg.k)
        with log_duration(logger, "cluster_train_fit", split="train", embeddings=train_count, k=cfg.k):
            clusterer.fit(
                source,
                k=cfg.k,
                batch_size=cfg.batch_size,
                initialization_vectors=evaluation_sample.vectors,
            )
            clusterer.save_model(artifacts.model)

        if clusterer.kmeans is None:
            raise RuntimeError("Trained clusterer did not produce a fitted model.")
        with log_duration(logger, "cluster_train_quality", split="train", embeddings=train_count, k=cfg.k):
            quality = self._quality_evaluator().evaluate_streaming(
                clusterer.kmeans,
                source,
                batch_size=cfg.batch_size,
                silhouette_sample=evaluation_sample,
                min_cluster_size=self.config.search.min_samples_per_cluster,
            )
            quality_path = save_quality_report(quality, artifacts.quality_report)
            logger.info(
                "Validated trained clustering model",
                artifact_dir=str(artifacts.directory),
                model_file=str(artifacts.model),
                quality_report=str(quality_path),
                normalized_inertia=quality.normalized_inertia,
                silhouette=quality.silhouette_score,
                calinski_harabasz=quality.calinski_harabasz_score,
                empty_clusters=quality.cluster_sizes.empty_clusters,
                below_minimum_clusters=quality.cluster_sizes.below_minimum_clusters,
            )

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

    def _quality_evaluator(self) -> ClusterQualityEvaluator:
        cfg = self.config.evaluation
        return ClusterQualityEvaluator(
            silhouette_sample_size=cfg.silhouette_sample_size,
            silhouette_samples_per_cluster=cfg.silhouette_samples_per_cluster,
            silhouette_metric=cfg.silhouette_metric,
            random_state=cfg.random_state,
        )
