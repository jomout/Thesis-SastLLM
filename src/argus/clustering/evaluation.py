from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

from argus.configs import get_logger

from .sampling import EmbeddingSample, ReservoirSampler
from .sources import EmbeddingSource, normalize_batches

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusterSizeStatistics:
    total_clusters: int
    represented_clusters: int
    empty_clusters: int
    singleton_clusters: int
    below_minimum_clusters: int
    minimum: int
    percentile_05: float
    median: float
    mean: float
    percentile_95: float
    maximum: int


@dataclass(frozen=True)
class PerClusterQuality:
    cluster_id: int
    size: int
    inertia: float
    mean_squared_distance: float
    rms_distance: float


@dataclass(frozen=True)
class ClusterQualityReport:
    scope: str
    k: int
    evaluated_samples: int
    population_size: int
    inertia: float
    normalized_inertia: float
    cohesion_rms_distance: float
    between_cluster_dispersion: float
    normalized_between_cluster_dispersion: float
    separation_rms_distance: float
    calinski_harabasz_score: float | None
    silhouette_score: float | None
    silhouette_metric: str
    silhouette_sample_size: int
    silhouette_represented_clusters: int
    silhouette_singleton_clusters: int
    silhouette_cluster_coverage: float
    cluster_sizes: ClusterSizeStatistics
    per_cluster: tuple[PerClusterQuality, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class ClusterQualityEvaluator:
    def __init__(
        self,
        *,
        silhouette_sample_size: int = 5000,
        silhouette_samples_per_cluster: int = 5,
        silhouette_metric: str = "euclidean",
        random_state: int = 42,
    ) -> None:
        if silhouette_sample_size <= 1:
            raise ValueError("silhouette_sample_size must be > 1.")
        if silhouette_samples_per_cluster < 2:
            raise ValueError("silhouette_samples_per_cluster must be >= 2.")
        self.silhouette_sample_size = silhouette_sample_size
        self.silhouette_samples_per_cluster = silhouette_samples_per_cluster
        self.silhouette_metric = silhouette_metric
        self.random_state = random_state

    def evaluate_sample(
        self,
        model: MiniBatchKMeans,
        sample: EmbeddingSample,
        *,
        min_cluster_size: int = 2,
    ) -> ClusterQualityReport:
        logger.debug(
            "Evaluating clustering quality on sample",
            k=int(model.n_clusters),
            sample_size=len(sample),
            population_size=sample.population_size,
        )
        labels = model.predict(sample.vectors).astype(np.int64)
        squared_distances = _assigned_squared_distances(sample.vectors, labels, model.cluster_centers_)
        counts = np.bincount(labels, minlength=model.n_clusters).astype(np.int64)
        per_cluster_inertia = np.bincount(labels, weights=squared_distances, minlength=model.n_clusters)
        report = self._build_report(
            scope="sample",
            model=model,
            vectors=sample.vectors,
            labels=labels,
            counts=counts,
            per_cluster_inertia=per_cluster_inertia,
            vector_sum=sample.vectors.sum(axis=0, dtype=np.float64),
            population_size=sample.population_size,
            min_cluster_size=min_cluster_size,
        )
        logger.debug(
            "Completed sample clustering evaluation",
            k=report.k,
            cohesion=report.cohesion_rms_distance,
            separation=report.separation_rms_distance,
            silhouette=report.silhouette_score,
            represented_clusters=report.cluster_sizes.represented_clusters,
        )
        return report

    def evaluate_streaming(
        self,
        model: MiniBatchKMeans,
        source: EmbeddingSource,
        *,
        batch_size: int,
        silhouette_sample: EmbeddingSample | None = None,
        stream_sample_size: int = 10000,
        min_cluster_size: int,
    ) -> ClusterQualityReport:
        logger.info(
            "Starting full streaming clustering evaluation",
            k=int(model.n_clusters),
            batch_size=batch_size,
            supplied_silhouette_sample=silhouette_sample is not None,
        )
        counts = np.zeros(model.n_clusters, dtype=np.int64)
        per_cluster_inertia = np.zeros(model.n_clusters, dtype=np.float64)
        vector_sum: np.ndarray | None = None
        evaluated_samples = 0
        sampler = None
        if silhouette_sample is None:
            sampler = ReservoirSampler(sample_size=stream_sample_size, random_state=self.random_state)

        for functionality_ids, vectors in normalize_batches(source, batch_size):
            labels = model.predict(vectors).astype(np.int64)
            squared_distances = _assigned_squared_distances(vectors, labels, model.cluster_centers_)
            counts += np.bincount(labels, minlength=model.n_clusters).astype(np.int64)
            per_cluster_inertia += np.bincount(labels, weights=squared_distances, minlength=model.n_clusters)
            batch_sum = vectors.sum(axis=0, dtype=np.float64)
            vector_sum = batch_sum if vector_sum is None else vector_sum + batch_sum
            evaluated_samples += len(vectors)
            if sampler is not None:
                sampler.add_batch(functionality_ids, vectors)

        if evaluated_samples == 0 or vector_sum is None:
            raise RuntimeError("Cannot evaluate an empty embedding source.")

        if silhouette_sample is None:
            if sampler is None:
                raise RuntimeError("Streaming silhouette sampler was not initialized.")
            silhouette_sample = sampler.result()
        silhouette_labels = model.predict(silhouette_sample.vectors).astype(np.int64)
        report = self._build_report(
            scope="full",
            model=model,
            vectors=silhouette_sample.vectors,
            labels=silhouette_labels,
            counts=counts,
            per_cluster_inertia=per_cluster_inertia,
            vector_sum=vector_sum,
            population_size=evaluated_samples,
            min_cluster_size=min_cluster_size,
        )
        logger.info(
            "Completed full streaming clustering evaluation",
            k=report.k,
            evaluated_samples=report.evaluated_samples,
            cohesion=report.cohesion_rms_distance,
            separation=report.separation_rms_distance,
            silhouette=report.silhouette_score,
            empty_clusters=report.cluster_sizes.empty_clusters,
            singleton_clusters=report.cluster_sizes.singleton_clusters,
        )
        return report

    def _build_report(
        self,
        *,
        scope: str,
        model: MiniBatchKMeans,
        vectors: np.ndarray,
        labels: np.ndarray,
        counts: np.ndarray,
        per_cluster_inertia: np.ndarray,
        vector_sum: np.ndarray,
        population_size: int,
        min_cluster_size: int,
    ) -> ClusterQualityReport:
        evaluated_samples = int(counts.sum())
        inertia = float(per_cluster_inertia.sum())
        normalized_inertia = inertia / evaluated_samples
        global_centroid = vector_sum / evaluated_samples
        centroid_offsets = model.cluster_centers_.astype(np.float64) - global_centroid
        centroid_squared_distances = np.einsum("ij,ij->i", centroid_offsets, centroid_offsets)
        between_dispersion = float(np.dot(counts.astype(np.float64), centroid_squared_distances))
        normalized_between = between_dispersion / evaluated_samples
        represented_clusters = int(np.count_nonzero(counts))
        calinski_harabasz = _calinski_harabasz(
            within_dispersion=inertia,
            between_dispersion=between_dispersion,
            sample_count=evaluated_samples,
            represented_clusters=represented_clusters,
        )
        silhouette = self._silhouette(vectors, labels, model.n_clusters)
        size_stats = _cluster_size_statistics(counts, min_cluster_size=min_cluster_size)
        per_cluster = _per_cluster_quality(counts, per_cluster_inertia)

        return ClusterQualityReport(
            scope=scope,
            k=int(model.n_clusters),
            evaluated_samples=evaluated_samples,
            population_size=population_size,
            inertia=inertia,
            normalized_inertia=normalized_inertia,
            cohesion_rms_distance=float(np.sqrt(normalized_inertia)),
            between_cluster_dispersion=between_dispersion,
            normalized_between_cluster_dispersion=normalized_between,
            separation_rms_distance=float(np.sqrt(normalized_between)),
            calinski_harabasz_score=calinski_harabasz,
            silhouette_score=silhouette.score,
            silhouette_metric=self.silhouette_metric,
            silhouette_sample_size=silhouette.sample_size,
            silhouette_represented_clusters=silhouette.represented_clusters,
            silhouette_singleton_clusters=silhouette.singleton_clusters,
            silhouette_cluster_coverage=silhouette.represented_clusters / model.n_clusters,
            cluster_sizes=size_stats,
            per_cluster=per_cluster,
        )

    def _silhouette(self, vectors: np.ndarray, labels: np.ndarray, total_clusters: int) -> _SilhouetteResult:
        rng = np.random.default_rng(self.random_state)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        eligible_labels = unique_labels[label_counts >= 2]
        max_clusters = self.silhouette_sample_size // self.silhouette_samples_per_cluster
        if len(eligible_labels) > max_clusters:
            eligible_labels = rng.choice(eligible_labels, size=max_clusters, replace=False)

        indexes: list[np.ndarray] = []
        if len(eligible_labels) > 0:
            per_cluster_budget = max(
                self.silhouette_samples_per_cluster,
                self.silhouette_sample_size // len(eligible_labels),
            )
            for label in eligible_labels:
                cluster_indexes = np.flatnonzero(labels == label)
                take = min(per_cluster_budget, len(cluster_indexes))
                indexes.append(rng.choice(cluster_indexes, size=take, replace=False))

        selected_indexes = np.concatenate(indexes) if indexes else np.empty(0, dtype=np.int64)
        if len(selected_indexes) > self.silhouette_sample_size:
            selected_indexes = rng.choice(selected_indexes, size=self.silhouette_sample_size, replace=False)
        sampled_vectors = vectors[selected_indexes]
        sampled_labels = labels[selected_indexes]
        sample_size = len(sampled_vectors)

        sampled_unique_labels, sampled_label_counts = np.unique(sampled_labels, return_counts=True)
        represented = len(sampled_unique_labels)
        singleton_clusters = int(np.count_nonzero(sampled_label_counts == 1))
        score: float | None = None
        if 2 <= represented < sample_size:
            try:
                score = float(silhouette_score(sampled_vectors, sampled_labels, metric=self.silhouette_metric))
            except ValueError as error:
                logger.warning(
                    "Silhouette calculation was unavailable",
                    error=str(error),
                    sample_size=sample_size,
                    represented_clusters=represented,
                )
                score = None
        else:
            logger.warning(
                "Silhouette sample lacks enough represented clusters",
                sample_size=sample_size,
                represented_clusters=represented,
                total_clusters=total_clusters,
            )
        return _SilhouetteResult(
            score=score,
            sample_size=sample_size,
            represented_clusters=represented,
            singleton_clusters=singleton_clusters,
        )


@dataclass(frozen=True)
class _SilhouetteResult:
    score: float | None
    sample_size: int
    represented_clusters: int
    singleton_clusters: int


def _assigned_squared_distances(vectors: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    residuals = vectors - centers[labels]
    return np.einsum("ij,ij->i", residuals, residuals).astype(np.float64)


def _calinski_harabasz(
    *,
    within_dispersion: float,
    between_dispersion: float,
    sample_count: int,
    represented_clusters: int,
) -> float | None:
    if represented_clusters <= 1 or sample_count <= represented_clusters or within_dispersion <= 0:
        return None
    return float((between_dispersion / (represented_clusters - 1)) / (within_dispersion / (sample_count - represented_clusters)))


def _cluster_size_statistics(counts: np.ndarray, *, min_cluster_size: int) -> ClusterSizeStatistics:
    nonempty = counts[counts > 0]
    if len(nonempty) == 0:
        return ClusterSizeStatistics(
            total_clusters=len(counts),
            represented_clusters=0,
            empty_clusters=len(counts),
            singleton_clusters=0,
            below_minimum_clusters=0,
            minimum=0,
            percentile_05=0.0,
            median=0.0,
            mean=0.0,
            percentile_95=0.0,
            maximum=0,
        )
    return ClusterSizeStatistics(
        total_clusters=len(counts),
        represented_clusters=len(nonempty),
        empty_clusters=int(np.count_nonzero(counts == 0)),
        singleton_clusters=int(np.count_nonzero(counts == 1)),
        below_minimum_clusters=int(np.count_nonzero((counts > 0) & (counts < min_cluster_size))),
        minimum=int(nonempty.min()),
        percentile_05=float(np.percentile(nonempty, 5)),
        median=float(np.median(nonempty)),
        mean=float(np.mean(nonempty)),
        percentile_95=float(np.percentile(nonempty, 95)),
        maximum=int(nonempty.max()),
    )


def _per_cluster_quality(counts: np.ndarray, per_cluster_inertia: np.ndarray) -> tuple[PerClusterQuality, ...]:
    reports: list[PerClusterQuality] = []
    for cluster_id in np.flatnonzero(counts):
        size = int(counts[cluster_id])
        inertia = float(per_cluster_inertia[cluster_id])
        mean_squared_distance = inertia / size
        reports.append(
            PerClusterQuality(
                cluster_id=int(cluster_id),
                size=size,
                inertia=inertia,
                mean_squared_distance=mean_squared_distance,
                rms_distance=float(np.sqrt(mean_squared_distance)),
            )
        )
    return tuple(reports)
