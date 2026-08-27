from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from argus.configs import get_logger

from .evaluation import ClusterQualityEvaluator, ClusterQualityReport
from .kmeans import MiniBatchKMeansClusterer
from .sampling import EmbeddingSample
from .sources import EmbeddingSource

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateKResult:
    k: int
    quality: ClusterQualityReport

    def summary(self) -> dict:
        sizes = self.quality.cluster_sizes
        return {
            "k": self.k,
            "normalized_inertia": self.quality.normalized_inertia,
            "cohesion_rms_distance": self.quality.cohesion_rms_distance,
            "separation_rms_distance": self.quality.separation_rms_distance,
            "calinski_harabasz_score": self.quality.calinski_harabasz_score,
            "silhouette_score": self.quality.silhouette_score,
            "silhouette_sample_size": self.quality.silhouette_sample_size,
            "silhouette_represented_clusters": self.quality.silhouette_represented_clusters,
            "silhouette_singleton_clusters": self.quality.silhouette_singleton_clusters,
            "silhouette_cluster_coverage": self.quality.silhouette_cluster_coverage,
            "represented_clusters": sizes.represented_clusters,
            "empty_clusters": sizes.empty_clusters,
            "singleton_clusters": sizes.singleton_clusters,
        }


@dataclass(frozen=True)
class KSelectionResult:
    selected_k: int
    selection_reason: str
    elbow_k: int | None
    silhouette_best_k: int | None
    calinski_harabasz_best_k: int | None
    candidates: tuple[CandidateKResult, ...]

    def to_dict(self, *, include_per_cluster: bool = False) -> dict:
        candidate_rows = []
        for candidate in self.candidates:
            quality = candidate.quality.to_dict()
            if not include_per_cluster:
                quality.pop("per_cluster", None)
            candidate_rows.append({"k": candidate.k, "quality": quality})
        return {
            "selected_k": self.selected_k,
            "selection_reason": self.selection_reason,
            "elbow_k": self.elbow_k,
            "silhouette_best_k": self.silhouette_best_k,
            "calinski_harabasz_best_k": self.calinski_harabasz_best_k,
            "candidates": candidate_rows,
        }


class ClusterCountSelector:
    def __init__(
        self,
        *,
        evaluator: ClusterQualityEvaluator,
        random_state: int,
        elbow_window_factor: float = 2.0,
        max_silhouette_singleton_fraction: float = 0.5,
    ) -> None:
        if elbow_window_factor < 1:
            raise ValueError("elbow_window_factor must be >= 1.")
        if not 0 <= max_silhouette_singleton_fraction <= 1:
            raise ValueError("max_silhouette_singleton_fraction must be in [0, 1].")
        self.evaluator = evaluator
        self.random_state = random_state
        self.elbow_window_factor = elbow_window_factor
        self.max_silhouette_singleton_fraction = max_silhouette_singleton_fraction

    def search(
        self,
        source: EmbeddingSource,
        evaluation_sample: EmbeddingSample,
        *,
        n: int,
        batch_size: int,
        min_samples_per_cluster: int,
        num_candidates: int,
    ) -> KSelectionResult:
        k_values = candidate_k_values(
            n=n,
            min_samples_per_cluster=min_samples_per_cluster,
            num_candidates=num_candidates,
        )
        results: list[CandidateKResult] = []
        sample_minimum = max(2, int(np.ceil(min_samples_per_cluster * len(evaluation_sample) / n)))
        logger.info(
            "Starting candidate K evaluation",
            population_size=n,
            candidate_count=len(k_values),
            candidate_min=int(k_values[0]),
            candidate_max=int(k_values[-1]),
            evaluation_sample_size=len(evaluation_sample),
        )

        for k in tqdm(k_values, desc="Evaluating candidate K"):
            clusterer = MiniBatchKMeansClusterer(random_state=self.random_state)
            try:
                initialization_vectors = evaluation_sample.vectors if len(evaluation_sample) >= k else None
                clusterer.fit(
                    source,
                    k=int(k),
                    batch_size=batch_size,
                    initialization_vectors=initialization_vectors,
                )
            except ValueError as error:
                logger.warning("Stopping candidate K evaluation", k=int(k), error=str(error))
                break
            if clusterer.kmeans is None:
                raise RuntimeError("Candidate clusterer did not produce a fitted model.")
            quality = self.evaluator.evaluate_sample(
                clusterer.kmeans,
                evaluation_sample,
                min_cluster_size=sample_minimum,
            )
            results.append(CandidateKResult(k=int(k), quality=quality))
            logger.info(
                "Evaluated candidate K",
                k=int(k),
                cohesion=quality.cohesion_rms_distance,
                separation=quality.separation_rms_distance,
                silhouette=quality.silhouette_score,
                calinski_harabasz=quality.calinski_harabasz_score,
                empty_clusters=quality.cluster_sizes.empty_clusters,
                singleton_clusters=quality.cluster_sizes.singleton_clusters,
            )

        if not results:
            logger.exception("No candidate K values could be evaluated", population_size=n)
            raise RuntimeError("Unable to evaluate any candidate K values.")
        selection = self._select(tuple(results))
        logger.info(
            "Completed candidate K selection",
            selected_k=selection.selected_k,
            reason=selection.selection_reason,
            elbow_k=selection.elbow_k,
            silhouette_best_k=selection.silhouette_best_k,
            calinski_harabasz_best_k=selection.calinski_harabasz_best_k,
        )
        return selection

    def _select(self, candidates: tuple[CandidateKResult, ...]) -> KSelectionResult:
        elbow_k = locate_elbow(candidates)
        pool = self._elbow_neighborhood(candidates, elbow_k)
        reliable_silhouettes = [candidate for candidate in pool if self._has_reliable_silhouette(candidate)]
        silhouette_best = max(reliable_silhouettes, key=_silhouette_key, default=None)
        ch_candidates = [candidate for candidate in pool if candidate.quality.calinski_harabasz_score is not None]
        ch_best = max(ch_candidates, key=_ch_key, default=None)

        if silhouette_best is not None:
            selected = silhouette_best
            reason = "best reliable sampled silhouette inside the inertia-elbow neighborhood"
        elif ch_best is not None:
            selected = ch_best
            reason = "best Calinski-Harabasz score inside the inertia-elbow neighborhood; sampled silhouette was unreliable"
        elif elbow_k is not None:
            selected = min(candidates, key=lambda candidate: abs(candidate.k - elbow_k))
            reason = "inertia elbow; silhouette and Calinski-Harabasz were unavailable"
        else:
            selected = min(candidates, key=lambda candidate: candidate.k)
            reason = "smallest evaluated K; no reliable validation metric or inertia elbow was available"

        all_reliable = [candidate for candidate in candidates if self._has_reliable_silhouette(candidate)]
        global_silhouette_best = max(all_reliable, key=_silhouette_key, default=None)
        all_ch = [candidate for candidate in candidates if candidate.quality.calinski_harabasz_score is not None]
        global_ch_best = max(all_ch, key=_ch_key, default=None)
        return KSelectionResult(
            selected_k=selected.k,
            selection_reason=reason,
            elbow_k=elbow_k,
            silhouette_best_k=global_silhouette_best.k if global_silhouette_best else None,
            calinski_harabasz_best_k=global_ch_best.k if global_ch_best else None,
            candidates=candidates,
        )

    def _elbow_neighborhood(
        self,
        candidates: tuple[CandidateKResult, ...],
        elbow_k: int | None,
    ) -> tuple[CandidateKResult, ...]:
        if elbow_k is None:
            return candidates
        lower = elbow_k / self.elbow_window_factor
        upper = elbow_k * self.elbow_window_factor
        neighborhood = tuple(candidate for candidate in candidates if lower <= candidate.k <= upper)
        return neighborhood or candidates

    def _has_reliable_silhouette(self, candidate: CandidateKResult) -> bool:
        quality = candidate.quality
        if quality.silhouette_score is None or quality.silhouette_represented_clusters < 2:
            return False
        represented_clusters = quality.cluster_sizes.represented_clusters
        if represented_clusters == 0:
            return False
        singleton_fraction = quality.cluster_sizes.singleton_clusters / represented_clusters
        return singleton_fraction <= self.max_silhouette_singleton_fraction


def candidate_k_values(*, n: int, min_samples_per_cluster: int, num_candidates: int) -> np.ndarray:
    if n <= 2:
        raise ValueError("n must be greater than 2 to search for K.")
    if min_samples_per_cluster <= 0:
        raise ValueError("min_samples_per_cluster must be > 0.")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be > 0.")
    k_max = max(2, n // min_samples_per_cluster)
    if k_max == 2:
        return np.array([2], dtype=np.int64)
    return np.unique(np.logspace(np.log10(2), np.log10(k_max), num=num_candidates).astype(np.int64))


def locate_elbow(candidates: tuple[CandidateKResult, ...]) -> int | None:
    if len(candidates) < 3:
        logger.warning("Cannot locate inertia elbow with fewer than three candidates", candidates=len(candidates))
        return None
    k_values = np.asarray([candidate.k for candidate in candidates], dtype=np.int64)
    inertias = [candidate.quality.normalized_inertia for candidate in candidates]
    knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing").knee
    if knee is None:
        logger.warning("KneeLocator did not identify an inertia elbow", candidates=len(candidates))
        return None
    logger.debug("Located inertia elbow", elbow_k=int(knee))
    return int(knee)


def save_selection_report(result: KSelectionResult, *, output_dir: str | Path, name: str) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}_quality.json"
    csv_path = directory / f"{name}_candidates.csv"
    plot_path = directory / f"{name}_quality.png"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    rows = [candidate.summary() for candidate in result.candidates]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_selection(result, plot_path)
    logger.info(
        "Saved clustering selection reports",
        json_file=str(json_path),
        csv_file=str(csv_path),
        plot_file=str(plot_path),
    )
    return json_path, csv_path, plot_path


def save_quality_report(report: ClusterQualityReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved clustering quality report", path=str(output), scope=report.scope, k=report.k)
    return output


def _plot_selection(result: KSelectionResult, path: Path) -> None:
    k_values = [candidate.k for candidate in result.candidates]
    metrics: tuple[tuple[str, list[float | None]], ...] = (
        ("Normalized inertia", [candidate.quality.normalized_inertia for candidate in result.candidates]),
        ("Silhouette", [candidate.quality.silhouette_score for candidate in result.candidates]),
        ("Calinski-Harabasz", [candidate.quality.calinski_harabasz_score for candidate in result.candidates]),
        ("Cohesion RMS", [candidate.quality.cohesion_rms_distance for candidate in result.candidates]),
        ("Separation RMS", [candidate.quality.separation_rms_distance for candidate in result.candidates]),
        ("Sample cluster coverage", [candidate.quality.silhouette_cluster_coverage for candidate in result.candidates]),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (title, values) in zip(axes.flat, metrics):
        x: list[int] = []
        y: list[float] = []
        for k, value in zip(k_values, values):
            if value is not None:
                x.append(k)
                y.append(float(value))
        axis.plot(x, y, marker="o")
        axis.axvline(result.selected_k, color="tab:red", linestyle="--", label="selected K")
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("K")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _silhouette_key(candidate: CandidateKResult) -> tuple[float, int]:
    score = candidate.quality.silhouette_score
    return (float(score) if score is not None else float("-inf"), -candidate.k)


def _ch_key(candidate: CandidateKResult) -> tuple[float, int]:
    score = candidate.quality.calinski_harabasz_score
    return (float(score) if score is not None else float("-inf"), -candidate.k)
