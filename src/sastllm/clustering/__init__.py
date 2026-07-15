"""
Clustering pipeline components.

This package owns the functionality-clustering phase: configuration parsing,
embedding retrieval, streaming MiniBatchKMeans, and cluster-id persistence.
"""

from .artifacts import ClusteringRunArtifacts, search_artifacts, training_artifacts
from .config import ClusteringConfig, ClusteringMode, EvaluationConfig, SearchConfig, TestConfig, TrainConfig
from .evaluation import ClusterQualityEvaluator, ClusterQualityReport
from .kmeans import MiniBatchKMeansClusterer
from .selection import ClusterCountSelector, KSelectionResult
from .service import FunctionalityClusteringService

__all__ = [
    "ClusteringConfig",
    "ClusteringMode",
    "ClusteringRunArtifacts",
    "ClusterCountSelector",
    "ClusterQualityEvaluator",
    "ClusterQualityReport",
    "EvaluationConfig",
    "FunctionalityClusteringService",
    "MiniBatchKMeansClusterer",
    "KSelectionResult",
    "SearchConfig",
    "TestConfig",
    "TrainConfig",
    "search_artifacts",
    "training_artifacts",
]
