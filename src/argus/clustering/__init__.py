"""
Clustering pipeline components.

This package owns the functionality-clustering phase: configuration parsing,
embedding retrieval, streaming MiniBatchKMeans, and cluster-id persistence.
"""

from .config import ClusteringConfig, ClusteringMode, EvaluationConfig, SearchConfig, TestConfig, TrainConfig
from .evaluation import ClusterQualityEvaluator, ClusterQualityReport
from .kmeans import MiniBatchKMeansClusterer
from .selection import ClusterCountSelector, KSelectionResult
from .service import FunctionalityClusteringService

__all__ = [
    "ClusterCountSelector",
    "ClusterQualityEvaluator",
    "ClusterQualityReport",
    "ClusteringConfig",
    "ClusteringMode",
    "EvaluationConfig",
    "FunctionalityClusteringService",
    "KSelectionResult",
    "MiniBatchKMeansClusterer",
    "SearchConfig",
    "TestConfig",
    "TrainConfig",
]
