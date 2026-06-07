from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import torch

from sastllm.dtos import GetClassificationRepositoryDto

from .config import load_label_map


@dataclass(frozen=True)
class LabelMapping:
    index_to_label: dict[int, str]

    @classmethod
    def from_split_config(cls, config_path: str = "configs/split.yaml") -> "LabelMapping":
        return cls(index_to_label=load_label_map(config_path))

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in self.index_to_label.items()}

    def normalize_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        return "benign" if label == "benign" else "malicious"

    def encode_label(self, label: str | None) -> int:
        normalized = self.normalize_label(label)
        if normalized is None:
            return -1
        return self.label_to_index.get(normalized, -1)


@dataclass(frozen=True)
class RepositoryEncoding:
    repository_ids: np.ndarray
    features: np.ndarray
    labels: np.ndarray


class RepositoryEncoderProtocol(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def encode(self, repositories: Iterable[GetClassificationRepositoryDto]) -> RepositoryEncoding: ...


class ClusterDistributionEncoder:
    """
    Encodes repositories as fixed-width distributions over functionality clusters.

    This is the current thesis baseline. Other encoders can implement the same
    `encode()` contract for sequence/time-series/transformer inputs later.
    """

    def __init__(
        self,
        *,
        num_clusters: int,
        labels: LabelMapping,
        matrix_normalization: bool = True,
    ) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0.")
        self.num_clusters = num_clusters
        self.labels = labels
        self.matrix_normalization = matrix_normalization

    @property
    def feature_dim(self) -> int:
        return self.num_clusters

    def encode(self, repositories: Iterable[GetClassificationRepositoryDto]) -> RepositoryEncoding:
        repos = list(repositories)
        features = np.zeros((len(repos), self.num_clusters), dtype=np.float32)
        repository_ids = np.empty(len(repos), dtype=np.int64)
        labels = np.empty(len(repos), dtype=np.int64)

        for row, repo in enumerate(repos):
            repository_ids[row] = int(repo.repository_id)
            labels[row] = self.labels.encode_label(repo.label)

            counts = self._validated_counts(repo.data or {})
            total = float(sum(counts.values()))
            if total > 0:
                for cluster_id, count in counts.items():
                    features[row, cluster_id] = count / total

        if self.matrix_normalization:
            norm = np.linalg.norm(features)
            if norm > 0:
                features /= norm

        return RepositoryEncoding(repository_ids=repository_ids, features=features, labels=labels)

    def encode_ids(self, ids: Iterable[int] | dict[int, int] | None) -> np.ndarray:
        if ids is None:
            return np.zeros(self.num_clusters, dtype=np.float32)
        counts = self._validated_counts(ids if isinstance(ids, dict) else {cluster_id: 1 for cluster_id in ids})
        total = float(sum(counts.values()))
        vector = np.zeros(self.num_clusters, dtype=np.float32)
        if total <= 0:
            return vector
        for cluster_id, count in counts.items():
            vector[cluster_id] = count / total
        return vector

    def encode_repo_tokens(
        self,
        repo: GetClassificationRepositoryDto,
        max_tokens: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = self._validated_counts(repo.data or {})
        items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:max_tokens]
        if not items:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
        cluster_ids, frequencies = zip(*items)
        total = float(sum(frequencies))
        return (
            torch.tensor(cluster_ids, dtype=torch.long),
            torch.tensor([freq / total for freq in frequencies], dtype=torch.float32),
        )

    def decode_vec(self, vector: np.ndarray, threshold: float = 0.5) -> list[int]:
        if vector.ndim != 1 or vector.shape[0] != self.num_clusters:
            raise ValueError(f"Expected vector shape ({self.num_clusters},), got {vector.shape}.")
        return np.where(vector >= threshold)[0].tolist()

    def _validated_counts(self, counts: dict[int, int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for key, value in counts.items():
            if not isinstance(key, int):
                raise TypeError(f"Cluster id must be int, got {type(key)}.")
            if not isinstance(value, int):
                raise TypeError(f"Cluster count must be int, got {type(value)}.")
            if key < 0 or key >= self.num_clusters:
                raise ValueError(f"Cluster id {key} out of range [0, {self.num_clusters}).")
            if value > 0:
                out[key] = value
        return out
