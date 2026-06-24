from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

import numpy as np
import torch

from sastllm.entities import RepositoryWithClusterDistribution

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
    sequence_lengths: np.ndarray | None = None


class RepositoryEncoderProtocol(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def encode(self, repositories: Iterable[RepositoryWithClusterDistribution]) -> RepositoryEncoding: ...


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

    def encode(self, repositories: Iterable[RepositoryWithClusterDistribution]) -> RepositoryEncoding:
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
        repo: RepositoryWithClusterDistribution,
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


class OrderedFunctionalityTimeSeriesEncoder:
    """
    Encodes each repository as an ordered functionality-cluster sequence.

    Output shape is `(num_repositories, sequence_length, num_clusters)`.
    Each time step is one functionality sorted by its database-assigned
    `functionality_id`. Those ids are used only for ordering and are not
    converted into feature indexes. The feature vector is a one-hot encoding of
    the code-assigned zero-based `cluster_id`.

    If `max_sequence_length` is omitted, the batch is padded to the longest
    repository sequence in the encoded batch. If provided, longer sequences are
    truncated according to `truncation`.
    """

    def __init__(
        self,
        *,
        num_clusters: int,
        labels: LabelMapping,
        max_sequence_length: int | None = None,
        truncation: Literal["first", "last"] = "first",
    ) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be > 0 when provided.")
        if truncation not in {"first", "last"}:
            raise ValueError("truncation must be either 'first' or 'last'.")
        self.num_clusters = num_clusters
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.truncation = truncation

    @property
    def feature_dim(self) -> int:
        return self.num_clusters

    def encode(self, repositories: Iterable[RepositoryWithClusterDistribution]) -> RepositoryEncoding:
        repos = list(repositories)
        repository_ids = np.empty(len(repos), dtype=np.int64)
        labels = np.empty(len(repos), dtype=np.int64)
        sequences = [self._cluster_ids_ordered_by_functionality_id(repo) for repo in repos]

        sequence_length = self.max_sequence_length
        if sequence_length is None:
            sequence_length = max((len(sequence) for sequence in sequences), default=0)

        features = np.zeros((len(repos), sequence_length, self.num_clusters), dtype=np.float32)
        sequence_lengths = np.zeros(len(repos), dtype=np.int64)

        for row, (repo, sequence) in enumerate(zip(repos, sequences)):
            repository_ids[row] = int(repo.repository_id)
            labels[row] = self.labels.encode_label(repo.label)

            truncated = self._truncate(sequence, sequence_length)
            sequence_lengths[row] = len(truncated)
            for step, cluster_id in enumerate(truncated):
                features[row, step, self._validated_cluster_index(cluster_id)] = 1.0

        return RepositoryEncoding(
            repository_ids=repository_ids,
            features=features,
            labels=labels,
            sequence_lengths=sequence_lengths,
        )

    def encode_repo(self, repo: RepositoryWithClusterDistribution) -> np.ndarray:
        sequence = self._cluster_ids_ordered_by_functionality_id(repo)
        sequence_length = self.max_sequence_length or len(sequence)
        features = np.zeros((sequence_length, self.num_clusters), dtype=np.float32)
        for step, cluster_id in enumerate(self._truncate(sequence, sequence_length)):
            features[step, self._validated_cluster_index(cluster_id)] = 1.0
        return features

    def _cluster_ids_ordered_by_functionality_id(self, repo: RepositoryWithClusterDistribution) -> list[int]:
        if not repo.ordered_functionalities:
            return []
        ordered = sorted(repo.ordered_functionalities, key=lambda functionality: functionality.functionality_id)
        return [int(functionality.cluster_id) for functionality in ordered if functionality.cluster_id is not None]

    def _truncate(self, sequence: list[int], sequence_length: int) -> list[int]:
        if len(sequence) <= sequence_length:
            return sequence
        if self.truncation == "first":
            return sequence[:sequence_length]
        return sequence[-sequence_length:]

    def _validated_cluster_index(self, cluster_id: int) -> int:
        if 0 <= cluster_id < self.num_clusters:
            return cluster_id
        raise ValueError(f"Cluster id {cluster_id} out of range [0, {self.num_clusters}).")


class OrderedFunctionalityTokenSequenceEncoder:
    """
    Encodes each repository as an ordered sequence of cluster-id tokens.

    This is the memory-efficient sequence representation for embedding-based
    models such as LSTMs and Transformers. Token `0` is reserved for padding;
    real zero-based cluster ids are shifted by one, so cluster id `0` becomes
    token `1` and cluster id `num_clusters - 1` becomes token `num_clusters`.
    """

    padding_token_id = 0

    def __init__(
        self,
        *,
        num_clusters: int,
        labels: LabelMapping,
        max_sequence_length: int | None = None,
        truncation: Literal["first", "last"] = "first",
    ) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be > 0 when provided.")
        if truncation not in {"first", "last"}:
            raise ValueError("truncation must be either 'first' or 'last'.")
        self.num_clusters = num_clusters
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.truncation = truncation

    @property
    def feature_dim(self) -> int:
        return self.num_clusters + 1

    def encode(self, repositories: Iterable[RepositoryWithClusterDistribution]) -> RepositoryEncoding:
        repos = list(repositories)
        repository_ids = np.empty(len(repos), dtype=np.int64)
        labels = np.empty(len(repos), dtype=np.int64)
        sequences = [self._cluster_ids_ordered_by_functionality_id(repo) for repo in repos]

        sequence_length = self.max_sequence_length
        if sequence_length is None:
            sequence_length = max((len(sequence) for sequence in sequences), default=0)

        features = np.zeros((len(repos), sequence_length), dtype=np.int64)
        sequence_lengths = np.zeros(len(repos), dtype=np.int64)

        for row, (repo, sequence) in enumerate(zip(repos, sequences)):
            repository_ids[row] = int(repo.repository_id)
            labels[row] = self.labels.encode_label(repo.label)

            truncated = self._truncate(sequence, sequence_length)
            sequence_lengths[row] = len(truncated)
            for step, cluster_id in enumerate(truncated):
                features[row, step] = self._cluster_token_id(cluster_id)

        return RepositoryEncoding(
            repository_ids=repository_ids,
            features=features,
            labels=labels,
            sequence_lengths=sequence_lengths,
        )

    def encode_repo(self, repo: RepositoryWithClusterDistribution) -> np.ndarray:
        sequence = self._cluster_ids_ordered_by_functionality_id(repo)
        sequence_length = self.max_sequence_length or len(sequence)
        features = np.zeros(sequence_length, dtype=np.int64)
        for step, cluster_id in enumerate(self._truncate(sequence, sequence_length)):
            features[step] = self._cluster_token_id(cluster_id)
        return features

    def _cluster_ids_ordered_by_functionality_id(self, repo: RepositoryWithClusterDistribution) -> list[int]:
        if not repo.ordered_functionalities:
            return []
        ordered = sorted(repo.ordered_functionalities, key=lambda functionality: functionality.functionality_id)
        return [int(functionality.cluster_id) for functionality in ordered if functionality.cluster_id is not None]

    def _truncate(self, sequence: list[int], sequence_length: int) -> list[int]:
        if len(sequence) <= sequence_length:
            return sequence
        if self.truncation == "first":
            return sequence[:sequence_length]
        return sequence[-sequence_length:]

    def _cluster_token_id(self, cluster_id: int) -> int:
        if 0 <= cluster_id < self.num_clusters:
            return cluster_id + 1
        raise ValueError(f"Cluster id {cluster_id} out of range [0, {self.num_clusters}).")
