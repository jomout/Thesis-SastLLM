from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from sklearn.model_selection import train_test_split

from argus.configs import get_logger
from argus.db import RepositoryManager
from argus.entities import RepositoryWithClusterDistribution
from argus.ml import RepositoryTensorDataset

from .encoders import RepositoryEncoderProtocol

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetBundle:
    dataset: RepositoryTensorDataset
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


class RepositoryDatasetBuilder:
    def __init__(
        self,
        *,
        repository_manager: RepositoryManager,
        encoder: RepositoryEncoderProtocol,
        batch_size: int,
        seed: int,
    ) -> None:
        self.repository_manager = repository_manager
        self.encoder = encoder
        self.batch_size = batch_size
        self.seed = seed

    def build(self, *, validation_size: float) -> DatasetBundle:
        repositories = list(self.repository_manager.get_repositories_with_cluster_ids(batch_size=self.batch_size))
        if not repositories:
            raise RuntimeError("No processed repositories with cluster ids returned from DB.")

        label_counts: dict[str, int] = {}
        with_clusters = 0
        for repo in repositories:
            label = "benign" if repo.label == "benign" else "malicious"
            label_counts[label] = label_counts.get(label, 0) + 1
            if repo.data:
                with_clusters += 1
        logger.info(
            "Fetched repositories for classification",
            repositories=len(repositories),
            repositories_with_clusters=with_clusters,
            label_counts=label_counts,
        )

        encoding = self.encoder.encode(self._normalize_labels(repositories))
        feature_dtype = torch.long if encoding.features.dtype.kind in {"i", "u"} else torch.float32
        dataset = RepositoryTensorDataset(
            ids=torch.tensor(encoding.repository_ids, dtype=torch.long),
            X=torch.tensor(encoding.features, dtype=feature_dtype),
            y=torch.tensor(encoding.labels, dtype=torch.long),
        )
        nonzero = int((dataset.X != 0).sum().item())
        total_features = int(dataset.X.numel())
        logger.info(
            "Encoded repository dataset",
            samples=len(dataset),
            feature_shape=tuple(dataset.X.shape),
            feature_dim=int(dataset.X.shape[-1]),
            nonzero_features=nonzero,
            feature_density=round(nonzero / total_features, 6) if total_features else 0.0,
        )

        id_to_index = {int(repo_id): index for index, repo_id in enumerate(encoding.repository_ids.tolist())}
        train_ids, train_labels = self._ids_and_labels("train")
        test_ids, _ = self._ids_and_labels("test")
        logger.info("Repository split ids loaded", train_ids=len(train_ids), test_ids=len(test_ids), validation_size=validation_size)

        if not train_ids:
            test_indices = [id_to_index[repo_id] for repo_id in test_ids if repo_id in id_to_index]
            logger.warning("No train repositories found; building test-only bundle", test_indices=len(test_indices))
            return DatasetBundle(
                dataset=dataset,
                train_indices=[],
                val_indices=[],
                test_indices=test_indices,
            )

        train_ids_arr, val_ids_arr = train_test_split(
            train_ids,
            test_size=validation_size,
            stratify=train_labels,
            random_state=self.seed,
        )

        bundle = DatasetBundle(
            dataset=dataset,
            train_indices=[id_to_index[repo_id] for repo_id in train_ids_arr if repo_id in id_to_index],
            val_indices=[id_to_index[repo_id] for repo_id in val_ids_arr if repo_id in id_to_index],
            test_indices=[id_to_index[repo_id] for repo_id in test_ids if repo_id in id_to_index],
        )
        logger.info(
            "Built classification dataset bundle",
            train_samples=len(bundle.train_indices),
            val_samples=len(bundle.val_indices),
            test_samples=len(bundle.test_indices),
        )
        return bundle

    def _ids_and_labels(self, split: Literal["train", "test"]) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        labels: list[int] = []
        label_to_index = self.encoder.labels.label_to_index  # type: ignore[attr-defined]
        for repo in self.repository_manager.get_repositories(split=split):
            ids.append(int(repo.repository_id))
            normalized = "benign" if repo.label == "benign" else "malicious"
            labels.append(label_to_index.get(normalized, -1))
        return ids, labels

    @staticmethod
    def _normalize_labels(repositories: list[RepositoryWithClusterDistribution]) -> list[RepositoryWithClusterDistribution]:
        for repo in repositories:
            if repo.label != "benign":
                repo.label = "malicious"
        return repositories
