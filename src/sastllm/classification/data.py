from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from sklearn.model_selection import train_test_split

from sastllm.db import RepositoryManager
from sastllm.dtos import GetClassificationRepositoryDto
from sastllm.ml import RepositoryTensorDataset

from .encoders import RepositoryEncoderProtocol


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

        encoding = self.encoder.encode(self._normalize_labels(repositories))
        dataset = RepositoryTensorDataset(
            ids=torch.tensor(encoding.repository_ids, dtype=torch.long),
            X=torch.tensor(encoding.features, dtype=torch.float32),
            y=torch.tensor(encoding.labels, dtype=torch.long),
        )

        id_to_index = {int(repo_id): index for index, repo_id in enumerate(encoding.repository_ids.tolist())}
        train_ids, train_labels = self._ids_and_labels("train")
        test_ids, _ = self._ids_and_labels("test")

        if not train_ids:
            return DatasetBundle(
                dataset=dataset,
                train_indices=[],
                val_indices=[],
                test_indices=[id_to_index[repo_id] for repo_id in test_ids if repo_id in id_to_index],
            )

        train_ids_arr, val_ids_arr = train_test_split(
            train_ids,
            test_size=validation_size,
            stratify=train_labels,
            random_state=self.seed,
        )

        return DatasetBundle(
            dataset=dataset,
            train_indices=[id_to_index[repo_id] for repo_id in train_ids_arr if repo_id in id_to_index],
            val_indices=[id_to_index[repo_id] for repo_id in val_ids_arr if repo_id in id_to_index],
            test_indices=[id_to_index[repo_id] for repo_id in test_ids if repo_id in id_to_index],
        )

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
    def _normalize_labels(repositories: list[GetClassificationRepositoryDto]) -> list[GetClassificationRepositoryDto]:
        for repo in repositories:
            if repo.label != "benign":
                repo.label = "malicious"
        return repositories
