from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from sastllm.dtos import GetClassificationRepositoryDto
from scripts.utils import load_yaml

DEFAULT_LABEL_MAP: Dict[int, str] = {0: "benign", 1: "malicious"}


def _load_binary_label_map(config_path: str = "configs/split.yaml") -> Dict[int, str]:
    """Load label mapping from YAML config.

    Expected YAML formats supported under key `classification.labels`:
    1) List of single-key mappings (current format):
       labels:
         - 0: "benign"
         - 1: "malicious"
    2) Direct mapping:
       labels:
         0: "benign"
         1: "malicious"

    Returns a dict[int, str] or DEFAULT_LABEL_MAP on failure.
    """
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_LABEL_MAP
    try:
        data = load_yaml(path)
        labels_raw = data.get("split", {}).get("binary_labels", None)
        if labels_raw is None:
            return DEFAULT_LABEL_MAP
        label_map: Dict[int, str] = {}
        if isinstance(labels_raw, dict):
            for k, v in labels_raw.items():
                if isinstance(k, int) and isinstance(v, str):
                    label_map[k] = v
        elif isinstance(labels_raw, list):
            for item in labels_raw:
                if isinstance(item, dict) and len(item) == 1:
                    ((k, v),) = item.items()
                    if isinstance(k, int) and isinstance(v, str):
                        label_map[k] = v
        return label_map or DEFAULT_LABEL_MAP
    except Exception:
        return DEFAULT_LABEL_MAP


BINARY_LABEL_MAP: Dict[int, str] = _load_binary_label_map()
BINARY_LABEL_TO_INDEX: Dict[str, int] = {v: k for k, v in BINARY_LABEL_MAP.items()}


class RepositoryEncoder:
    """
    Numpy encoder for cluster IDs.
    - Output is a (num_clusters,) float32 vector of per-cluster percentages.
    - Accepts dicts of counts {cid: freq} or iterables of cids (implies freq=1).
    - Validates IDs and disallows mixed types.
    - Supports single and batch operations.
    """

    def __init__(self, num_clusters: int, label_to_index: Dict[str, int]) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters must be > 0")
        self.num_clusters = num_clusters
        self.label_to_index = label_to_index

    # ---------- public API ----------
    def encode_ids(self, ids: Optional[Union[Iterable[int], Dict[int, int]]]) -> np.ndarray:
        """
        Encode IDs or counts -> (num_clusters,) float32 vector of percentages.
        If ids is None/empty, returns zeros.
        """
        x = np.zeros(self.num_clusters, dtype=np.float32)
        if not ids:
            return x

        # Normalize to a {cid: count} dict
        if isinstance(ids, dict):
            counts = self._validated_counts(ids)
        else:
            # iterable of ints -> each gets count 1
            self._validate_all_int(ids)
            counts = {}
            for cid in ids:
                self._validate_id(cid)
                counts[cid] = counts.get(cid, 0) + 1

        total = sum(counts.values())
        if total <= 0:
            return x  # defensive; should not happen after validation

        for cid, c in counts.items():
            self._validate_id(cid)
            x[cid] = float(c) / float(total)

        return x

    def decode_vec(self, vec: np.ndarray, threshold: float = 0.5) -> List[int]:
        """
        Decode a percentage vector -> sorted list of active IDs (0-based) using a threshold.
        """
        self._validate_vec(vec)
        return (np.where(vec >= threshold)[0]).tolist()

    def encode_repo(self, repo: GetClassificationRepositoryDto) -> np.ndarray:
        """
        Encode a ClassificationRepository:
          - repo.data is expected to be Dict[int, int] (counts) or None.
        """
        if repo.data is None:
            return np.zeros(self.num_clusters, dtype=np.float32)
        return self.encode_ids(repo.data)

    def encode_repos(self, repos):
        n = len(repos)
        X = np.zeros((n, self.num_clusters), dtype=np.float32)
        ids = np.empty(n, dtype=np.int32)
        y = np.empty(n, dtype=np.int16)

        for i, r in enumerate(repos):
            ids[i] = r.repository_id

            # clusters
            if r.data:
                counts = self._validated_counts(r.data)
                total = float(sum(counts.values()))
                if total > 0:
                    for cid, c in counts.items():
                        X[i, cid] = c / total

            if r.label is None:
                y[i] = -1
            else:
                label = r.label
                y[i] = self.label_to_index[label] if label in self.label_to_index else -1

        # normalize entire matrix (your choice)
        norm = np.linalg.norm(X)
        if norm > 0:
            X /= norm

        return X, ids, y

    def encode_repo_tokens(
        self,
        repo: GetClassificationRepositoryDto,
        max_tokens: int = 512,
    ):
        if not repo.data:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
            )

        items = []
        for cid, count in repo.data.items():
            # HARD SAFETY CHECK
            if not isinstance(cid, int):
                continue
            if cid < 0 or cid >= self.num_clusters:
                continue
            if count <= 0:
                continue
            items.append((cid, count))

        if not items:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
            )

        # keep most frequent
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:max_tokens]

        cids, counts = zip(*items)
        total = float(sum(counts))

        tokens = torch.tensor(cids, dtype=torch.long)
        values = torch.tensor([c / total for c in counts], dtype=torch.float32)

        return tokens, values

    # ---------- validation helpers ----------
    def _validate_id(self, cid) -> None:
        if not isinstance(cid, int):
            raise TypeError(f"Cluster id must be int, got {type(cid)}: {cid!r}")
        if cid < 0 or cid >= self.num_clusters:
            raise ValueError(f"Cluster id {cid} out of range [0, {self.num_clusters})")

    def _validate_vec(self, vec: np.ndarray) -> None:
        if not isinstance(vec, np.ndarray):
            raise TypeError("vec must be a numpy.ndarray")
        if vec.ndim != 1 or vec.shape[0] != self.num_clusters:
            raise ValueError(f"Expected shape ({self.num_clusters},), got {vec.shape}")
        if not np.issubdtype(vec.dtype, np.floating):
            raise TypeError(f"vec dtype must be float, got {vec.dtype}")

    def _validate_all_int(self, data: Iterable[Union[int, str]]) -> None:
        for x in data:
            if not isinstance(x, int):
                raise TypeError(f"data must be all ints for encoding; got non-int element: {x!r} (type {type(x)})")

    def _validated_counts(self, counts: Dict[int, int]) -> Dict[int, int]:
        if not isinstance(counts, dict):
            raise TypeError(f"counts must be a dict[int, int], got {type(counts)}")
        out: Dict[int, int] = {}
        for k, v in counts.items():
            if not isinstance(k, int):
                raise TypeError(f"Cluster id must be int, got key {k!r} (type {type(k)})")
            if not isinstance(v, int):
                raise TypeError(f"Count must be int for cid {k}, got {type(v)}")
            if v < 0:
                raise ValueError(f"Count must be >= 0 for cid {k}, got {v}")
            if v == 0:
                continue  # ignore zero-count entries
            out[k] = v
        return out
