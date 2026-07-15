from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sastllm.configs import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusteringRunArtifacts:
    """Paths shared by one timestamped clustering run."""

    directory: Path
    stem: str

    @property
    def model(self) -> Path:
        return self.directory / f"{self.stem}.joblib"

    @property
    def quality_report(self) -> Path:
        return self.directory / f"{self.stem}_quality.json"


def training_artifacts(root: str | Path, *, k: int, timestamp: str | None = None) -> ClusteringRunArtifacts:
    run_timestamp = timestamp or _timestamp()
    stem = f"clusterer_{k}_{run_timestamp}"
    artifacts = ClusteringRunArtifacts(directory=Path(root) / stem, stem=stem)
    logger.debug("Prepared training artifact paths", k=k, artifact_dir=str(artifacts.directory))
    return artifacts


def search_artifacts(
    root: str | Path,
    *,
    n: int,
    k: int,
    timestamp: str | None = None,
) -> ClusteringRunArtifacts:
    run_timestamp = timestamp or _timestamp()
    stem = f"clusterer_{n}_{k}_{run_timestamp}"
    directory = Path(root) / f"clusterers_{n}_{run_timestamp}" / stem
    artifacts = ClusteringRunArtifacts(directory=directory, stem=stem)
    logger.debug("Prepared search artifact paths", n=n, k=k, artifact_dir=str(artifacts.directory))
    return artifacts


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
