from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx  # type: ignore[import-untyped]
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from argus.configs import get_logger

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
CLUSTERER_JOBLIB_FILENAME = "model.joblib"
CLUSTERER_ONNX_FILENAME = "model.onnx"
CLUSTERER_MANIFEST_FILENAME = "manifest.json"
CLUSTERER_QUALITY_REPORT_FILENAME = "quality.json"


@dataclass(frozen=True)
class ClusteringRunArtifacts:
    """Paths shared by one timestamped clustering run."""

    directory: Path
    stem: str

    @property
    def joblib_model(self) -> Path:
        return self.directory / CLUSTERER_JOBLIB_FILENAME

    @property
    def onnx_model(self) -> Path:
        return self.directory / CLUSTERER_ONNX_FILENAME

    @property
    def manifest(self) -> Path:
        return self.directory / CLUSTERER_MANIFEST_FILENAME

    @property
    def quality_report(self) -> Path:
        return self.directory / CLUSTERER_QUALITY_REPORT_FILENAME


@dataclass(frozen=True)
class LoadedClustererArtifact:
    model: MiniBatchKMeans
    n_clusters: int
    embedding_dimension: int
    random_state: int | None


def export_clusterer_bundle(
    *,
    model: MiniBatchKMeans,
    model_dir: str | Path,
    metadata: Mapping[str, Any] | None,
    random_state: int,
) -> Path:
    artifact_dir = Path(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = artifact_dir / CLUSTERER_JOBLIB_FILENAME
    onnx_path = artifact_dir / CLUSTERER_ONNX_FILENAME
    manifest_path = artifact_dir / CLUSTERER_MANIFEST_FILENAME

    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    n_clusters = int(model.n_clusters)
    onnx_model = to_onnx(model, centers[:1])
    onnx_bytes = onnx_model.SerializeToString()
    parity_samples = centers[: min(32, len(centers))]
    validation = _validate_onnx_parity(model, onnx_bytes, parity_samples)

    artifact_metadata = dict(metadata or {})
    artifact_metadata.update(
        {
            "clusterer": model.__class__.__name__,
            "random_state": random_state,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
        }
    )
    _atomic_joblib_dump(
        {
            "model": model,
            "n_clusters": n_clusters,
            "embedding_dimension": int(centers.shape[1]),
            "metadata": artifact_metadata,
        },
        joblib_path,
    )
    _atomic_write_bytes(onnx_path, onnx_bytes)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_type": "functionality_clusterer",
        "algorithm": "sklearn.cluster.MiniBatchKMeans",
        "n_clusters": n_clusters,
        "embedding_dimension": int(centers.shape[1]),
        "input": {
            "name": validation["input_name"],
            "dtype": "float32",
            "shape": [None, int(centers.shape[1])],
            "normalization": "l2",
        },
        "metadata": artifact_metadata,
        "validation": validation,
        "files": {
            joblib_path.name: _file_record(joblib_path),
            onnx_path.name: _file_record(onnx_path),
        },
    }
    _atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    logger.info(
        "Saved clusterer artifact bundle",
        model_dir=str(artifact_dir),
        joblib=str(joblib_path),
        onnx=str(onnx_path),
        manifest=str(manifest_path),
        parity_samples=validation["samples"],
    )
    return manifest_path


def load_clusterer_bundle(
    model_dir: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> LoadedClustererArtifact:
    artifact_dir = Path(model_dir)
    manifest = verify_clusterer_bundle(artifact_dir, expected_metadata=expected_metadata)
    joblib_path = artifact_dir / CLUSTERER_JOBLIB_FILENAME

    payload: Any = joblib.load(joblib_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Clusterer artifact must contain a mapping: {joblib_path}")
    model = payload.get("model")
    n_clusters = payload.get("n_clusters")
    embedding_dimension = payload.get("embedding_dimension")
    saved_metadata = payload.get("metadata")
    if not isinstance(model, MiniBatchKMeans):
        raise TypeError(f"Unsupported clustering model in {joblib_path}.")
    if not isinstance(n_clusters, int) or n_clusters <= 0 or n_clusters != int(model.n_clusters):
        raise ValueError(f"Missing or inconsistent cluster count in {joblib_path}.")
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0 or embedding_dimension != int(model.cluster_centers_.shape[1]):
        raise ValueError(f"Missing or inconsistent embedding dimension in {joblib_path}.")
    if manifest.get("n_clusters") != n_clusters or manifest.get("embedding_dimension") != embedding_dimension:
        raise ValueError(f"Clusterer manifest dimensions do not match {joblib_path}.")
    if not isinstance(saved_metadata, dict) or saved_metadata != manifest.get("metadata"):
        raise ValueError(f"Clusterer joblib metadata does not match {artifact_dir / CLUSTERER_MANIFEST_FILENAME}.")

    random_state = saved_metadata.get("random_state")
    return LoadedClustererArtifact(
        model=model,
        n_clusters=n_clusters,
        embedding_dimension=embedding_dimension,
        random_state=random_state if isinstance(random_state, int) else None,
    )


def verify_clusterer_bundle(model_dir: str | Path, *, expected_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    artifact_dir = Path(model_dir)
    manifest_path = artifact_dir / CLUSTERER_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clusterer manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Clusterer manifest must contain an object: {manifest_path}")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION or manifest.get("artifact_type") != "functionality_clusterer":
        raise ValueError(f"Unsupported clusterer manifest: {manifest_path}")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError(f"Clusterer manifest does not contain file checksums: {manifest_path}")
    expected_filenames = {CLUSTERER_JOBLIB_FILENAME, CLUSTERER_ONNX_FILENAME}
    if set(files) != expected_filenames:
        raise ValueError(f"Clusterer manifest must describe exactly {sorted(expected_filenames)}: {manifest_path}")
    for filename, record in files.items():
        artifact_path = artifact_dir / filename
        expected_hash = record.get("sha256") if isinstance(record, dict) else None
        if not artifact_path.is_file() or not isinstance(expected_hash, str) or _file_record(artifact_path)["sha256"] != expected_hash:
            raise ValueError(f"Clusterer artifact checksum verification failed: {artifact_path}")

    saved_metadata = manifest.get("metadata")
    if not isinstance(saved_metadata, dict):
        raise TypeError(f"Clusterer manifest does not contain metadata: {manifest_path}")
    for key, expected_value in (expected_metadata or {}).items():
        if expected_value is not None and saved_metadata.get(key) != expected_value:
            raise ValueError(f"Clusterer artifact metadata mismatch for {key!r}: expected {expected_value!r}, got {saved_metadata.get(key)!r}.")

    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("onnx_sklearn_prediction_parity") is not True:
        raise ValueError(f"Clusterer manifest does not record successful ONNX validation: {manifest_path}")
    return manifest


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


def _validate_onnx_parity(model: MiniBatchKMeans, onnx_bytes: bytes, samples: np.ndarray) -> dict[str, Any]:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = {output.name for output in session.get_outputs()}
    if "label" not in output_names:
        raise RuntimeError("Exported ONNX clusterer does not provide the expected 'label' output.")

    sklearn_labels = model.predict(samples)
    onnx_labels = session.run(["label"], {input_name: samples})[0].reshape(-1)
    if not np.array_equal(sklearn_labels, onnx_labels):
        mismatches = int(np.count_nonzero(sklearn_labels != onnx_labels))
        raise RuntimeError(f"ONNX clusterer parity validation failed for {mismatches}/{len(samples)} samples.")
    return {
        "onnx_sklearn_prediction_parity": True,
        "samples": len(samples),
        "input_name": input_name,
        "output_name": "label",
    }


def _atomic_joblib_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        joblib.dump(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
