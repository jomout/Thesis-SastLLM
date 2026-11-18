from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxArgMin, OnnxCast, OnnxGemm, OnnxMax, OnnxReduceSumSquareApi18, OnnxSqrt  # type: ignore[import-untyped]
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType  # type: ignore[import-untyped]
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from argus.configs import get_logger

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
CLUSTERER_JOBLIB_FILENAME = "model.joblib"
CLUSTERER_ONNX_FILENAME = "model.onnx"
CLUSTERER_MANIFEST_FILENAME = "manifest.json"
CLUSTERER_QUALITY_REPORT_FILENAME = "quality.json"
ONNX_PARITY_SAMPLE_SIZE = 256
ONNX_ASSIGNMENT_DISTANCE_ATOL = 1e-6
ONNX_ASSIGNMENT_DISTANCE_RTOL = 1e-5
# For L2-normalized embeddings, the terms in sklearn's float32 squared-distance calculation are of order one.
ONNX_ASSIGNMENT_SQUARED_ROUNDOFF_ATOL = float(4 * np.finfo(np.float32).eps)


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
    validation_samples: np.ndarray,
) -> Path:
    """Validate ONNX predictions and write a clusterer bundle with its manifest last."""
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    embedding_dimension = centers.shape[1]
    samples = _select_validation_samples(validation_samples, embedding_dimension=embedding_dimension)
    onnx_bytes = _export_onnx_model(centers)
    validation = _validate_onnx_parity(model, onnx_bytes, samples)

    model_fields = {
        "n_clusters": int(model.n_clusters),
        "embedding_dimension": embedding_dimension,
        "metadata": {
            **(metadata or {}),
            "clusterer": model.__class__.__name__,
            "random_state": random_state,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
        },
    }
    manifest = {
        **model_fields,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_type": "functionality_clusterer",
        "algorithm": "sklearn.cluster.MiniBatchKMeans",
        "input": {
            "name": validation["input_name"],
            "dtype": "float32",
            "shape": [None, embedding_dimension],
            "normalization": "l2",
        },
        "validation": validation,
    }
    artifact_dir = Path(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / CLUSTERER_MANIFEST_FILENAME
    with TemporaryDirectory(dir=artifact_dir) as temporary:
        staging_dir = Path(temporary)
        joblib_path = staging_dir / CLUSTERER_JOBLIB_FILENAME
        onnx_path = staging_dir / CLUSTERER_ONNX_FILENAME
        joblib.dump({"model": model, **model_fields}, joblib_path)
        onnx_path.write_bytes(onnx_bytes)
        manifest["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in (joblib_path, onnx_path)}
        staged_manifest = staging_dir / CLUSTERER_MANIFEST_FILENAME
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        for path in (joblib_path, onnx_path, staged_manifest):
            path.replace(artifact_dir / path.name)

    logger.info(
        "Saved clusterer artifact bundle",
        model_dir=str(artifact_dir),
        parity_samples=validation["samples"],
        label_mismatches=validation["label_mismatches"],
        distance_equivalent_mismatches=validation["distance_equivalent_mismatches"],
        float32_roundoff_mismatches=validation["float32_roundoff_mismatches"],
    )
    return manifest_path


def load_clusterer_bundle(
    model_dir: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> LoadedClustererArtifact:
    """Restore a trusted joblib model after verifying the bundle."""
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
    """Check the manifest and file digests without loading the model."""
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
    expected_filenames = (CLUSTERER_JOBLIB_FILENAME, CLUSTERER_ONNX_FILENAME)
    if files.keys() != set(expected_filenames):
        raise ValueError(f"Clusterer manifest must describe exactly {expected_filenames}: {manifest_path}")
    for filename in expected_filenames:
        record = files[filename]
        artifact_path = artifact_dir / filename
        expected_hash = record.get("sha256") if isinstance(record, dict) else None
        if not artifact_path.is_file() or not isinstance(expected_hash, str) or _sha256(artifact_path) != expected_hash:
            raise ValueError(f"Clusterer artifact checksum verification failed: {artifact_path}")

    saved_metadata = manifest.get("metadata")
    if not isinstance(saved_metadata, dict):
        raise TypeError(f"Clusterer manifest does not contain metadata: {manifest_path}")
    for key, expected_value in (expected_metadata or {}).items():
        if expected_value is not None and saved_metadata.get(key) != expected_value:
            raise ValueError(f"Clusterer artifact metadata mismatch for {key!r}: expected {expected_value!r}, got {saved_metadata.get(key)!r}.")

    validation = manifest.get("validation")
    # The version 1 selected model records only exact prediction parity.
    if not isinstance(validation, dict) or validation.get("validation_passed", validation.get("onnx_sklearn_prediction_parity")) is not True:
        raise ValueError(f"Clusterer manifest does not record successful ONNX validation: {manifest_path}")
    return manifest


def training_artifact_dir(root: Path, *, k: int) -> Path:
    """Return the output directory for a training run."""
    return root / f"clusterer_{k}_{_timestamp()}"


def search_artifact_dir(root: Path, *, n: int, k: int) -> Path:
    """Return the output directory for a selected clusterer."""
    timestamp = _timestamp()
    return root / f"clusterers_{n}_{timestamp}" / f"clusterer_{n}_{k}_{timestamp}"


def _export_onnx_model(centers: np.ndarray) -> bytes:
    """Export nearest-centroid inference with a float32 interface and float64 distance arithmetic."""
    opset = 18
    centers = centers.astype(np.float64)
    n_clusters, embedding_dimension = centers.shape
    vectors = OnnxCast("X", to=np.float64, op_version=opset)
    ranking = OnnxGemm(vectors, centers, np.sum(centers**2, axis=1), alpha=-2.0, transB=1, op_version=opset)
    labels = OnnxArgMin(ranking, axis=1, keepdims=0, output_names=["label"], op_version=opset)

    # The sample norm is constant across centroids, so add it only for the distance output.
    sample_norms_squared = OnnxReduceSumSquareApi18(vectors, axes=[1], keepdims=1, op_version=opset)
    squared_distances = OnnxAdd(ranking, sample_norms_squared, op_version=opset)
    nonnegative_distances = OnnxMax(squared_distances, np.array(0, dtype=np.float64), op_version=opset)
    scores = OnnxCast(OnnxSqrt(nonnegative_distances, op_version=opset), to=np.float32, output_names=["scores"], op_version=opset)
    onnx_model = labels.to_onnx(
        inputs={"X": FloatTensorType([None, embedding_dimension])},
        outputs={"label": Int64TensorType([None]), "scores": FloatTensorType([None, n_clusters])},
        other_outputs=[scores],
        target_opset=opset,
    )
    return onnx_model.SerializeToString()


def _validate_onnx_parity(model: MiniBatchKMeans, onnx_bytes: bytes, samples: np.ndarray) -> dict[str, Any]:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sklearn_labels = model.predict(samples)
    onnx_labels = session.run(["label"], {input_name: samples})[0]
    if onnx_labels.shape != sklearn_labels.shape or onnx_labels.dtype != np.int64:
        raise RuntimeError("ONNX clusterer must return one int64 cluster label per input row.")
    if np.any((onnx_labels < 0) | (onnx_labels >= model.n_clusters)):
        raise RuntimeError("ONNX clusterer produced an out-of-range cluster label.")

    mismatch_mask = sklearn_labels != onnx_labels
    mismatch_samples = samples[mismatch_mask].astype(np.float64)
    mismatches = len(mismatch_samples)
    centers = model.cluster_centers_
    sklearn_distances = np.linalg.norm(mismatch_samples - centers[sklearn_labels[mismatch_mask]], axis=1)
    onnx_distances = np.linalg.norm(mismatch_samples - centers[onnx_labels[mismatch_mask]], axis=1)
    distance_deltas = np.abs(onnx_distances - sklearn_distances)
    relative_deltas = distance_deltas / np.maximum(sklearn_distances, ONNX_ASSIGNMENT_DISTANCE_ATOL)
    maximum_absolute_distance_delta = float(distance_deltas.max(initial=0.0))
    maximum_relative_distance_delta = float(relative_deltas.max(initial=0.0))

    equivalent = np.isclose(onnx_distances, sklearn_distances, rtol=ONNX_ASSIGNMENT_DISTANCE_RTOL, atol=ONNX_ASSIGNMENT_DISTANCE_ATOL)
    # Float32 predict can pick a farther centroid when the squared-distance gap rounds away.
    squared_improvement = sklearn_distances**2 - onnx_distances**2
    roundoff = (squared_improvement > 0) & (squared_improvement <= ONNX_ASSIGNMENT_SQUARED_ROUNDOFF_ATOL)
    non_equivalent = np.count_nonzero(~(equivalent | roundoff))
    if non_equivalent:
        raise RuntimeError(
            "ONNX clusterer assignment validation failed: "
            f"{mismatches}/{len(samples)} label mismatches, including {non_equivalent} non-equivalent nearest-centroid assignments; "
            f"maximum absolute distance delta={maximum_absolute_distance_delta:.3e}, maximum relative distance delta={maximum_relative_distance_delta:.3e}."
        )

    distance_equivalent_mismatches = int(np.count_nonzero(equivalent))

    return {
        "validation_passed": True,
        "onnx_sklearn_prediction_parity": mismatches == 0,
        "onnx_sklearn_assignment_distance_equivalence": distance_equivalent_mismatches == mismatches,
        "label_mismatches": mismatches,
        "distance_equivalent_mismatches": distance_equivalent_mismatches,
        "float32_roundoff_mismatches": mismatches - distance_equivalent_mismatches,
        "maximum_absolute_assignment_distance_delta": maximum_absolute_distance_delta,
        "maximum_relative_assignment_distance_delta": maximum_relative_distance_delta,
        "assignment_distance_absolute_tolerance": ONNX_ASSIGNMENT_DISTANCE_ATOL,
        "assignment_distance_relative_tolerance": ONNX_ASSIGNMENT_DISTANCE_RTOL,
        "assignment_squared_distance_roundoff_tolerance": ONNX_ASSIGNMENT_SQUARED_ROUNDOFF_ATOL,
        "samples": len(samples),
        "sample_source": "evaluation_reservoir",
        "input_name": input_name,
        "output_name": "label",
    }


def _select_validation_samples(samples: np.ndarray, *, embedding_dimension: int) -> np.ndarray:
    """Select deterministic representative rows for ONNX assignment validation."""
    matrix = np.asarray(samples, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("ONNX validation samples must be a non-empty two-dimensional array.")
    if matrix.shape[1] != embedding_dimension:
        raise ValueError(f"ONNX validation samples have dimension {matrix.shape[1]}; expected {embedding_dimension}.")
    if not np.isfinite(matrix).all():
        raise ValueError("ONNX validation samples contain non-finite values.")

    sample_count = min(ONNX_PARITY_SAMPLE_SIZE, len(matrix))
    indexes = np.linspace(0, len(matrix) - 1, num=sample_count, dtype=np.int64)
    return np.ascontiguousarray(matrix[indexes])


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
