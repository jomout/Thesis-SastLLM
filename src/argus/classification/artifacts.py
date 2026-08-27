from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from argus.configs import get_logger
from argus.ml.models import RepositoryClassifierModule

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
CLASSIFIER_CHECKPOINT_FILENAME = "model.ckpt"
CLASSIFIER_WEIGHTS_FILENAME = "model.pt"
CLASSIFIER_ONNX_FILENAME = "model.onnx"


def export_classifier_bundle(
    *,
    model: RepositoryClassifierModule,
    model_dir: Path,
    validation_features: torch.Tensor,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    labels: dict[int, str],
    encoder_config: dict[str, Any],
    source: str,
    source_type: str,
) -> Path:
    if validation_features.ndim < 2 or len(validation_features) < 1:
        raise ValueError("ONNX validation features must contain at least one sample.")

    model = model.cpu().eval()
    features = validation_features.detach().cpu().contiguous()
    example_features = features[: min(2, len(features))]
    onnx_path = model_dir / CLASSIFIER_ONNX_FILENAME

    batch_dimension = torch.export.Dim("batch", min=1)
    onnx_program = torch.onnx.export(
        model,
        (example_features,),
        input_names=["features"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch_dimension}},
        dynamo=True,
        verify=False,
    )
    _atomic_save_onnx(onnx_program, onnx_path)
    parity = _validate_onnx_parity(model, onnx_path, features)

    input_encoding = model_config.get("input_encoding")
    if input_encoding is None:
        input_encoding = "ordered_tokens" if model_config.get("name") == "lstm" else "cluster_distribution"

    files = {
        path.name: _file_record(path)
        for path in (
            model_dir / CLASSIFIER_CHECKPOINT_FILENAME,
            model_dir / CLASSIFIER_WEIGHTS_FILENAME,
            onnx_path,
            model_dir / "config.json",
            model_dir / "meta.json",
        )
    }
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_type": "repository_classifier",
        "source": source,
        "source_type": source_type,
        "model": model_config,
        "training": training_config,
        "encoder": encoder_config,
        "labels": {str(index): label for index, label in sorted(labels.items())},
        "input": {
            "name": "features",
            "encoding": input_encoding,
            "dtype": str(features.numpy().dtype),
            "shape": [None, *features.shape[1:]],
            "batch_dimension_dynamic": True,
        },
        "output": {
            "name": "logits",
            "dtype": "float32",
            "shape": [None, len(labels)],
        },
        "versions": {
            "python": sys.version.split()[0],
            "lightning": version("lightning"),
            "numpy": version("numpy"),
            "onnx": version("onnx"),
            "onnxruntime": version("onnxruntime"),
            "onnxscript": version("onnxscript"),
            "torch": version("torch"),
        },
        "validation": parity,
        "files": files,
    }
    manifest_path = model_dir / "manifest.json"
    _atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    logger.info(
        "Saved classifier production bundle",
        checkpoint=str(model_dir / CLASSIFIER_CHECKPOINT_FILENAME),
        weights=str(model_dir / CLASSIFIER_WEIGHTS_FILENAME),
        onnx=str(onnx_path),
        manifest=str(manifest_path),
        parity_samples=parity["samples"],
        maximum_absolute_error=parity["maximum_absolute_error"],
    )
    return manifest_path


def verify_classifier_bundle(
    *,
    model_dir: Path,
    expected_model_config: dict[str, Any],
    expected_k: int,
    expected_labels: dict[int, str],
    expected_encoder_config: dict[str, Any],
) -> None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning("Classifier manifest not found; loading legacy checkpoint without integrity verification", model_dir=str(model_dir))
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported classifier manifest: {manifest_path}")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError(f"Classifier manifest does not contain file checksums: {manifest_path}")
    for filename, record in files.items():
        artifact_path = model_dir / filename
        expected_hash = record.get("sha256") if isinstance(record, dict) else None
        if not artifact_path.is_file() or not isinstance(expected_hash, str) or _file_record(artifact_path)["sha256"] != expected_hash:
            raise ValueError(f"Classifier artifact checksum verification failed: {artifact_path}")

    if manifest.get("model") != expected_model_config:
        raise ValueError("Configured classifier model does not match the saved artifact manifest.")
    saved_training = manifest.get("training")
    if not isinstance(saved_training, dict) or saved_training.get("k") != expected_k:
        raise ValueError(f"Configured cluster count k={expected_k} does not match the saved classifier artifact.")
    expected_label_payload = {str(index): label for index, label in sorted(expected_labels.items())}
    if manifest.get("labels") != expected_label_payload:
        raise ValueError("Configured classifier labels do not match the saved artifact manifest.")
    if manifest.get("encoder") != expected_encoder_config:
        raise ValueError("Configured classifier encoder does not match the saved artifact manifest.")


def _validate_onnx_parity(model: RepositoryClassifierModule, onnx_path: Path, features: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        expected = model(features).detach().cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"features": features.numpy()})[0]
    if expected.shape != actual.shape:
        raise RuntimeError(f"Classifier ONNX output shape mismatch: PyTorch {expected.shape}, ONNX {actual.shape}.")
    if not np.allclose(expected, actual, rtol=PARITY_RTOL, atol=PARITY_ATOL):
        maximum_error = float(np.max(np.abs(expected - actual)))
        raise RuntimeError(f"Classifier ONNX logits parity failed; maximum absolute error was {maximum_error}.")

    expected_labels = expected.argmax(axis=1)
    actual_labels = actual.argmax(axis=1)
    if not np.array_equal(expected_labels, actual_labels):
        mismatches = int(np.count_nonzero(expected_labels != actual_labels))
        raise RuntimeError(f"Classifier ONNX prediction parity failed for {mismatches}/{len(features)} samples.")

    absolute_error = np.abs(expected - actual)
    denominator = np.maximum(np.abs(expected), np.finfo(np.float32).eps)
    return {
        "onnx_pytorch_logits_close": True,
        "onnx_pytorch_predictions_equal": True,
        "samples": len(features),
        "absolute_tolerance": PARITY_ATOL,
        "relative_tolerance": PARITY_RTOL,
        "maximum_absolute_error": float(absolute_error.max(initial=0.0)),
        "maximum_relative_error": float((absolute_error / denominator).max(initial=0.0)),
    }


def _atomic_save_onnx(program: torch.onnx.ONNXProgram, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        program.save(temporary, external_data=False)
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
