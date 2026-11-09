from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import shap  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F

from argus.classification import ClassificationConfig, RepositoryClassificationService
from argus.configs import get_logger
from argus.ml import RepositoryDataModule
from argus.ml.models import RepositoryClassifierModule

logger = get_logger(__name__)

DatasetSplit = Literal["train", "val", "test"]
DeviceName = Literal["auto", "cpu", "cuda"]

warnings.filterwarnings(
    "ignore",
    message=r".*NumPy global RNG was seeded by calling `np\.random\.seed`.*",
    category=FutureWarning,
)

# Curated descriptions for clusters analyzed in the thesis. Unlisted cluster ids
# retain a stable generic name instead of coupling explanations to one experiment.
FEATURE_DESCRIPTIONS = {
    9053: "Memory Buffer Copy Operations",
    617: "Assembly Metadata and Version Configuration",
    9293: "HTML Table Row and Cell Creation",
    5345: "CPU Write Protection Toggling",
    539: "Tracing and Debug Logging",
    8253: "PHP Environment Fingerprinting and Configuration",
    7613: "Network Connection Reporting and Netstat Output",
    4569: "Energy Spectrum Computation and Binning",
    9934: "Bot Population Monitoring and Metrics",
    1887: "SOCKS Proxy Protocol Implementation",
    2455: "Socket Initialization and Hooking Framework",
    988: "Header Inclusion and Module Configuration",
    1235: "Socket Closure and Connection Teardown",
    8813: "INI Configuration File Parsing and Management",
    2746: ".NET Namespace Imports and Interop References",
    2662: "Nickname Generation and Management",
    8408: "YAML Loading and Validation",
    6868: "API Macros and COM Wrappers",
    511: "YAML Configuration Loading and Merging",
    4572: "Command and Script Execution Handling",
    9439: "Hooked File API Resolution and Forwarding",
    4094: "Scientific Computation and Math Routines",
    367: "JSON Success Response Generation",
    10453: "C Standard Header and Utility Includes",
    2076: "Tuple Construction and Manipulation Utilities",
    8910: "UI Form and Control Layout Configuration",
    8353: "HTML Directory Listing and Navigation",
    6894: "Bot Authentication and Lifecycle Management",
    666: "Embedded Resource Extraction and Deployment",
    10503: "JSON API Response Parsing",
    37: "Dictionary Sanitization and Cleaning",
    79: "Base64 Encoding and Decoding",
    138: "Header File Inclusion and Configuration",
    222: "Windows Registry Startup Persistence",
    387: "Queue Creation and Management",
    534: "Filesystem Path and Directory Configuration",
    552: "Dynamic Memory Allocation and Deallocation",
    579: "Execution Context and State Management",
    717: "Telnet Communication and Negotiation",
    977: "INI Configuration Parsing",
    1047: "Socket Creation and Connection",
    1081: "File List Management",
    1192: "Formatted Socket Message Transmission",
    1255: "Fade and Opacity Transitions",
    1341: "Function Prototype Declarations",
    1369: "Request Parameter Construction and Handling",
    1630: "Cache Invalidation and Removal",
    1874: "PHP Runtime Configuration and Detection",
    2035: "AWS and EC2 Resource Management",
    2440: "Resource Enumeration",
    2521: "Process Hiding and Concealed Execution",
    2524: "Network Packet Header Construction",
    2765: "Shell Command Output and Display",
    3089: "Java Class and Classpath Management",
    3304: "Cryptographic Salt Handling",
    3744: "File and Directory Hiding",
    3761: "Removable Drive Detection and Infection",
    3952: "Scrolling and Scrollbar Management",
    4306: "Generator Yield and Control Flow",
    4832: "IRC Status Reporting and Messaging",
    5101: "Connection Event Logging",
    5776: "Threaded Connection Handling",
    6305: "Process and Program Termination",
    6591: "Device Detection and Selection",
    6628: "Linked List Management",
    6713: "API Endpoint URL Construction",
    6721: "Windows Registry Operations",
    6958: "URL Construction and Retrieval",
    7440: "Color Brightness and Contrast Adjustment",
    7708: "File Upload and Payload Transfer",
    8051: "Newline-Terminated Buffered Input",
    8199: "TCP Socket and Port Connections",
    8211: "Networking Header Inclusion",
    8270: "Socket Binding and Listening",
    8328: "Text Banner Display",
    8496: "Data Structure and Metadata Definitions",
    8806: "User Interface Widget Management",
    9175: "Working Directory Management",
    9215: "Shell Command Execution",
    9595: "Executable Copying and Deployment",
    9717: "Windows API Header Inclusion",
    9771: "Encryption and Decryption",
    9851: "Session Closure and Disconnection",
    10050: "Function Argument and Parameter Handling",
    10063: "Directory Traversal and Entry Management",
    10198: "Assembly Metadata and Versioning",
}


@dataclass(frozen=True)
class ShapExplanationConfig:
    """Validated runtime settings for a repository-classifier SHAP run."""

    classification_config_path: Path = Path("configs/classification.yaml")
    output_dir: Path = Path("plots/shap")
    explained_class: str = "malicious"
    split: DatasetSplit = "test"
    background_split: DatasetSplit = "train"
    background_samples: int = 200
    explanation_samples: int = 16
    top_features: int = 30
    device: DeviceName = "auto"
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("split", "background_split"):
            if getattr(self, name) not in {"train", "val", "test"}:
                raise ValueError(f"{name} must be one of: train, val, test.")
        for name in ("background_samples", "explanation_samples", "top_features"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")


@dataclass(frozen=True)
class ExplanationBatch:
    repository_ids: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor


class ProbabilityModel(torch.nn.Module):
    """Expose classifier probabilities as SHAP explanation targets."""

    def __init__(self, model: RepositoryClassifierModule) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(features), dim=1)


def build_feature_names(feature_count: int) -> list[str]:
    """Build stable cluster-distribution feature labels."""
    return [f"[{cluster_id}] {FEATURE_DESCRIPTIONS.get(cluster_id, 'Unknown cluster')}" for cluster_id in range(feature_count)]


def class_shap_to_2d(
    shap_values: object,
    *,
    feature_shape: tuple[int, int],
    class_index: int,
) -> np.ndarray:
    """Normalize supported SHAP multi-output layouts to ``[samples, features]``."""
    if isinstance(shap_values, list):
        if not 0 <= class_index < len(shap_values):
            raise ValueError(f"SHAP output does not contain class index {class_index}.")
        values = np.asarray(shap_values[class_index])
    else:
        raw_values = np.asarray(shap_values)
        if raw_values.ndim == 2:
            values = raw_values
        elif raw_values.ndim == 3 and raw_values.shape[:2] == feature_shape:
            if not 0 <= class_index < raw_values.shape[2]:
                raise ValueError(f"SHAP output does not contain class index {class_index}.")
            values = raw_values[:, :, class_index]
        elif raw_values.ndim == 3 and raw_values.shape[1:] == feature_shape:
            if not 0 <= class_index < raw_values.shape[0]:
                raise ValueError(f"SHAP output does not contain class index {class_index}.")
            values = raw_values[class_index]
        else:
            raise RuntimeError(f"Unexpected SHAP output shape {raw_values.shape}; expected samples/features {feature_shape}.")

    if values.shape != feature_shape:
        if values.shape == feature_shape[::-1]:
            values = values.T
        else:
            raise RuntimeError(f"Unexpected class SHAP shape {values.shape}; expected {feature_shape}.")
    return values


def run_shap(settings: ShapExplanationConfig | None = None) -> Path:
    """Explain the configured classifier and return the output directory."""
    settings = settings or ShapExplanationConfig()
    classification_config = ClassificationConfig.from_yaml("test", settings.classification_config_path)
    if classification_config.load_model_dir is None:
        raise ValueError("classification.test.load_model_dir must be configured for SHAP explanations.")

    service = RepositoryClassificationService(config=classification_config)
    datamodule = service.datamodule()
    background = _collect_samples(datamodule, settings.background_split, settings.background_samples)
    explained = _collect_samples(datamodule, settings.split, settings.explanation_samples)
    _validate_gradient_features(background.features, split=settings.background_split)
    _validate_gradient_features(explained.features, split=settings.split)

    class_index = _class_index(service, settings.explained_class)
    device = _resolve_device(settings.device)
    model = service.load_model().to(device).eval()
    probability_model = ProbabilityModel(model).to(device)
    background_features = background.features.to(device)
    explained_features = explained.features.to(device)
    seed = classification_config.training.seed if settings.seed is None else settings.seed

    logger.info(
        "Starting classifier SHAP explanation",
        model=classification_config.model.name,
        model_dir=str(classification_config.load_model_dir),
        explained_class=settings.explained_class,
        class_index=class_index,
        split=settings.split,
        background_split=settings.background_split,
        background_samples=len(background.repository_ids),
        explanation_samples=len(explained.repository_ids),
        feature_shape=tuple(explained.features.shape),
        device=str(device),
        seed=seed,
    )

    explainer = shap.GradientExplainer(probability_model, background_features)
    shap_output = explainer.shap_values(explained_features, rseed=seed)
    feature_matrix = explained.features.detach().cpu().numpy()
    shap_values = class_shap_to_2d(
        shap_output,
        feature_shape=(feature_matrix.shape[0], feature_matrix.shape[1]),
        class_index=class_index,
    )
    with torch.inference_mode():
        base_value = float(probability_model(background_features)[:, class_index].mean().item())

    _write_outputs(
        settings=settings,
        classification_config=classification_config,
        background_sample_count=len(background.repository_ids),
        explained=explained,
        feature_matrix=feature_matrix,
        shap_values=shap_values,
        class_index=class_index,
        base_value=base_value,
        seed=seed,
    )
    logger.info("Completed classifier SHAP explanation", output_dir=str(settings.output_dir))
    return settings.output_dir


def _collect_samples(datamodule: RepositoryDataModule, split: DatasetSplit, limit: int) -> ExplanationBatch:
    loader = datamodule.dataloader_for_split(split)
    repository_ids: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_count = 0
    for batch_ids, batch_features, batch_labels in loader:
        remaining = limit - sample_count
        if remaining <= 0:
            break
        repository_ids.append(batch_ids[:remaining])
        features.append(batch_features[:remaining])
        labels.append(batch_labels[:remaining])
        sample_count += min(len(batch_ids), remaining)

    if not repository_ids:
        raise RuntimeError(f"The {split!r} split contains no repositories to explain.")
    if sample_count < limit:
        logger.warning("Requested more SHAP samples than the split contains", split=split, requested=limit, available=sample_count)
    return ExplanationBatch(
        repository_ids=torch.cat(repository_ids),
        features=torch.cat(features),
        labels=torch.cat(labels),
    )


def _validate_gradient_features(features: torch.Tensor, *, split: DatasetSplit) -> None:
    if features.ndim != 2:
        raise ValueError(f"SHAP GradientExplainer requires rank-2 cluster-distribution features; {split!r} has shape {tuple(features.shape)}.")
    if not torch.is_floating_point(features):
        raise ValueError(
            f"SHAP GradientExplainer supports the MLP and cluster-distribution Transformer encodings, but the configured model produced discrete ordered-token features for the {split!r} split."
        )


def _class_index(service: RepositoryClassificationService, explained_class: str) -> int:
    try:
        return service.labels.label_to_index[explained_class]
    except KeyError as error:
        available = ", ".join(sorted(service.labels.label_to_index))
        raise ValueError(f"Unknown explained class {explained_class!r}; expected one of: {available}.") from error


def _resolve_device(device_name: DeviceName) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SHAP, but CUDA is not available.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _write_outputs(
    *,
    settings: ShapExplanationConfig,
    classification_config: ClassificationConfig,
    background_sample_count: int,
    explained: ExplanationBatch,
    feature_matrix: np.ndarray,
    shap_values: np.ndarray,
    class_index: int,
    base_value: float,
    seed: int,
) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = build_feature_names(feature_matrix.shape[1])
    top_features = min(settings.top_features, feature_matrix.shape[1])
    ranked_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-top_features:][::-1]

    np.save(settings.output_dir / "shap_values.npy", shap_values)
    np.save(settings.output_dir / "inputs.npy", feature_matrix)
    np.save(settings.output_dir / "labels.npy", explained.labels.detach().cpu().numpy())
    np.save(settings.output_dir / "ids.npy", explained.repository_ids.detach().cpu().numpy())

    shap.summary_plot(
        shap_values,
        feature_matrix,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_features,
        show=False,
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(settings.output_dir / "summary_bar.png", bbox_inches="tight")
    plt.close()

    _plot_top_signed_features(
        shap_values=shap_values,
        feature_names=feature_names,
        ranked_indices=ranked_indices,
        output_path=settings.output_dir / "signed_mean.png",
        title=f"Top {top_features} features ({settings.explained_class})",
    )

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            data=feature_matrix[0],
            feature_names=feature_names,
            base_values=base_value,
        ),
        max_display=top_features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(settings.output_dir / "local_waterfall.png", bbox_inches="tight")
    plt.close()

    metadata = {
        "explained_class": settings.explained_class,
        "class_index": class_index,
        "model": classification_config.model.model_dump(mode="json"),
        "model_dir": str(classification_config.load_model_dir),
        "classification_config": str(settings.classification_config_path),
        "split": settings.split,
        "background_split": settings.background_split,
        "background_samples": background_sample_count,
        "explanation_samples": len(explained.repository_ids),
        "top_features": top_features,
        "base_value": base_value,
        "seed": seed,
    }
    (settings.output_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _plot_top_signed_features(
    *,
    shap_values: np.ndarray,
    feature_names: list[str],
    ranked_indices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    signed_means = shap_values.mean(axis=0)[ranked_indices]
    names = [feature_names[index] for index in ranked_indices]
    colors = ["red" if value >= 0 else "blue" for value in signed_means]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(ranked_indices)), signed_means, color=colors)
    plt.yticks(range(len(ranked_indices)), names)
    plt.axvline(0)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean SHAP value (signed)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
