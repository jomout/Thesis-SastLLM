from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sastllm.configs import get_logger
from sastllm.ml import CodeDataModule, CodeModel
from sastllm.ml.repository_classifier import ClassifierConfig, RepositoryClassifier
from sastllm.utils.repository_encoder import BINARY_LABEL_MAP

logger = get_logger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*NumPy global RNG was seeded by calling `np\.random\.seed`.*",
    category=FutureWarning,
)

# -----------------------------
# Feature descriptions - Enter Cluster titles here
# -----------------------------
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
}


def build_feature_names(n_features: int) -> list[str]:
    names = []
    for i in range(n_features):
        desc = FEATURE_DESCRIPTIONS.get(i)
        names.append(f"[{i}] {desc}" if desc else f"[{i}] Unknown Feature")
    return names


# -----------------------------
# Plotting
# -----------------------------
def plot_top_signed_features(
    sv: np.ndarray,
    out_path: Path,
    top_k: int,
    feature_names: list[str],
    title: str,
):
    mean_sv = sv.mean(axis=0)
    idx = np.argsort(np.abs(mean_sv))[-top_k:][::-1]

    vals = mean_sv[idx]
    names = [feature_names[i] for i in idx]
    colors = ["red" if v >= 0 else "blue" for v in vals]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), vals, color=colors)
    plt.yticks(range(top_k), names)
    plt.axvline(0)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean SHAP Value (Signed)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -----------------------------
# SHAP wrapper
# -----------------------------
class ShapModelWrapper(torch.nn.Module):
    def __init__(self, model: CodeModel):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x), dim=1)


# -----------------------------
# Helpers
# -----------------------------
def get_background_tensor(dm: CodeDataModule, n: int, device):
    dm.setup()
    xs = []
    for _, X, _ in dm.train_dataloader():
        xs.append(X)
        if sum(x.size(0) for x in xs) >= n:
            break
    return torch.cat(xs, dim=0)[:n].to(device)


def get_explain_batch(dm: CodeDataModule, split: Literal["train", "test"], n: int):
    dm.setup()
    ds = dm.train_ds if split == "train" else dm.test_ds
    ids, X, y = next(iter(DataLoader(ds, batch_size=n, shuffle=True)))
    return ids, X, y


def class_shap_to_2d(shap_values, X_np, class_idx):
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[class_idx])
        if sv.shape == (X_np.shape[1], X_np.shape[0]):
            sv = sv.T
        return sv
    sv = np.asarray(shap_values)
    if sv.ndim == 3 and sv.shape[:2] == X_np.shape:
        return sv[:, :, class_idx]
    if sv.ndim == 3 and sv.shape[1:] == X_np.shape:
        return sv[class_idx]
    raise RuntimeError(f"Unexpected SHAP shape {sv.shape}")


@torch.no_grad()
def compute_base_value(model, background, class_idx):
    return model(background)[:, class_idx].mean().item()


# -----------------------------
# Main
# -----------------------------
def run_shap(
    model_dir: str | Path,
    out_dir: str | Path,
    n_background=200,
    n_explain=16,
    explain_class: Literal["benign", "malicious"] = "malicious",
    split: Literal["train", "test"] = "test",
    top_k=30,
):
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ClassifierConfig(
        batch_size=16,
        epochs=30,
        lr=0.0005,
        weight_decay=1e-3,
        l1_lambda=1e-3,
        seed=42,
        k=10661,
    )

    clf = RepositoryClassifier(config=cfg)
    dm = clf._setup_datamodule(full_dataset=clf.full_dataset, validation_size=0.1, batch_size=cfg.batch_size)

    labels = clf.full_dataset.y.numpy(force=True)
    class_counts = dict(zip(*np.unique(labels, return_counts=True)))

    model = CodeModel.load_from_checkpoint(
        model_dir / "best.ckpt",
        class_counts=class_counts,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    wrapped = ShapModelWrapper(model).to(device)

    background = get_background_tensor(dm, n_background, device)
    explainer = shap.GradientExplainer(wrapped, background)

    ids, X, y = get_explain_batch(dm, split, n_explain)
    X = X.to(device)

    shap_values = explainer.shap_values(X)
    class_idx = {v: k for k, v in BINARY_LABEL_MAP.items()}[explain_class]

    X_np = X.cpu().numpy()
    sv = class_shap_to_2d(shap_values, X_np, class_idx)

    feature_names = build_feature_names(X_np.shape[1])

    np.save(out_dir / "shap_values.npy", sv)
    np.save(out_dir / "inputs.npy", X_np)
    np.save(out_dir / "labels.npy", y.numpy())
    np.save(out_dir / "ids.npy", ids.numpy())

    mean_sv = sv.mean(axis=0)
    top_idx = np.argsort(np.abs(mean_sv))[-top_k:][::-1]

    shap.summary_plot(
        sv[:, top_idx],
        X_np[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        plot_type="bar",
        show=False,
    )
    plt.xlabel("Mean absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", bbox_inches="tight")
    plt.close()

    plot_top_signed_features(
        sv[:, top_idx],
        out_dir / "signed_mean.png",
        top_k,
        [feature_names[i] for i in top_idx],
        f"Top {top_k} Features ({explain_class})",
    )

    base = compute_base_value(wrapped, background, class_idx)
    order = np.argsort(np.abs(sv[0, top_idx]))[::-1]

    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0, top_idx][order],
            data=X_np[0, top_idx][order],
            feature_names=[feature_names[top_idx[i]] for i in order],
            base_values=base,
        ),
        show=False,
    )
    plt.savefig(out_dir / "local_waterfall.png", bbox_inches="tight")
    plt.close()

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "explained_class": explain_class,
                "class_index": class_idx,
                "top_k": top_k,
                "checkpoint": str(model_dir),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run_shap(
        model_dir="models/classification/trained_models/binary_classifier_model",
        out_dir="./plots/shap",
        explain_class="malicious",
        split="test",
    )
