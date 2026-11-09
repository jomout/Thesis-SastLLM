from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from argus.configs import get_logger, setup_logging


class ClusteringMode(str, Enum):
    search = "search"
    train = "train"
    test = "test"


class ClassificationMode(str, Enum):
    search = "search"
    train = "train"
    test = "test"


class RepositorySplit(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class ComputeDevice(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


class PlotFormat(str, Enum):
    png = "png"
    pdf = "pdf"
    svg = "svg"


logger = get_logger(__name__)

app = typer.Typer(
    name="argus",
    help="ARGUS — Automated Recognition and Guarding against Untrusted Source code.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def init():
    try:
        load_dotenv()
        setup_logging()
    except Exception as e:
        logger.exception("Failed to initialize ARGUS", error=str(e))
        raise RuntimeError(f"Failed to initialize project: {e}") from e


# --- Train ---
@app.command("generate_functionalities")
def run_train():
    """
    Run the functionality generation pipeline.

    This command uses the configured LLM to generate functionality descriptions
    from code snippets stored in the database.
    """
    from .pipelines import generate_functionalities

    logger.info("Starting generating functionalities pipeline.")
    generate_functionalities()


@app.command("generate_functionalities_batch_api")
def run_batch():
    """
    Run the functionality generation pipeline using OpenAI's Batch API.

    This command creates batch files for code snippets, uploads them to the API,
    and polls for results to process them into the database.
    """
    from .pipelines import generate_functionalities_batch_api

    logger.info("Starting generating functionalities with batch API pipeline.")
    generate_functionalities_batch_api()


@app.command("split")
def run_split():
    """
    Run the splitting pipeline.

    This command splits the dataset into training, validation, and test sets.
    """
    from .pipelines import split_dataset

    logger.info("Starting splitting dataset pipeline.")
    split_dataset()


@app.command("cluster")
def run_cluster(
    mode: Annotated[ClusteringMode, typer.Option("--mode", "-m")],
):
    """
    Run the clustering pipeline.

    This command clusters functionalities based on their similarity.
    """
    from .pipelines import cluster_functionalities

    logger.info("Starting clustering pipeline.")
    cluster_functionalities(mode.value)


# --- Classification ---
@app.command("classify")
def run_classify(
    mode: Annotated[ClassificationMode, typer.Option("--mode", "-m")],
):
    """
    Run the classification pipeline.

    This command classifies functionalities based on their similarity.
    """
    from .pipelines import classify_repositories

    logger.info("Starting classification pipeline.")
    classify_repositories(mode=mode.value)


@app.command("explain-shap")
def run_explain_shap(
    config_path: Annotated[Path, typer.Option(help="Classification YAML containing the test model contract.")] = Path("configs/classification.yaml"),
    output_dir: Annotated[Path, typer.Option(help="Directory for SHAP arrays, metadata, and plots.")] = Path("plots/shap"),
    explained_class: Annotated[str, typer.Option("--class", help="Configured label whose probability should be explained.")] = "malicious",
    split: Annotated[RepositorySplit, typer.Option(help="Repository split to explain.")] = RepositorySplit.test,
    background_split: Annotated[RepositorySplit, typer.Option(help="Repository split used as the SHAP background.")] = RepositorySplit.train,
    background_samples: Annotated[int, typer.Option(min=1, help="Maximum number of background repositories.")] = 200,
    explanation_samples: Annotated[int, typer.Option(min=1, help="Maximum number of repositories to explain.")] = 16,
    top_features: Annotated[int, typer.Option(min=1, help="Maximum number of features displayed in each plot.")] = 30,
    device: Annotated[ComputeDevice, typer.Option(help="Compute device; auto selects CUDA when available.")] = ComputeDevice.auto,
    seed: Annotated[int | None, typer.Option(help="SHAP sampling seed; defaults to classification.test.params.seed.")] = None,
) -> None:
    """Explain the configured repository classifier with SHAP."""
    from argus.utils.explain_shapley import ShapExplanationConfig, run_shap

    output_path = run_shap(
        ShapExplanationConfig(
            classification_config_path=config_path,
            output_dir=output_dir,
            explained_class=explained_class,
            split=split.value,
            background_split=background_split.value,
            background_samples=background_samples,
            explanation_samples=explanation_samples,
            top_features=top_features,
            device=device.value,
            seed=seed,
        )
    )
    typer.echo(f"SHAP artifacts written to {output_path}")


@app.command("confusion-matrix")
def run_confusion_matrix_command(
    config_path: Annotated[Path, typer.Option(help="Classification YAML containing the test model contract.")] = Path("configs/classification.yaml"),
    output_dir: Annotated[Path, typer.Option(help="Directory for confusion-matrix plots and metadata.")] = Path("plots/confusion_matrix"),
    split: Annotated[RepositorySplit, typer.Option(help="Repository split to evaluate.")] = RepositorySplit.test,
    log_scale: Annotated[bool, typer.Option("--log-scale/--linear-scale", help="Use logarithmic colors for the count matrix.")] = True,
    image_format: Annotated[PlotFormat, typer.Option("--format", help="Plot image format.")] = PlotFormat.png,
) -> None:
    """Evaluate the configured classifier and plot its confusion matrix."""
    from argus.utils.confusion_matrix import ConfusionMatrixConfig, run_confusion_matrix

    output_path = run_confusion_matrix(
        ConfusionMatrixConfig(
            classification_config_path=config_path,
            output_dir=output_dir,
            split=split.value,
            log_scale=log_scale,
            image_format=image_format.value,
        )
    )
    typer.echo(f"Confusion-matrix artifacts written to {output_path}")


# --- Pipelines ---
@app.command("train")
def run_train_pipeline():
    """
    Run the training pipeline.
    """
    from .pipelines import train_pipeline

    logger.info("Starting classification pipeline.")
    train_pipeline()


@app.command("test")
def run_test_pipeline():
    """
    Run the testing pipeline.
    """
    from .pipelines import test_pipeline

    logger.info("Starting testing pipeline.")
    test_pipeline()


# --- Setup ---
@app.command("load")
def run_load():
    """
    Run the loading pipeline.

    This command inserts File and Snippet records into the database from a local dataset path.
    """
    from .pipelines import load_dataset

    logger.info("Starting loading project.")
    load_dataset()


@app.command("download_benign_dataset")
def run_setup_eval():
    """
    Run the downloading benign dataset pipeline.

    This command downloads the CodeSearchNet dataset and organizes it into the
    local dataset directory for evaluation purposes.
    """
    from .download_dataset import download_benign_dataset

    logger.info("Starting downloading benign dataset (CSN).")
    download_benign_dataset()


@app.command("load_cache_functionalities")
def run_load_cache_functionalities(
    directory: str = typer.Argument(..., help="Path to directory containing cached functionalities"),
):
    """Load cached functionalities."""
    from .pipelines import load_functionalities_from_dir

    logger.info(f"Starting loading cached functionalities from {directory}")

    load_functionalities_from_dir(directory)


def main():
    app()
