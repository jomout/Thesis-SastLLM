from enum import Enum
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
