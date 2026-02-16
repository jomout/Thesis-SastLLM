from enum import Enum

import typer
from dotenv import load_dotenv

from sastllm.configs import get_logger, setup_logging

from .download_dataset import download_benign_dataset
from .pipelines import (
    classify_repositories,
    cluster_functionalities,
    generate_functionalities,
    generate_functionalities_batch_api,
    load_dataset,
    load_functionalities_from_dir,
    split_dataset,
    test_pipeline,
    train_pipeline,
)


class ClusteringMode(str, Enum):
    search = "search"
    train = "train"
    test = "test"


class ClassificationMode(str, Enum):
    train = "train"
    test = "test"


logger = get_logger(__name__)

app = typer.Typer(help="SAST-LLM CLI")


@app.callback()
def init():
    try:
        load_dotenv()
        setup_logging()
    except Exception as e:
        logger.error("Failed to initialize project: %s", e)
        raise RuntimeError(f"Failed to initialize project: {e}") from e


# --- Train ---
@app.command("generate_functionalities")
def run_train():
    """
    Run the functionality generation pipeline.

    This command uses the configured LLM to generate functionality descriptions
    from code snippets stored in the database.
    """
    logger.info("Starting generating functionalities pipeline.")

    generate_functionalities()


@app.command("generate_functionalities_batch_api")
def run_batch():
    """
    Run the functionality generation pipeline using OpenAI's Batch API.

    This command creates batch files for code snippets, uploads them to the API,
    and polls for results to process them into the database.
    """
    logger.info("Starting generating functionalities with batch API pipeline.")

    generate_functionalities_batch_api()


@app.command("split")
def run_split():
    """
    Run the splitting pipeline.

    This command splits the dataset into training, validation, and test sets.
    """
    logger.info("Starting splitting dataset pipeline.")

    # Split Dataset
    split_dataset()


@app.command("cluster")
def run_cluster(
    mode: ClusteringMode = typer.Option(..., "--mode", "-m"),
):
    """
    Run the clustering pipeline.

    This command clusters functionalities based on their similarity.
    """
    logger.info("Starting clustering pipeline.")
    cluster_functionalities(mode.value)


# --- Classification ---
@app.command("classify")
def run_classify(
    mode: ClassificationMode = typer.Option(..., "--mode", "-m"),
):
    """
    Run the classification pipeline.

    This command classifies functionalities based on their similarity.
    """
    logger.info("Starting classification pipeline.")

    # Classify Repositories
    classify_repositories(mode=mode.value)


# --- Pipelines ---
@app.command("train")
def run_train_pipeline():
    """
    Run the training pipeline.
    """
    logger.info("Starting classification pipeline.")

    train_pipeline()


@app.command("test")
def run_test_pipeline():
    """
    Run the testing pipeline.
    """
    logger.info("Starting testing pipeline.")

    test_pipeline()


# --- Setup ---
@app.command("load")
def run_load():
    """
    Run the loading pipeline.

    This command inserts File and Snippet records into the database from a local dataset path.
    """
    logger.info("Starting loading project.")

    load_dataset()


@app.command("download_benign_dataset")
def run_setup_eval():
    """
    Run the downloading benign dataset pipeline.

    This command downloads the CodeSearchNet dataset and organizes it into the
    local dataset directory for evaluation purposes.
    """
    logger.info("Starting downloading benign dataset (CSN).")

    download_benign_dataset()


@app.command("load_cache_functionalities")
def run_load_cache_functionalities(
    directory: str = typer.Argument(..., help="Path to directory containing cached functionalities"),
):
    """Load cached functionalities."""
    logger.info(f"Starting loading cached functionalities from {directory}")

    load_functionalities_from_dir(directory)


def main():
    app()
