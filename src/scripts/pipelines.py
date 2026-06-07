import glob
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from sastllm.classification import ClassificationConfig, RepositoryClassificationService
from sastllm.clustering import FunctionalityClusteringService
from sastllm.configs import get_logger
from sastllm.db import FunctionalityManager, SnippetManager
from sastllm.dtos import CreateFunctionalityDto
from sastllm.dtos.update_dtos import UpdateSnippetDto
from sastllm.processors import (
    BatchFileProcessor,
    BatchFilesGenerator,
    CodeProcessor,
    SnippetProcessor,
)
from sastllm.utils.dataset_splitter import DatasetSplitter
from sastllm.utils.observability import log_duration

from .utils import get_model, load_yaml

logger = get_logger(__name__)


def load_dataset() -> None:
    """
    Insert File and Snippet records into the database from a local dataset path.
    """
    try:
        config = load_yaml()

        # Load dataset
        sastllm_dataset: Optional[str] = config.get("paths", {}).get("dataset")

        if not sastllm_dataset:
            msg = "`paths.dataset` is not defined in the YAML config."
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Loading dataset", dataset_path=sastllm_dataset)

        loader = CodeProcessor(
            root_path=sastllm_dataset,
        )
        with log_duration(logger, "dataset_load", dataset_path=sastllm_dataset):
            loader.run()

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def split_dataset() -> None:
    """
    Embed all functionalities and split the dataset into training, validation, and test sets.
    """
    logger.info("Splitting dataset into training and test sets.")

    config = load_yaml("configs/split.yaml")

    model_name = config["split"]["model_name"]
    train_size = config["split"]["training"]["ratio"]
    test_size = config["split"]["testing"]["ratio"]

    logger.info("Split configuration loaded", model_name=model_name, train_size=train_size, test_size=test_size)
    database_splitter = DatasetSplitter(model_name=model_name)
    with log_duration(logger, "dataset_split", model_name=model_name, train_size=train_size, test_size=test_size):
        # database_splitter.embed_all_repositories()
        database_splitter.split_repositories(train_size=train_size, test_size=test_size)


def cluster_functionalities(mode: Literal["search", "train", "test"]) -> None:
    """
    Cluster functionalities and assign cluster IDs using vector embeddings.
    """
    logger.info("Clustering functionalities according to functionality tags")

    config = load_yaml("configs/split.yaml")

    model_name = config["split"]["model_name"]
    collection_name = model_name.replace("/", "_")

    logger.info("Clustering configuration loaded", mode=mode, collection_name=collection_name)
    processor = FunctionalityClusteringService(collection_name=collection_name)

    try:
        with log_duration(logger, "functionality_clustering", mode=mode, collection_name=collection_name):
            processor.run(mode=mode)
    except Exception as e:
        logger.error(f"Functionality clustering failed: {e}")
        raise RuntimeError(f"Functionality clustering failed: {e}") from e


def generate_functionalities_batch_api() -> None:
    model = "gpt-5-mini"
    batch_files_dir = Path("api_batches_extra")
    batch_files_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating batch API files", model=model, batch_files_dir=str(batch_files_dir))
    gen = BatchFilesGenerator(model=model)
    with log_duration(logger, "generate_functionality_batch_files", model=model):
        gen.create_api_batches(output_dir=batch_files_dir)

    batch_results_dir = Path("batch_results_extra")
    batch_results_dir.mkdir(parents=True, exist_ok=True)

    processor = BatchFileProcessor(
        batch_files_dir=batch_files_dir,
        output_dir=batch_results_dir,
        poll_interval=60,  # check every 60 seconds
        max_wait_hours=30,
    )
    with log_duration(logger, "process_functionality_batch_files", batch_files_dir=str(batch_files_dir), output_dir=str(batch_results_dir)):
        processor.process_all()


def generate_functionalities() -> None:
    """
    Generate functionality descriptions from code snippets using the configured LLM.
    """
    logger.info("Generating functionalities from snippets")

    llm = get_model(processor="snippet_processor")

    processor = SnippetProcessor(
        llm=llm,
        batch_size=50,
        sleep_interval=5,
    )

    try:
        with log_duration(logger, "functionality_generation", batch_size=processor.batch_size, sleep_interval=processor.sleep_interval):
            processor.run()
    except Exception as e:
        logger.error(f"Functionality generation failed: {e}")
        raise RuntimeError(f"Functionality generation failed: {e}") from e


def classify_repositories(mode: Literal["search", "train", "test"]) -> None:
    """
    Classify repositories by their clusters using ML.
    """
    logger.info("Classifying repositories by their clusters using ML.")

    config = ClassificationConfig.from_yaml(mode=mode)
    logger.info(
        "Classification configuration loaded",
        mode=mode,
        model=config.model.name,
        k=config.training.k,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
    )
    service = RepositoryClassificationService(config=config, build_bundle=mode != "search")

    try:
        if mode == "search":
            with log_duration(logger, "repository_classification_search"):
                service.search()
        elif mode == "train":
            with log_duration(logger, "repository_classification_train"):
                model_dir = service.fit()
            print(f"Model saved to: {model_dir}")
            with log_duration(logger, "repository_classification_evaluate", split="train", model_dir=str(model_dir)):
                service.evaluate(split="train", model_dir=model_dir)
        else:
            with log_duration(logger, "repository_classification_test"):
                service.test()
            with log_duration(logger, "repository_classification_evaluate", split="test"):
                service.evaluate(split="test")

    except Exception as e:
        logger.error(f"Repository classification failed: {e}")
        raise RuntimeError(f"Repository classification failed: {e}") from e


def train_pipeline() -> None:
    """Run the full training pipeline end-to-end."""
    cluster_functionalities(mode="train")
    classify_repositories(mode="train")


def test_pipeline() -> None:
    """Run the full testing pipeline end-to-end."""
    cluster_functionalities(mode="test")
    classify_repositories(mode="test")


def load_functionalities_from_dir(directory: str) -> None:
    """Load all JSON files from a directory into Pydantic DTOs."""
    all_functionalities: List[CreateFunctionalityDto] = []
    all_snippets: List[UpdateSnippetDto] = []
    for file_path in glob.glob(f"{directory}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            filename = os.path.basename(file_path)
            snippet_id = int(filename.split("_")[1].split(".")[0])

            # Convert each dict into a CreateFunctionalityDto
            data = json.load(f)
            for item in data:
                all_functionalities.append(CreateFunctionalityDto(**item))
            all_snippets.append(UpdateSnippetDto(snippet_id=snippet_id, processed=True))

    functionality_db = FunctionalityManager()
    logger.info("Loading cached functionalities", directory=directory, files=len(glob.glob(f"{directory}/*.json")))
    with log_duration(
        logger,
        "load_cached_functionalities",
        directory=directory,
        functionalities=len(all_functionalities),
        snippets=len(all_snippets),
    ):
        functionality_db.add_bulk_functionalities(all_functionalities)

        snippet_db = SnippetManager()
        snippet_db.update_bulk_snippets(all_snippets)

    logger.info(f"Inserted {len(all_functionalities)} functionalities from {directory}")
