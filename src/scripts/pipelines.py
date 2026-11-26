import glob
import json
import os
from pathlib import Path
from typing import Literal

from argus.classification import ClassificationConfig, RepositoryClassificationService
from argus.clustering import FunctionalityClusteringService
from argus.configs import get_logger
from argus.db import FunctionalityManager, SnippetManager
from argus.dtos import CreateFunctionalityDto
from argus.dtos.update_dtos import UpdateSnippetDto
from argus.processors import (
    BatchFileProcessor,
    BatchFilesGenerator,
    BatchFunctionalityConfig,
    BatchFunctionalityResultImporter,
    CodeProcessor,
    SnippetProcessor,
)
from argus.utils.dataset_splitter import DatasetSplitter
from argus.utils.observability import log_duration

from .utils import get_model, load_yaml

logger = get_logger(__name__)


def load_dataset(dataset_path: str | Path | None = None) -> None:
    """
    Insert File and Snippet records into the database from a local dataset path.
    """
    try:
        config = load_yaml()

        # Load dataset
        configured_dataset = config.get("paths", {}).get("dataset")
        dataset = dataset_path if dataset_path is not None else configured_dataset

        if not isinstance(dataset, (str, Path)) or not str(dataset).strip():
            msg = "`paths.dataset` is not defined in the YAML config."
            logger.exception(msg)
            raise ValueError(msg)

        resolved_dataset = Path(dataset).expanduser().resolve()
        if not resolved_dataset.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {resolved_dataset}")

        logger.info("Loading dataset", dataset_path=str(resolved_dataset))

        loader = CodeProcessor(
            root_path=str(resolved_dataset),
        )
        with log_duration(logger, "dataset_load", dataset_path=str(resolved_dataset)):
            loader.run()

    except Exception as e:
        logger.exception("Dataset loading failed", error=str(e))
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def embed_functionalities() -> None:
    """Create missing functionality-tag embeddings in Qdrant."""
    config = load_yaml("configs/split.yaml")
    model_name = config.get("split", {}).get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("`split.model_name` must be a non-empty string in configs/split.yaml.")

    collection_name = model_name.replace("/", "_")
    logger.info(
        "Embedding functionality tags",
        model_name=model_name,
        collection_name=collection_name,
    )
    try:
        splitter = DatasetSplitter(model_name=model_name)
        with log_duration(
            logger,
            "functionality_embedding",
            model_name=model_name,
            collection_name=collection_name,
        ):
            splitter.embed_all_repositories()
    except Exception as error:
        logger.exception(
            "Functionality embedding failed",
            model_name=model_name,
            collection_name=collection_name,
            error=str(error),
        )
        raise RuntimeError(f"Functionality embedding failed: {error}") from error


def split_dataset() -> None:
    """Assign repositories and existing Qdrant embeddings to train/test splits."""
    logger.info("Splitting dataset into training and test sets.")

    config = load_yaml("configs/split.yaml")

    model_name = config["split"]["model_name"]
    train_size = config["split"]["training"]["ratio"]
    test_size = config["split"]["testing"]["ratio"]

    logger.info("Split configuration loaded", model_name=model_name, train_size=train_size, test_size=test_size)
    database_splitter = DatasetSplitter(model_name=model_name)
    with log_duration(logger, "dataset_split", model_name=model_name, train_size=train_size, test_size=test_size):
        database_splitter.split_repositories(train_size=train_size, test_size=test_size)


def cluster_functionalities(mode: Literal["search", "train", "test"], *, config_path: Path) -> None:
    """
    Cluster functionalities and assign cluster IDs using vector embeddings.
    """
    logger.info("Clustering functionalities according to functionality tags")

    config = load_yaml("configs/split.yaml")

    model_name = config["split"]["model_name"]
    collection_name = model_name.replace("/", "_")

    logger.info("Starting clustering stage", mode=mode, collection_name=collection_name, config_path=str(config_path))
    processor = FunctionalityClusteringService(collection_name=collection_name, embedding_model_name=model_name, config_path=config_path)

    try:
        with log_duration(logger, "functionality_clustering", mode=mode, collection_name=collection_name):
            processor.run(mode=mode)
    except Exception as e:
        logger.exception("Functionality clustering failed", mode=mode, error=str(e))
        raise RuntimeError(f"Functionality clustering failed: {e}") from e


def generate_functionalities_batch_api() -> None:
    raw_config = load_yaml("configs/llms.yaml").get("batch_functionality")
    config = BatchFunctionalityConfig.model_validate(raw_config)
    processor = BatchFileProcessor(
        batch_files_dir=config.batch_files_dir,
        output_dir=config.results_dir,
        poll_interval=config.poll_interval_seconds,
        max_wait_hours=config.max_wait_hours,
    )

    logger.info(
        "Generating Batch API files",
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        batch_files_dir=str(config.batch_files_dir),
    )
    generator = BatchFilesGenerator(
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
        snippet_batch_size=config.snippet_batch_size,
        api_batch_size=config.requests_per_file,
        max_file_size_bytes=config.max_file_size_bytes,
    )
    with log_duration(logger, "generate_functionality_batch_files", model=config.model):
        created_files = generator.create_api_batches(output_dir=config.batch_files_dir)
    if not created_files:
        logger.info("No unprocessed snippets were available for Batch API generation")
        return

    with log_duration(
        logger,
        "process_functionality_batch_files",
        batch_files_dir=str(config.batch_files_dir),
        output_dir=str(config.results_dir),
        max_concurrent_batches=config.max_concurrent_batches,
    ):
        processor.process_all_concurrently(max_concurrent=config.max_concurrent_batches)

    with log_duration(logger, "import_functionality_batch_results", results_dir=str(config.results_dir)):
        BatchFunctionalityResultImporter(config.results_dir).import_results()


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
        logger.exception("Functionality generation failed", error=str(e))
        raise RuntimeError(f"Functionality generation failed: {e}") from e


def classify_repositories(mode: Literal["search", "train", "test"], *, config_path: Path) -> None:
    """
    Classify repositories by their clusters using ML.
    """
    logger.info("Classifying repositories by their clusters using ML.")

    config = ClassificationConfig.from_yaml(mode=mode, config_path=config_path)
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
        logger.exception("Repository classification failed", mode=mode, error=str(e))
        raise RuntimeError(f"Repository classification failed: {e}") from e


def train_pipeline(*, config_path: Path, clustering_config_path: Path) -> None:
    """Run the full training pipeline end-to-end."""
    cluster_functionalities(mode="train", config_path=clustering_config_path)
    classify_repositories(mode="train", config_path=config_path)


def test_pipeline(*, config_path: Path, clustering_config_path: Path) -> None:
    """Run the full testing pipeline end-to-end."""
    cluster_functionalities(mode="test", config_path=clustering_config_path)
    classify_repositories(mode="test", config_path=config_path)


def load_functionalities_from_dir(directory: str) -> None:
    """Load all JSON files from a directory into Pydantic DTOs."""
    all_functionalities: list[CreateFunctionalityDto] = []
    all_snippets: list[UpdateSnippetDto] = []
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
