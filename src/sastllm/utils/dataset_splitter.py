from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sastllm.cluster import Embedder
from sastllm.configs.logging_config import get_logger
from sastllm.db import EmbeddingsManager, FunctionalityManager, RepositoryManager
from sastllm.dtos.update_dtos import UpdateRepositoryDto

logger = get_logger(__name__)


class DatasetSplitter:
    """
    A class to split datasets into training, validation, and test sets.
    """

    def __init__(self, model_name: str) -> None:
        self.repository_db = RepositoryManager()
        self.functionality_db = FunctionalityManager()
        self.embeddings_manager = EmbeddingsManager()
        self.embedder = Embedder(model_name=model_name)
        self.model_name = model_name

        # Collection name for embeddings
        self.collection_name = self.model_name.replace("/", "_")

    def _fetch_repositories(self) -> List[Tuple[int, str]]:
        """
        Fetch all repository IDs and their associated names from the database.

        Returns:
            List of tuples containing repository IDs and names.
        """
        repositories = []
        for repo in self.repository_db.get_repositories():
            repositories.append((repo.repository_id, repo.label))

        return repositories

    @staticmethod
    def _split(repositories: List[Tuple[int, str]], test_size: float) -> dict:
        """
        Split the dataset into training, and test sets.
        Args:
            repositories (List[Tuple[int, str]]): The data to be split [id, label].
            test_size (float): Proportion of the dataset to include in the test set.

        Returns:
            dict: A dictionary containing the split datasets.
        """
        X = []
        labels = []
        for repo_id, label in repositories:
            labels.append(label)
            X.append(repo_id)

        if test_size == 1.0:
            return {
                "train": {"X": [], "y": []},
                "test": {"X": X, "y": labels},
            }

        # First, split off the test set
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, stratify=labels, random_state=42)

        return {
            "train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test},
        }

    def embed_all_repositories(self) -> None:
        """
        Embed all repositories in the database and store their embeddings.

        Returns:
            None
        """
        repositories = self._fetch_repositories()

        try:
            for repo_id, repo_name in tqdm(repositories, desc="Embedding repositories"):
                functionalities = [(func.functionality_id, func.tag) for func in self.functionality_db.get_functionalities_by_repository(repo_id)]
                if not functionalities:
                    continue  # Skip repositories with no functionalities

                # Generating embeddings
                embeddings = self.embedder.embed(func_ids_tags=functionalities).tolist()

                # Store embeddings in Qdrant
                self.embeddings_manager.insert_embeddings(
                    collection_name=self.collection_name,
                    ids=[func_id for func_id, tag in functionalities],
                    embeddings=embeddings,
                    payloads=[{"repository_id": repo_id, "split": "full", "tag": tag} for func_id, tag in functionalities],
                )
        except Exception as e:
            logger.error(f"Failed to embed repositories: {e}")
            raise RuntimeError(f"Failed to embed repositories: {e}") from e

    def split_repositories(self, train_size: float, test_size: float) -> None:
        """
        Split repositories into training, validation, and test sets.

        Args:
            train_size (float): Proportion of the dataset to include in the training set.
            val_size (float): Proportion of the dataset to include in the validation set.
            test_size (float): Proportion of the dataset to include in the test set.

        Returns:
            None
        """
        repositories = self._fetch_repositories()
        assert abs(train_size + test_size - 1.0) < 1e-8, "Sizes must sum to 1."

        datasets = self._split(repositories, test_size)

        try:
            for split in {"train", "test"}:
                logger.info(f"Processing {split} set.")
                dataset = datasets[split]
                X = dataset["X"]

                # Gather functionalities for repositories in current split
                for repo_id in tqdm(X, desc=f"Updating repositories for {split} set"):
                    # Update repository split in the database
                    self.repository_db.update_repository(repository=UpdateRepositoryDto(repository_id=repo_id, split=split))

                    # Update embeddings in Qdrant
                    for functionality in self.functionality_db.get_functionalities_by_repository(repo_id):
                        self.embeddings_manager.update_embedding_payload(
                            collection_name=self.collection_name,
                            id=functionality.functionality_id,
                            payload={
                                "repository_id": repo_id,
                                "split": split,
                                "tag": functionality.tag,
                            },
                        )
        except Exception as e:
            logger.error(f"Failed to split repositories: {e}")
            raise RuntimeError(f"Failed to split repositories: {e}") from e
