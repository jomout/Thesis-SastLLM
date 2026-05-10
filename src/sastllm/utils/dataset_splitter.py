from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sastllm.cluster import Embedder
from sastllm.configs.logging_config import get_logger
from sastllm.db import EmbeddingsManager, FunctionalityManager, RepositoryManager

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Optimized dataset splitter with:
    - batched embedding
    - bulk DB access
    - bulk Qdrant updates
    """

    BATCH_SIZE = 256  # tune based on your hardware

    def __init__(self, model_name: str) -> None:
        self.repository_db = RepositoryManager()
        self.functionality_db = FunctionalityManager()
        self.embeddings_manager = EmbeddingsManager()
        self.embedder = Embedder(model_name=model_name)
        self.model_name = model_name

        self.collection_name = self.model_name.replace("/", "_")

    # -------------------------
    # DATA FETCHING
    # -------------------------

    def _fetch_repositories(self) -> List[Tuple[int, str]]:
        return [(repo.repository_id, (repo.label or "unknown")) for repo in self.repository_db.get_repositories()]

    def _fetch_all_functionalities(self) -> Dict[int, List[Tuple[int, str]]]:
        """
        Returns:
            {repo_id: [(func_id, tag), ...]}
        """
        grouped = defaultdict(list)

        all_funcs = self.functionality_db.get_all_functionalities()

        for repo_id, func in all_funcs:
            grouped[repo_id].append((func.functionality_id, func.tag))

        return grouped

    # -------------------------
    # SPLITTING
    # -------------------------

    @staticmethod
    def _split(repositories: List[Tuple[int, str]], test_size: float) -> dict:
        X, labels = zip(*repositories)

        if test_size == 1.0:
            return {
                "train": {"X": [], "y": []},
                "test": {"X": list(X), "y": list(labels)},
            }

        X_train, X_test, y_train, y_test = train_test_split(
            list(X),
            list(labels),
            test_size=test_size,
            stratify=labels,
            random_state=42,
        )

        return {
            "train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test},
        }

    # -------------------------
    # EMBEDDING (OPTIMIZED)
    # -------------------------

    def embed_all_repositories(self) -> None:
        """
        Fully optimized embedding:
        - single DB fetch
        - global batching
        - bulk inserts
        """

        repositories = self._fetch_repositories()
        all_funcs = self._fetch_all_functionalities()

        # Flatten everything
        all_functionalities = [(func_id, tag, repo_id) for repo_id, _ in repositories for func_id, tag in all_funcs.get(repo_id, [])]

        if not all_functionalities:
            logger.warning("No functionalities found.")
            return

        logger.info(f"Total functionalities to embed: {len(all_functionalities)}")

        try:
            for i in tqdm(
                range(0, len(all_functionalities), self.BATCH_SIZE),
                desc="Embedding (batched)",
            ):
                batch = all_functionalities[i : i + self.BATCH_SIZE]

                func_pairs = [(fid, tag) for fid, tag, _ in batch]

                embeddings = self.embedder.embed(func_ids_tags=func_pairs)

                self.embeddings_manager.insert_embeddings(
                    collection_name=self.collection_name,
                    ids=[fid for fid, _, _ in batch],
                    embeddings=embeddings.tolist(),
                    payloads=[
                        {
                            "repository_id": repo_id,
                            "split": "full",
                            "tag": tag,
                        }
                        for fid, tag, repo_id in batch
                    ],
                )

        except Exception as e:
            logger.error(f"Failed to embed repositories: {e}")
            raise RuntimeError(f"Failed to embed repositories: {e}") from e

    # -------------------------
    # SPLIT + BULK UPDATE
    # -------------------------

    def split_repositories(self, train_size: float, test_size: float) -> None:
        """
        Optimized splitting:
        - no per-repo loops
        - bulk DB + Qdrant updates
        """

        repositories = self._fetch_repositories()

        assert abs(train_size + test_size - 1.0) < 1e-8

        datasets = self._split(repositories, test_size)

        try:
            for split in ["train", "test"]:
                repo_ids = datasets[split]["X"]

                logger.info(f"Updating {split} set ({len(repo_ids)} repos)")

                if not repo_ids:
                    continue

                # -------------------------
                # BULK DB UPDATE (you implement this)
                # -------------------------
                self.repository_db.bulk_update_split(repo_ids, split)
                logger.info(f"Updated repository DB for {split} set with {len(repo_ids)} repositories.")

                # -------------------------
                # BULK QDRANT UPDATE
                # -------------------------
                self.embeddings_manager.update_payload_by_filter(
                    collection_name=self.collection_name,
                    filter={"repository_id": {"$in": repo_ids}},
                    payload={"split": split},
                )
                logger.info(f"Updated Qdrant for {split} set with {len(repo_ids)} repositories.")

        except Exception as e:
            logger.error(f"Failed to split repositories: {e}")
            raise RuntimeError(f"Failed to split repositories: {e}") from e
