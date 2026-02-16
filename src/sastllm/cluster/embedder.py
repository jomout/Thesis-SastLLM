from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from sastllm.configs import get_logger
from sastllm.db.managers.embeddings_manager import EmbeddingsManager

logger = get_logger()


class Embedder:
    """
    A utility class for generating semantic vector embeddings of normalized
    reasoning or explanation texts using a SentenceTransformer model.

    It supports both single-string and batch embedding, checks for existing
    embeddings in db (via EmbeddingsManager), and avoids recomputation.
    """

    def __init__(self, model_name: str):
        logger.debug("Initializing Embedder with model: %s", model_name)

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize Embedder model {model_name}: {e}")
            raise ValueError(f"Failed to initialize Embedder model {model_name}: {e}") from e

        # Qdrant setup
        self.collection_name = model_name.replace("/", "_")
        self.embeddings_manager = EmbeddingsManager()

        # Check if collection exists before trying to fetch from it
        if self.embeddings_manager.grpc.collection_exists(self.collection_name):
            self.cached_ids = self.embeddings_manager.get_existing_ids_from_collection(collection_name=self.collection_name)
            logger.info(f"Initialized Embedder: found {len(self.cached_ids)} existing embeddings in db collection '{self.collection_name}'.")
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist yet. No cached embeddings loaded.")
            self.cached_ids = set()

    def embed(self, func_ids_tags: List[Tuple[int, str]], normalize: bool = True) -> np.ndarray:
        """
        Generate dense vector embeddings for given input texts.
        Checks db for existing embeddings and computes new ones only for missing IDs.

        Args:
            func_ids_tags (List[Tuple[int, str]]): (id, text) tuples.
            normalize (bool): Whether to normalize embeddings to unit length.

        Returns:
            np.ndarray: Embedding matrix (n_samples, embedding_dim).
        """
        logger.debug(f"Generating embeddings for {len(func_ids_tags)} texts")

        # Determine which IDs need to be computed
        missing = [(i, t) for i, t in func_ids_tags if i not in self.cached_ids]

        if missing:
            logger.info(f"Computing embeddings for {len(missing)} new items (not in db).")
            missing_ids, missing_texts = zip(*missing)
            new_embeddings = self.model.encode(
                list(missing_texts),
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            # Note: Not saving these new embeddings to db (read-only mode)
            computed = {i: emb for i, emb in zip(missing_ids, new_embeddings)}
        else:
            computed = {}
            logger.info("All items found in db cache.")

        # Collect final embeddings (from db + newly computed)
        embeddings = []
        for i, _ in func_ids_tags:
            if i in self.cached_ids:
                # Lazy retrieve from db
                for point in self.embeddings_manager.get_embeddings_by_ids(self.collection_name, [i]):
                    embeddings.append(point.vector)
                    break
            else:
                embeddings.append(computed[i])

        embeddings = np.stack(embeddings, axis=0)
        logger.debug(f"Generated embeddings for {len(embeddings)} texts.")
        return embeddings
