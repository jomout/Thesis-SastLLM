import logging
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from sastllm.configs import get_logger

logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = get_logger(__name__)


class EmbeddingsManager:
    """
    Manages the storage and retrieval of embeddings using Qdrant.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, grpc_port: int = 6334):
        self.rest = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=100000)
        self.grpc = QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True, timeout=100000)
        logger.debug("Initialized Qdrant embedding manager", host=host, rest_port=port, grpc_port=grpc_port)

    def insert_embeddings(
        self,
        collection_name: str,
        ids: list,
        embeddings: list,
        payloads: list,
        batch_size: int = 1000,
    ):
        """
        Inserts embeddings into the specified collection in batches.
        """
        # Create collection if it doesn't exist
        if not embeddings:
            logger.error("Cannot insert an empty embedding batch", collection=collection_name)
            raise ValueError("Embeddings list is empty.")

        logger.info(
            "Inserting embeddings into Qdrant",
            collection=collection_name,
            embeddings=len(embeddings),
            vector_dim=len(embeddings[0]),
            batch_size=batch_size,
        )

        size = len(embeddings[0])

        if not self.grpc.collection_exists(collection_name):
            self.grpc.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )
        # Prepare points for insertion
        points = [models.PointStruct(id=ids[i], vector=embeddings[i], payload=payloads[i]) for i in range(len(embeddings))]
        # Upsert points in batches
        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            self.grpc.upsert(collection_name=collection_name, points=batch_points)
        logger.info("Inserted embeddings into Qdrant", collection=collection_name, embeddings=len(points))

    def update_payload_by_filter(
        self,
        collection_name: str,
        filter: dict,
        payload: dict,
    ):
        """
        Bulk payload update using Qdrant filters.

        Example filter:
            {"repository_id": {"$in": [1,2,3]}}

        NOTE: Translates to Qdrant Filter internally.
        """

        def build_filter(f: dict) -> models.Filter:
            must_conditions: list[models.Condition] = []

            for key, value in f.items():
                if isinstance(value, dict):
                    if "$in" in value:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value["$in"]),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported operator in filter for key '{key}': {value}")
                elif isinstance(value, (bool, int, str)):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported filter value type for key '{key}': {type(value)}")

            return models.Filter(must=must_conditions)

        try:
            qdrant_filter = build_filter(filter)

            self.rest.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=models.FilterSelector(filter=qdrant_filter),
            )
            logger.info(
                "Updated Qdrant payload by filter",
                collection=collection_name,
                filter=filter,
                payload_keys=sorted(payload),
            )

        except Exception as e:
            logger.error("Qdrant payload update failed", collection=collection_name, error=str(e), exc_info=True)
            raise RuntimeError(f"Failed bulk payload update: {e}") from e

    def update_embedding_payload(
        self,
        collection_name: str,
        id: int,
        payload: dict,
    ):
        """
        Updates the payload of a specific embedding by its ID.
        """
        self.grpc.set_payload(
            collection_name=collection_name,
            points=[id],
            payload=payload,
        )

    def get_existing_ids_from_collection(self, collection_name: str, batch_size: int = 1000) -> set:
        """
        Retrieves existing IDs from the specified collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            set: A set of existing IDs in the collection.
        """
        existing_ids = set()
        next_page = None

        while True:
            points, next_page = self.grpc.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                existing_ids.add(p.id)
            if next_page is None:
                break

        logger.debug("Fetched existing Qdrant ids", collection=collection_name, ids=len(existing_ids))
        return existing_ids

    def get_embeddings_by_ids(self, collection_name: str, ids: list, batch_size: int = 100):
        """
        Lazily retrieves embeddings (and payloads) by their IDs.

        Example:
            for point in manager.iter_embeddings_by_ids("sentences", big_id_list):
                print(point.id, point.payload)
        """
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            points = self.grpc.retrieve(
                collection_name=collection_name,
                ids=batch_ids,
                with_vectors=True,
                with_payload=False,
            )
            for p in points:
                yield p

    def get_n_embeddings(
        self,
        collection_name: str,
        n: int | None = None,
        batch_size: int = 1000,
    ):
        """
        Lazily yields up to n embeddings from the collection.

        Example:
            async for point in manager.get_n_embeddings("sentences", n=500):
                print(point.id, point.vector)
        """

        yielded = 0
        logger.debug("Streaming embeddings from Qdrant", collection=collection_name, limit=n, batch_size=batch_size)
        next_page = None

        while True:
            # determine batch limit
            limit = batch_size if n is None else min(batch_size, n - yielded)
            if limit <= 0:
                return

            points, next_page = self.grpc.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=next_page,
                with_vectors=True,
                with_payload=False,
            )

            if not points:
                return

            for p in points:
                yield int(p.id), np.asarray(p.vector, dtype=np.float32)
                yielded += 1

                if n is not None and yielded >= n:
                    logger.debug("Completed bounded Qdrant embedding stream", collection=collection_name, embeddings=yielded)
                    return

            if next_page is None:
                logger.debug("Completed Qdrant embedding stream", collection=collection_name, embeddings=yielded)
                return

    def get_embeddings_by_payload_field(
        self,
        collection_name: str,
        field: str,
        values: str | list[str],
        batch_size: int = 1000,
    ) -> Iterable[tuple[int, np.ndarray]]:
        """
        Lazily yields embeddings (and their IDs) matching the given payload field values.
        Args:
            collection_name (str): The name of the collection.
            field (str): The payload field to filter on.
            values (str | list[str]): The value(s) to match in the specified field.
            batch_size (int): Number of embeddings to fetch per batch.

        Yields:
            Iterable[tuple[int, np.ndarray]]: Tuples of (ID, embedding vector).
        """
        # Normalize to list
        if isinstance(values, str):
            values = [values]

        # Qdrant expects OR logic in "should"
        query_filter = models.Filter(should=[models.FieldCondition(key=field, match=models.MatchValue(value=v)) for v in values])

        next_page = None
        yielded = 0
        logger.debug(
            "Streaming filtered embeddings from Qdrant",
            collection=collection_name,
            field=field,
            values=values,
            batch_size=batch_size,
        )

        while True:
            points, next_page = self.grpc.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=batch_size,
                offset=next_page,
                with_payload=False,
                with_vectors=True,
            )

            if not points:
                break

            for p in points:
                yielded += 1
                yield int(p.id), np.asarray(p.vector, dtype=np.float32)

            if next_page is None:
                break
        logger.debug("Completed filtered Qdrant embedding stream", collection=collection_name, embeddings=yielded, field=field)

    def count_embeddings_by_payload_field(self, collection_name: str, field: str, values: str | list[str]) -> int:
        """
        Counts the number of embeddings matching the given payload field values.
        Args:
            collection_name (str): The name of the collection.
            field (str): The payload field to filter on.
            values (str | list[str]): The value(s) to match in the specified field.

        Returns:
            int: The count of matching embeddings.
        """
        # Normalize to list
        if isinstance(values, str):
            values = [values]

        # Qdrant expects OR logic in "should"
        query_filter = models.Filter(should=[models.FieldCondition(key=field, match=models.MatchValue(value=v)) for v in values])

        count_result = self.grpc.count(collection_name=collection_name, count_filter=query_filter)
        logger.debug("Counted filtered Qdrant embeddings", collection=collection_name, field=field, values=values, count=count_result.count)
        return count_result.count
