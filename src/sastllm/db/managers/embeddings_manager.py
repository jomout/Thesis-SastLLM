import logging
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.getLogger("qdrant_client.http").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class EmbeddingsManager:
    """
    Manages the storage and retrieval of embeddings using Qdrant.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, grpc_port: int = 6334):
        self.rest = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=None)
        self.grpc = QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True, timeout=None)

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
            raise ValueError("Embeddings list is empty.")

        size = len(embeddings[0])

        if not self.grpc.collection_exists(collection_name):
            self.grpc.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )
        # Prepare points for insertion
        points = [
            models.PointStruct(id=ids[i], vector=embeddings[i], payload=payloads[i])
            for i in range(len(embeddings))
        ]
        # Upsert points in batches
        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            self.grpc.upsert(collection_name=collection_name, points=batch_points)

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
                    return

            if next_page is None:
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
        query_filter = models.Filter(
            should=[
                models.FieldCondition(key=field, match=models.MatchValue(value=v)) for v in values
            ]
        )

        next_page = None

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
                yield int(p.id), np.asarray(p.vector, dtype=np.float32)

            if next_page is None:
                break

    def count_embeddings_by_payload_field(
        self, collection_name: str, field: str, values: str | list[str]
    ) -> int:
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
        query_filter = models.Filter(
            should=[
                models.FieldCondition(key=field, match=models.MatchValue(value=v)) for v in values
            ]
        )

        count_result = self.grpc.count(collection_name=collection_name, count_filter=query_filter)
        return count_result.count
