from collections import defaultdict
from typing import Dict, Iterator, Literal, Optional, cast

from sqlalchemy import func

from sastllm.configs import get_logger
from sastllm.db.db import SessionLocal
from sastllm.db.models import (
    FileModel,
    FunctionalityModel,
    RepositoryModel,
    SnippetModel,
)
from sastllm.dtos import (
    CreateRepositoryDto,
    GetClassificationRepositoryDto,
    GetRepositoryDto,
    UpdateRepositoryDto,
)

logger = get_logger(__name__)


class RepositoryManager:
    """
    Manages database operations related to repositories.
    Provides methods to add new repositories and fetch existing ones.
    Uses SQLAlchemy ORM for database interactions.

    Attributes:
        Session: A SQLAlchemy session factory for creating database sessions.

    Methods:
        add_repository: Inserts a new repository record into the database.
        get_repositories: Lazily fetches all repository records from the database.
        get_repository: Fetches a repository record from the database identified by ID.
        delete_repository: Deletes a repository record from the database identified by ID.
        update_repository: Updates an existing repository record. Only updates provided fields.
    """

    def __init__(self):
        self.Session = SessionLocal

    def add_repository(self, repository: CreateRepositoryDto) -> int:
        """
        Inserts a new repository record into the database.

        Args:
            repository (CreateRepositoryDto): The repository data transfer object containing
            all necessary information.

        Returns:
            int: The auto-generated ID of the newly inserted repository.
        """
        logger.debug(f"Adding repository: {repository.name} with label: {repository.label}")
        try:
            # Session + transaction; commits on success, rolls back on exception.
            with self.Session.begin() as session:
                repo_model = RepositoryModel(
                    name=repository.name, label=repository.label, split=repository.split
                )
                session.add(repo_model)
                session.flush()  # populate PK before exiting the context
                new_id = cast(int, repo_model.repository_id)
            return new_id
        except Exception as e:
            # Rolled back automatically by the context manager.
            logger.error(f"Failed to add repository: {e}")
            raise RuntimeError(f"Failed to add repository: {e}") from e

    def get_repositories(
        self, split: Optional[Literal["train", "test"]] = None, batch_size: int = 100
    ) -> Iterator[GetRepositoryDto]:
        """
        Lazily fetches all repository records from the database and yields them
        as dataclass instances.

        Args:
            split (Optional[Literal["train", "test"]]): Optional filter to fetch
            repositories by their split.
            batch_size (int): Number of rows fetched per batch from the database cursor.

        Yields:
            GetRepositoryDto: A data transfer object for each repository in the database.
        """
        logger.debug("Fetching repositories from database.")
        with self.Session() as session:
            query = session.query(RepositoryModel).yield_per(batch_size)
            if split is not None:
                query = query.filter(RepositoryModel.split == split)
            for r in query:
                yield GetRepositoryDto(
                    repository_id=cast(int, r.repository_id),
                    split=str(r.split) if r.split is not None else None,
                    name=str(r.name),
                    label=str(r.label) if r.label is not None else None,
                    processed=bool(r.processed),
                )

    def get_repository(self, repository_id: int) -> Optional[GetRepositoryDto]:
        """
        Fetches a repository record from the database identified by ID.

        Args:
            repository_id (int): The ID of a repository record.

        Returns:
            Optional[Repository]: A Repository object with the corresponding ID.
        """
        logger.debug(f"Fetching repository with ID: {repository_id}")
        with self.Session() as session:
            r = session.query(RepositoryModel).filter_by(repository_id=repository_id).one_or_none()
            if r is None:
                return None
            return GetRepositoryDto(
                repository_id=cast(int, r.repository_id),
                split=str(r.split) if r.split is not None else None,
                name=str(r.name),
                label=str(r.label) if r.label is not None else None,
                processed=bool(r.processed),
            )

    def get_repository_by_name(self, name: str) -> Optional[GetRepositoryDto]:
        """
        Fetches a repository record from the database identified by name.

        Args:
            name (str): The name of a repository record.
        Returns:
            Optional[Repository]: A Repository object with the corresponding name.
        """
        logger.debug(f"Fetching repository with name: {name}")
        with self.Session() as session:
            r = session.query(RepositoryModel).filter_by(name=name).one_or_none()
            if r is None:
                return None
            return GetRepositoryDto(
                repository_id=cast(int, r.repository_id),
                split=str(r.split) if r.split is not None else None,
                name=str(r.name),
                label=str(r.label) if r.label is not None else None,
                processed=bool(r.processed),
            )

    def delete_repository(self, repository_id: int) -> None:
        """
        Deletes a repository record from the database identified by ID.

        Args:
            repository_id (int): The ID of a repository record.
        """
        logger.debug(f"Deleting repository with ID: {repository_id}")
        try:
            with self.Session.begin() as session:
                session.query(RepositoryModel).filter_by(repository_id=repository_id).delete()
        except Exception as e:
            logger.error(f"Failed to delete repository {repository_id}: {e}")
            raise RuntimeError(f"Failed to delete repository {repository_id}: {e}") from e

    def update_repository(self, repository: UpdateRepositoryDto) -> None:
        """
        Updates an existing repository record. Only updates provided fields.

        Args:
            repository (UpdateRepositoryDto): The repository data transfer object
            containing fields to update.
        """
        logger.debug(f"Updating repository with ID: {repository.repository_id}")
        fields = {}
        if repository.name is not None:
            fields[RepositoryModel.name] = repository.name
        if repository.label is not None:
            fields[RepositoryModel.label] = repository.label
        if repository.split is not None:
            fields[RepositoryModel.split] = repository.split
        if repository.processed is not None:
            fields[RepositoryModel.processed] = repository.processed

        if not fields:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                session.query(RepositoryModel).filter_by(
                    repository_id=repository.repository_id
                ).update(fields, synchronize_session=False)
        except Exception as e:
            logger.error(f"Failed to update repository {repository.repository_id}: {e}")
            raise RuntimeError(
                f"Failed to update repository {repository.repository_id}: {e}"
            ) from e

    def get_repositories_with_cluster_ids(
        self, split: Optional[Literal["train", "test"]] = None, batch_size: int = 100
    ) -> Iterator[GetClassificationRepositoryDto]:
        with self.Session() as session:
            # LEFT JOIN through the graph so empty repos are kept
            query = (
                session.query(
                    RepositoryModel.repository_id,
                    FunctionalityModel.cluster_id,
                    RepositoryModel.label,
                )
                .select_from(RepositoryModel)
                .join(
                    FileModel,
                    FileModel.repository_id == RepositoryModel.repository_id,
                    isouter=True,
                )
                .join(
                    SnippetModel,
                    SnippetModel.file_id == FileModel.file_id,
                    isouter=True,
                )
                .join(
                    FunctionalityModel,
                    FunctionalityModel.snippet_id == SnippetModel.snippet_id,
                    isouter=True,
                )
                .filter(RepositoryModel.processed.is_(True))
                .order_by(RepositoryModel.repository_id)  # cluster_id order not needed for counting
                .execution_options(stream_results=True)
                .yield_per(batch_size)
            )
            if split is not None:
                query = query.filter(RepositoryModel.split == split)

            current_repo_id: Optional[int] = None
            current_label: Optional[str] = None
            current_counts: Dict[int, int] = defaultdict(int)

            for repo_id, cluster_id, label in query:
                # boundary: new repository encountered
                if current_repo_id is not None and repo_id != current_repo_id:
                    yield GetClassificationRepositoryDto(
                        repository_id=current_repo_id,
                        data=dict(current_counts) if current_counts else None,
                        label=current_label,
                        labels=[],
                    )
                    current_counts.clear()
                    current_label = None

                current_repo_id = repo_id
                current_label = label

                if cluster_id is not None:
                    current_counts[cluster_id] += 1

            # emit the final group
            if current_repo_id is not None:
                yield GetClassificationRepositoryDto(
                    repository_id=current_repo_id,
                    data=dict(current_counts) if current_counts else None,
                    label=current_label,
                    labels=[],
                )

    def get_num_repositories(self) -> int:
        """
        Fetches the total number of processed repositories in the database.

        Returns:
            int: The total number of processed repositories.
        """
        with self.Session() as session:
            return (
                session.query(func.count(RepositoryModel.repository_id))
                .filter(RepositoryModel.processed.is_(True))
                .scalar()
            )

    def get_num_of_clusters(self) -> int:
        with self.Session() as session:
            return session.query(func.count(FunctionalityModel.cluster_id.distinct())).scalar()
