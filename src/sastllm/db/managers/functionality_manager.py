from typing import Iterator, List, Optional, cast

from sastllm.configs import get_logger
from sastllm.db.db import SessionLocal
from sastllm.db.models import FileModel, FunctionalityModel, SnippetModel
from sastllm.dtos import CreateFunctionalityDto, GetFunctionalityDto, UpdateFunctionalityDto

logger = get_logger(__name__)


class FunctionalityManager:
    """
    Manages database operations related to functionalities.
    Provides methods to add new functionalities and fetch existing ones.
    Uses SQLAlchemy ORM for database interactions.

    Attributes:
        Session: A SQLAlchemy session factory for creating database sessions.
        
    Methods:
        add_functionality: Inserts a new functionality record into the database.
        add_bulk_functionalities: Inserts multiple functionality records in a single transaction.
        get_functionalities: Lazily fetches all functionality records from the database.
        get_functionality: Fetches a functionality record from the database identified by ID.
        delete_functionality: Deletes a functionality record from the database identified by ID.
        update_functionality: Updates an existing functionality record. Only updates provided fields.
    """

    def __init__(self):
        self.Session = SessionLocal


    def add_functionality(self, functionality: CreateFunctionalityDto) -> int:
        """
        Inserts a new functionality record linked to a code snippet.

        Args:
            functionality (CreateFunctionalityDto): The functionality data transfer object containing all necessary information.

        Returns:
            int: The ID of the newly inserted functionality record.
        """
        logger.debug(f"Adding functionality for snippet ID: {functionality.snippet_id}")
        try:
            # Opens a session + transaction; commits on success, rolls back on error.
            with self.Session.begin() as session:
                functionality_model = FunctionalityModel(
                    snippet_id=functionality.snippet_id,
                    description=functionality.description,
                    tag=functionality.tag,
                    cluster_id=functionality.cluster_id,
                )
                session.add(functionality_model)
                session.flush()  # populate PK before exiting the context
                new_id = cast(int, functionality_model.functionality_id)
            return new_id
        except Exception as e:
            # Transaction is already rolled back by the context manager
            logger.error(f"Failed to add functionality: {e}")
            raise RuntimeError(f"Failed to add functionality: {e}") from e


    def add_bulk_functionalities(self, functionalities: List[CreateFunctionalityDto]) -> List[int]:
        """
        Inserts multiple functionality records in a single transaction.

        Args:
            functionalities (List[CreateFunctionalityDto]): A list of functionality data transfer objects.

        Returns:
            List[int]: The IDs of the newly inserted functionality rows, in input order.
        """
        logger.debug(f"Adding {len(functionalities)} functionalities (bulk)")

        if not functionalities:
            return []

        try:
            with self.Session.begin() as session:
                objs: List[FunctionalityModel] = []
                for f in functionalities:
                    obj = FunctionalityModel(
                        snippet_id=cast(int, f.snippet_id),
                        description=cast(str, f.description),
                        tag=cast(str, f.tag),
                        cluster_id=cast(Optional[int], f.cluster_id),
                    )
                    objs.append(obj)

                session.add_all(objs)
                session.flush()  # ensure PKs are populated

                ids: List[int] = [cast(int, o.functionality_id) for o in objs]
                logger.debug(f"Successfully added {len(ids)} functionalities (bulk)")
                return ids
        except Exception as e:
            logger.error(f"Failed bulk add functionalities: {e}")
            raise RuntimeError(f"Failed to add bulk functionalities: {e}") from e


    def get_functionalities(self, batch_size: int = 100) -> Iterator[GetFunctionalityDto]:
        """
        Lazily fetches all functionality records from the database and yields them as dataclass instances.

        Args:
            batch_size (int): Number of rows fetched per batch from the database.

        Yields:
            GetFunctionalityDto: A GetFunctionalityDto instance for each functionality.
        """
        logger.debug("Fetching functionalities from database.")
        with self.Session() as session:
            query = session.query(FunctionalityModel).yield_per(batch_size)
            for f in query:
                yield GetFunctionalityDto(
                    functionality_id=cast(int, f.functionality_id),
                    snippet_id=cast(int, f.snippet_id),
                    description=str(f.description),
                    tag=str(f.tag),
                    cluster_id=cast(Optional[int], f.cluster_id)
                )
            
    
    def get_functionality(self, functionality_id: int) -> Optional[GetFunctionalityDto]:
        """
        Fetches a functionality record from the database identified by ID.

        Args:
            functionality_id (int): The ID of a functionality record.
            
        Returns:
            Optional[GetFunctionalityDto]: A GetFunctionalityDto object with the corresponding ID.
        """
        logger.debug(f"Fetching functionality with ID: {functionality_id}")
        with self.Session() as session:
            f = session.query(FunctionalityModel).filter_by(functionality_id=functionality_id).one_or_none()
            if f is None:
                return None
            return GetFunctionalityDto(
                functionality_id=cast(int, f.functionality_id),
                snippet_id=cast(int, f.snippet_id),
                description=str(f.description),
                tag=str(f.tag),
                cluster_id=cast(Optional[int], f.cluster_id)
            )


    def delete_functionality(self, functionality_id: int) -> None:
        """
        Deletes a functionality record from the database identified by ID.

        Args:
            functionality_id (int): The ID of a functionality record.
        """
        logger.debug(f"Deleting functionality with ID: {functionality_id}")
        try:
            with self.Session.begin() as session:
                session.query(FunctionalityModel).filter_by(functionality_id=functionality_id).delete()
        except Exception as e:
            logger.error(f"Failed to delete functionality {functionality_id}: {e}")
            raise RuntimeError(f"Failed to delete functionality {functionality_id}: {e}") from e


    def update_functionality(self, functionality: UpdateFunctionalityDto) -> None:
        """
        Updates an existing functionality record. Only updates provided fields.

        Args:
            functionality (UpdateFunctionalityDto): The functionality data transfer object containing the ID and fields to update.
        """
        logger.debug(f"Updating functionality with ID: {functionality.functionality_id}")
        fields = {}
        if functionality.snippet_id is not None:
            fields[FunctionalityModel.snippet_id] = functionality.snippet_id
        if functionality.description is not None:
            fields[FunctionalityModel.description] = functionality.description
        if functionality.tag is not None:
            fields[FunctionalityModel.tag] = functionality.tag
        if functionality.cluster_id is not None:
            fields[FunctionalityModel.cluster_id] = functionality.cluster_id

        if not fields:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                session.query(FunctionalityModel).filter_by(
                    functionality_id=functionality.functionality_id
                ).update(fields, synchronize_session=False)
        except Exception as e:
            logger.error(f"Failed to update functionality {functionality.functionality_id}: {e}")
            raise RuntimeError(f"Failed to update functionality {functionality.functionality_id}: {e}") from e
        

    def update_bulk_functionalities(self, functionalities: List[UpdateFunctionalityDto]) -> None:
        """
        Updates multiple functionality records in a single transaction. Only updates provided fields.

        Args:
            functionalities (List[UpdateFunctionalityDto]): A list of functionality data transfer objects containing IDs and fields to update.
        """
        logger.debug(f"Updating {len(functionalities)} functionalities (bulk)")

        if not functionalities:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                for functionality in functionalities:
                    fields = {}
                    if functionality.snippet_id is not None:
                        fields[FunctionalityModel.snippet_id] = functionality.snippet_id
                    if functionality.description is not None:
                        fields[FunctionalityModel.description] = functionality.description
                    if functionality.tag is not None:
                        fields[FunctionalityModel.tag] = functionality.tag
                    if functionality.cluster_id is not None:
                        fields[FunctionalityModel.cluster_id] = functionality.cluster_id

                    if not fields:
                        continue  # Nothing to update for this record

                    session.query(FunctionalityModel).filter_by(
                        functionality_id=functionality.functionality_id
                    ).update(fields, synchronize_session=False)
        except Exception as e:
            logger.error(f"Failed bulk update functionalities: {e}")
            raise RuntimeError(f"Failed to bulk update functionalities: {e}") from e
        

    def get_functionalities_by_repository(self, repository_id: int, batch_size: int = 100) -> Iterator[GetFunctionalityDto]:
        """
        Lazily fetches all functionality records associated with a specific repository.

        Args:
            repository_id (int): The ID of the repository.
            batch_size (int): Number of rows fetched per batch from the database.

        Yields:
            GetFunctionalityDto: A GetFunctionalityDto instance for each functionality.
        """
        logger.debug(f"Fetching functionalities for repository ID: {repository_id}")

        with self.Session() as session:
            query = (
                session.query(FunctionalityModel)
                .join(SnippetModel, FunctionalityModel.snippet_id == SnippetModel.snippet_id)
                .join(FileModel, SnippetModel.file_id == FileModel.file_id)
                .filter(FileModel.repository_id == repository_id)
            )

            for f in query.yield_per(batch_size):
                yield GetFunctionalityDto(
                    functionality_id=cast(int, f.functionality_id),
                    snippet_id=cast(int, f.snippet_id),
                    description=str(f.description),
                    tag=str(f.tag),
                    cluster_id=cast(Optional[int], f.cluster_id)
                )