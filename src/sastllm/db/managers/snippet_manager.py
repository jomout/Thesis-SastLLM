from typing import Iterator, List, Optional, cast

from sqlalchemy import func

from sastllm.configs import get_logger
from sastllm.db.db import SessionLocal
from sastllm.db.models import FileModel, SnippetModel
from sastllm.dtos import CreateSnippetDto, GetExtendedSnippetDto, GetSnippetDto, UpdateSnippetDto

logger = get_logger(__name__)


class SnippetManager:
    """
    Manages database operations related to snippets.
    Provides methods to add new snippets and fetch existing ones.
    Uses SQLAlchemy ORM for database interactions.

    Attributes:
        Session: A SQLAlchemy session factory for creating database sessions.
        
    Methods:
        add_snippet: Inserts a new snippet record into the database.
        add_bulk_snippets: Inserts multiple snippet records in a single transaction.
        get_snippets: Lazily fetches all snippet records from the database.
        get_snippet: Fetches a snippet record from the database identified by ID.
        delete_snippet: Deletes a snippet record from the database identified by ID.
        update_snippet: Updates an existing snippet record. Only updates provided fields.
    """

    def __init__(self):
        self.Session = SessionLocal


    def add_snippet(self, snippet: CreateSnippetDto) -> int:
        """
        Inserts a new code snippet associated with a file.

        Args:
            snippet (CreateSnippetDto): The code snippet data transfer object containing all necessary information.

        Returns:
            int: The ID of the newly inserted code snippet.
        """
        logger.debug(f"Adding snippet for file ID: {snippet.file_id} (lines {snippet.start_line}-{snippet.end_line})")
        try:
            # Opens a session + transaction; commits on success, rolls back on error.
            with self.Session.begin() as session:
                snippet_model = SnippetModel(
                    file_id=snippet.file_id,
                    code=snippet.code,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                )
                session.add(snippet_model)
                session.flush()  # ensure PK is populated
                new_id = cast(int, snippet_model.snippet_id)
            return new_id
        except Exception as e:
            # The context manager already rolled back on exception.
            logger.error(f"Failed to add snippet: {e}")
            raise RuntimeError(f"Failed to add snippet: {e}") from e


    def add_bulk_snippets(self, snippets: List[CreateSnippetDto]) -> List[int]:
        """
        Inserts multiple code snippets for the same file in a single transaction.

        Args:
            snippets (List[CreateSnippetDto]): List of snippet DTOs.

        Returns:
            List[int]: The IDs of the inserted snippets, in the same order as provided.
        """
        logger.debug(f"Adding {len(snippets)} snippets")

        if not snippets:
            return []

        try:
            with self.Session.begin() as session:
                objects: List[SnippetModel] = []
                for s in snippets:
                    obj = SnippetModel(
                        file_id=cast(int, s.file_id),
                        code=cast(str, s.code),
                        start_line=cast(int, s.start_line),
                        end_line=cast(int, s.end_line),
                    )
                    objects.append(obj)

                session.add_all(objects)
                session.flush()  # ensure PKs are populated

                ids: List[int] = [cast(int, o.snippet_id) for o in objects]
                return ids
        except Exception as e:
            logger.error(f"Failed to add bulk snippets: {e}")
            raise RuntimeError(f"Failed to add bulk snippets: {e}") from e


    def get_snippets(self, batch_size: int = 100) -> Iterator[GetSnippetDto]:
        """
        Lazily fetches all code snippet records from the database and yields them as dataclass instances.

        Args:
            batch_size (int): Number of rows fetched per batch from the database.

        Yields:
            GetSnippetDto: A dataclass instance for each code snippet.
        """
        logger.debug("Fetching snippets from database.")
        with self.Session() as session:
            query = session.query(SnippetModel).yield_per(batch_size)
            for s in query:
                yield GetSnippetDto(
                    snippet_id=cast(int, s.snippet_id),
                    file_id=cast(int, s.file_id),
                    code=str(s.code),
                    start_line=cast(int, s.start_line),
                    end_line=cast(int, s.end_line),
                    processed=cast(bool, s.processed)
                )


    def get_snippet(self, snippet_id: int) -> Optional[GetSnippetDto]:
        """
        Fetches a snippet record from the database identified by ID.

        Args:
            snippet_id (int): The ID of a snippet record.
            
        Returns:
            Optional[GetSnippetDto]: A GetSnippetDto object with the corresponding ID.
        """
        logger.debug(f"Fetching snippet with ID: {snippet_id}")
        with self.Session() as session:
            s = session.query(SnippetModel).filter_by(snippet_id=snippet_id).one_or_none()
            if s is None:
                return None
            return GetSnippetDto(
                snippet_id=cast(int, s.snippet_id),
                file_id=cast(int, s.file_id),
                code=str(s.code),
                start_line=cast(int, s.start_line),
                end_line=cast(int, s.end_line),
                processed=cast(bool, s.processed)
            )


    def delete_snippet(self, snippet_id: int) -> None:
        """
        Deletes a snippet record from the database identified by ID.

        Args:
            snippet_id (int): The ID of a snippet record.
        """
        logger.debug(f"Deleting snippet with ID: {snippet_id}")        
        try:
            with self.Session.begin() as session:
                session.query(SnippetModel).filter_by(snippet_id=snippet_id).delete()
        except Exception as e:
            logger.error(f"Failed to delete snippet {snippet_id}: {e}")
            raise RuntimeError(f"Failed to delete snippet {snippet_id}: {e}") from e


    def update_snippet(self, snippet: UpdateSnippetDto) -> None:
        """
        Updates an existing snippet record. Only updates provided fields.

        Args:
            snippet_id (int): The ID of a snippet record.
            file_id (Optional[int]): Identifier for the file it belongs to.
            code (Optional[str]): The snippet or chunk of source code.
            start_line (Optional[int]): The starting line of the snippet.
            end_line (Optional[str]): The ending line of the snippet.
            processed (Optional[bool]): Whether the snippet has been processed.
        """
        logger.debug(f"Updating snippet with ID: {snippet.snippet_id}")
        fields = {}
        if snippet.file_id is not None:
            fields[SnippetModel.file_id] = snippet.file_id
        if snippet.code is not None:
            fields[SnippetModel.code] = snippet.code
        if snippet.start_line is not None:
            fields[SnippetModel.start_line] = snippet.start_line
        if snippet.end_line is not None:
            fields[SnippetModel.end_line] = snippet.end_line
        if snippet.processed is not None:
            fields[SnippetModel.processed] = snippet.processed

        if not fields:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                session.query(SnippetModel).filter_by(snippet_id=snippet.snippet_id).update(
                    fields, synchronize_session=False
                )
        except Exception as e:
            logger.error(f"Failed to update snippet {snippet.snippet_id}: {e}")
            raise RuntimeError(f"Failed to update snippet {snippet.snippet_id}: {e}") from e


    def update_bulk_snippets(self, snippets: List[UpdateSnippetDto]) -> None:
        """
        Updates multiple existing snippet records in a single transaction.
        Only updates provided fields for each snippet.

        Args:
            snippets (List[UpdateSnippetDto]): List of snippet DTOs containing fields to update.
        """
        logger.debug(f"Updating {len(snippets)} snippets")

        if not snippets:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                for snippet in snippets:
                    fields = {}
                    if snippet.file_id is not None:
                        fields[SnippetModel.file_id] = snippet.file_id
                    if snippet.code is not None:
                        fields[SnippetModel.code] = snippet.code
                    if snippet.start_line is not None:
                        fields[SnippetModel.start_line] = snippet.start_line
                    if snippet.end_line is not None:
                        fields[SnippetModel.end_line] = snippet.end_line
                    if snippet.processed is not None:
                        fields[SnippetModel.processed] = snippet.processed

                    if not fields:
                        continue  # Nothing to update for this snippet

                    session.query(SnippetModel).filter_by(snippet_id=snippet.snippet_id).update(
                        fields, synchronize_session=False
                    )
        except Exception as e:
            logger.error(f"Failed to update bulk snippets: {e}")
            raise RuntimeError(f"Failed to update bulk snippets: {e}") from e
        

    def get_num_snippets(self) -> int:
        """
        Fetches the total number of unprocessed code snippets in the database.
        
        Returns:
            int: The total number of unprocessed code snippets.
        """
        with self.Session() as session:
            return session.query(func.count(SnippetModel.snippet_id)).filter(SnippetModel.processed.is_(False)).scalar()
        

    def get_snippets_with_file_meta(self, batch_size: int = 100) -> Iterator[GetExtendedSnippetDto]:
        """
        Lazily fetches all code snippets joined with their file metadata, to be processed.

        Args:
            batch_size (int): Number of rows fetched per batch from the database.

        Yields:
            GetExtendedSnippetDto: An object containing snippet ID, code, filename, 
                                 repository, filepath, and programming language.
        """
        logger.debug("Fetching code snippets with file metadata from database.")
        with self.Session() as session:
            query = (
                session.query(
                    SnippetModel.snippet_id,
                    SnippetModel.code,
                    SnippetModel.start_line,
                    SnippetModel.end_line,
                    SnippetModel.processed,
                    SnippetModel.file_id,
                    FileModel.filename,
                    FileModel.repository_id,
                    FileModel.filepath,
                    FileModel.language,
                )
                .filter(SnippetModel.processed.is_(False))
                .join(FileModel, SnippetModel.file_id == FileModel.file_id)
                .order_by(SnippetModel.snippet_id.asc())
                .execution_options(stream_results=True)
                .yield_per(batch_size)
            )
            for row in query:
                yield GetExtendedSnippetDto(
                    snippet_id=row.snippet_id,
                    file_id=row.file_id,
                    code=row.code,
                    filename=row.filename,
                    repository_id=row.repository_id,
                    filepath=row.filepath,
                    language=row.language,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    processed=row.processed
                )