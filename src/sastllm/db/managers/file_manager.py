from typing import Iterator, Optional, cast

from sastllm.configs import get_logger
from sastllm.db.db import SessionLocal
from sastllm.db.models import FileModel
from sastllm.dtos import CreateFileDto, GetFileDto, UpdateFileDto

logger = get_logger(__name__)


class FileManager:
    """
    Manages database operations related to files.
    Provides methods to add new files and fetch existing ones.
    Uses SQLAlchemy ORM for database interactions.

    Attributes:
        Session: A SQLAlchemy session factory for creating database sessions.
    
    Methods:
        add_file: Inserts a new file record into the database.
        get_files: Lazily fetches all file records from the database.
        get_file: Fetches a file record from the database identified by ID.
        delete_file: Deletes a file record from the database identified by ID.
        update_file: Updates an existing file record. Only updates provided fields.
    """

    def __init__(self):
        self.Session = SessionLocal


    def add_file(self, file: CreateFileDto) -> int:
        """
        Inserts a new file record into the database.

        Args:
            file (CreateFileDto): The file data transfer object containing all necessary information.

        Returns:
            int: The auto-generated ID of the newly inserted file.
        """
        logger.debug(f"Adding file: {file.filename} at path: {file.filepath}")
        try:
            # Session + transaction; commits on success, rolls back on exception.
            with self.Session.begin() as session:
                file_model = FileModel(
                    repository_id=file.repository_id,
                    language=file.language,
                    filename=file.filename,
                    filepath=file.filepath
                )
                session.add(file_model)
                session.flush()  # populate PK before exiting the context
                new_id = cast(int, file_model.file_id)
            return new_id
        except Exception as e:
            # Rolled back automatically by the context manager.
            logger.error(f"Failed to add file: {e}")
            raise RuntimeError(f"Failed to add file: {e}") from e


    def get_files(self, batch_size: int = 100) -> Iterator[GetFileDto]:
        """
        Lazily fetches all file records from the database and yields them as dataclass instances.

        Args:
            batch_size (int): Number of rows fetched per batch from the database cursor.

        Yields:
            GetFileDto: A File data transfer object for each file in the database.
        """
        logger.debug("Fetching files from database.")
        with self.Session() as session:
            query = session.query(FileModel).yield_per(batch_size)
            for f in query:
                yield GetFileDto(
                    file_id=cast(int, f.file_id),
                    repository_id=cast(int, f.repository_id),
                    language=str(f.language),
                    filename=str(f.filename),
                    filepath=str(f.filepath),
                    processed=bool(f.processed)
                )


    def get_file(self, file_id: int) -> Optional[GetFileDto]:
        """
        Fetches a file record from the database identified by ID.

        Args:
            file_id (int): The ID of a file record.

        Returns:
            Optional[GetFileDto]: A GetFileDto object with the corresponding ID.
        """
        logger.debug(f"Fetching file with ID: {file_id}")
        with self.Session() as session:
            f = session.query(FileModel).filter_by(file_id=file_id).one_or_none()
            if f is None:
                return None
            return GetFileDto(
                file_id=cast(int, f.file_id),
                repository_id=cast(int, f.repository_id),
                language=str(f.language),
                filename=str(f.filename),
                filepath=str(f.filepath),
                processed=bool(f.processed)
            )
        

    def delete_file(self, file_id: int) -> None:
        """
        Deletes a file record from the database identified by ID.

        Args:
            file_id (int): The ID of a file record.
        """
        logger.debug(f"Deleting file with ID: {file_id}")
        try:
            with self.Session.begin() as session:
                session.query(FileModel).filter_by(file_id=file_id).delete()
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            raise RuntimeError(f"Failed to delete file {file_id}: {e}") from e
            

    def update_file(self, file: UpdateFileDto) -> None:
        """
        Updates an existing file record. Only updates provided fields.

        Args:
            file (UpdateFileDto): The file data transfer object containing fields to update.
        """
        logger.debug(f"Updating file with ID: {file.file_id}")
        fields = {}
        if file.repository_id is not None:
            fields[FileModel.repository_id] = file.repository_id
        if file.language is not None:
            fields[FileModel.language] = file.language
        if file.filename is not None:
            fields[FileModel.filename] = file.filename
        if file.filepath is not None:
            fields[FileModel.filepath] = file.filepath

        if not fields:
            return  # Nothing to update

        try:
            with self.Session.begin() as session:
                session.query(FileModel).filter_by(file_id=file.file_id).update(
                    fields, synchronize_session=False
                )
        except Exception as e:
            logger.error(f"Failed to update file {file.file_id}: {e}")
            raise RuntimeError(f"Failed to update file {file.file_id}: {e}") from e