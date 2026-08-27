from __future__ import annotations

from pydantic import BaseModel


class UpdateRepositoryDto(BaseModel):
    """
    Data transfer object for updating repository.
    """

    repository_id: int
    name: str | None = None
    label: str | None = None
    split: str | None = None
    processed: bool | None = None


class UpdateFileDto(BaseModel):
    """
    Data transfer object for updating file.
    """

    file_id: int
    repository_id: int | None = None
    language: str | None = None
    filename: str | None = None
    filepath: str | None = None
    processed: bool | None = None


class UpdateFunctionalityDto(BaseModel):
    """
    Data transfer object for updating functionality.
    """

    functionality_id: int
    snippet_id: int | None = None
    file_id: int | None = None
    description: str | None = None
    tag: str | None = None
    cluster_id: int | None = None


class UpdateSnippetDto(BaseModel):
    """
    Data transfer object for updating snippet.
    """

    snippet_id: int
    file_id: int | None = None
    code: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    processed: bool | None = None


class UpdateClusterDto(BaseModel):
    cluster_id: int
    label: str | None = None
