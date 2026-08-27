from __future__ import annotations

from pydantic import BaseModel


class CreateRepositoryDto(BaseModel):
    """
    Data transfer object for creating repository.
    """

    name: str
    label: str | None = None
    split: str | None = None


class CreateFileDto(BaseModel):
    """
    Data transfer object for creating file.
    """

    repository_id: int
    language: str
    filename: str
    filepath: str


class CreateFunctionalityDto(BaseModel):
    """
    Data transfer object for creating functionality.
    """

    snippet_id: int
    description: str
    tag: str
    cluster_id: int | None = None


class CreateSnippetDto(BaseModel):
    """
    Data transfer object for creating snippet.
    """

    file_id: int
    code: str
    start_line: int
    end_line: int


class CreateClusterDto(BaseModel):
    label: str | None = None
