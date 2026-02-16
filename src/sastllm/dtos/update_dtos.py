from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UpdateRepositoryDto(BaseModel):
    """
    Data transfer object for updating repository.
    """

    repository_id: int
    name: Optional[str] = None
    label: Optional[str] = None
    split: Optional[str] = None
    processed: Optional[bool] = None


class UpdateFileDto(BaseModel):
    """
    Data transfer object for updating file.
    """

    file_id: int
    repository_id: Optional[int] = None
    language: Optional[str] = None
    filename: Optional[str] = None
    filepath: Optional[str] = None
    processed: Optional[bool] = None


class UpdateFunctionalityDto(BaseModel):
    """
    Data transfer object for updating functionality.
    """

    functionality_id: int
    snippet_id: Optional[int] = None
    file_id: Optional[int] = None
    description: Optional[str] = None
    tag: Optional[str] = None
    cluster_id: Optional[int] = None


class UpdateSnippetDto(BaseModel):
    """
    Data transfer object for updating snippet.
    """

    snippet_id: int
    file_id: Optional[int] = None
    code: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    processed: Optional[bool] = None


class UpdateClusterDto(BaseModel):
    cluster_id: int
    label: Optional[str] = None
