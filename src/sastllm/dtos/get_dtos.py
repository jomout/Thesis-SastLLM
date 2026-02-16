from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class GetRepositoryDto(BaseModel):
    """
    Data transfer object for fetching repositories.
    """

    repository_id: int
    name: str
    label: Optional[str]
    split: Optional[str]
    processed: bool


class GetFileDto(BaseModel):
    """
    Data transfer object for fetching files.
    """

    file_id: int
    repository_id: int
    language: str
    filename: str
    filepath: str
    processed: bool


class GetFunctionalityDto(BaseModel):
    """
    Data transfer object for fetching functionalities.
    """

    functionality_id: int
    snippet_id: int
    description: str
    tag: str
    cluster_id: Optional[int]


class GetSnippetDto(BaseModel):
    """
    Data transfer object for fetching snippets.
    """

    snippet_id: int
    file_id: int
    code: str
    start_line: int
    end_line: int
    processed: bool


class GetClusterDto(BaseModel):
    cluster_id: int
    label: Optional[str]


class GetExtendedSnippetDto(GetSnippetDto):
    """
    Extended DTO for fetching snippets including additional file and repository information.
    """

    snippet_id: int
    file_id: int
    code: str
    start_line: int
    end_line: int
    processed: bool
    language: str
    filename: str
    filepath: str
    repository_id: int


class GetClassificationRepositoryDto(BaseModel):
    """
    Data transfer object for fetching classification repositories with cluster distribution.
    """

    repository_id: int
    data: Optional[dict[int, int]]  # Mapping of cluster_id to count
    label: Optional[str]
