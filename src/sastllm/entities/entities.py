from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Repository:
    """
    Represents a repository.
    """

    repository_id: int
    name: str
    label: Optional[str]
    split: Optional[str]
    processed: bool


@dataclass
class File:
    """
    Represents a file.
    """

    file_id: int
    repository_id: int
    language: str
    filename: str
    filepath: str
    processed: bool


@dataclass
class Functionality:
    """
    Represents a functionality.
    """

    functionality_id: int
    snippet_id: int
    description: str
    tag: str
    cluster_id: Optional[int]


@dataclass
class Snippet:
    """
    Represents a snippet.
    """

    snippet_id: int
    file_id: int
    code: str
    start_line: int
    end_line: int
    processed: bool


@dataclass
class Cluster:
    cluster_id: int
    label: Optional[str]


@dataclass
class FunctionalityWithCluster:
    """
    Minimal functionality representation for sequence encoders.
    """

    functionality_id: int
    cluster_id: Optional[int]


@dataclass
class SnippetWithFileAndRepository(Snippet):
    """
    Represents a snippet along with its associated file and repository information.
    """

    language: str
    filename: str
    filepath: str
    repository_id: int


@dataclass
class RepositoryWithClusterDistribution:
    """
    Represents a repository along with its cluster distribution and ordered functionalities.
    """

    repository_id: int
    data: Optional[dict[int, int]]
    label: Optional[str]
    ordered_functionalities: Optional[list[FunctionalityWithCluster]] = None
