from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from dotenv import load_dotenv

from argus.classification import ClassificationConfig
from argus.classification.encoders import LabelMapping
from argus.configs import setup_logging
from argus.db.db import SessionLocal
from argus.db.models import FileModel, FunctionalityModel, RepositoryModel, SnippetModel
from argus.entities import FunctionalityWithCluster, RepositoryWithClusterDistribution


@dataclass(frozen=True)
class FunctionalityInspection:
    functionality_id: int
    snippet_id: int
    start_line: int
    end_line: int
    description: str
    tag: str
    cluster_id: int | None


@dataclass
class SnippetInspection:
    snippet_id: int
    start_line: int
    end_line: int
    functionalities: list[FunctionalityInspection] = field(default_factory=list)


@dataclass
class FileInspection:
    file_id: int
    filepath: str
    filename: str
    language: str
    snippets: dict[int, SnippetInspection] = field(default_factory=dict)


@dataclass
class RepositoryInspection:
    repository_id: int
    name: str
    label: str | None
    split: str | None
    processed: bool
    files: dict[int, FileInspection] = field(default_factory=dict)
    functionalities: list[FunctionalityInspection] = field(default_factory=list)

    def to_classification_entity(self) -> RepositoryWithClusterDistribution:
        counts: dict[int, int] = defaultdict(int)
        ordered_functionalities: list[FunctionalityWithCluster] = []
        for functionality in self.functionalities:
            if functionality.cluster_id is not None:
                counts[functionality.cluster_id] += 1
            ordered_functionalities.append(
                FunctionalityWithCluster(
                    functionality_id=functionality.functionality_id,
                    cluster_id=functionality.cluster_id,
                )
            )
        return RepositoryWithClusterDistribution(
            repository_id=self.repository_id,
            data=dict(counts) if counts else None,
            label=self.label,
            ordered_functionalities=ordered_functionalities or None,
        )


def initialize_script() -> None:
    load_dotenv()
    setup_logging()


def add_repository_selector(parser: argparse.ArgumentParser) -> None:
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--repository-id", type=int, help="Repository id to inspect.")
    selector.add_argument("--repository-name", help="Repository name to inspect.")
    parser.add_argument("--config", default="configs/classification.yaml", help="Classification config path used to read k.")
    parser.add_argument("--mode", choices=("train", "test"), default="train", help="Classification config section used to read k.")
    parser.add_argument("--num-clusters", type=int, help="Override the number of clusters instead of reading classification config.")
    parser.add_argument("--show-full-vector", action="store_true", help="Print full vectors/rows instead of only non-zero values.")
    parser.add_argument("--max-description-chars", type=int, default=180, help="Maximum functionality description characters to print.")


def load_num_clusters(config_path: str, mode: str, override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("--num-clusters must be > 0.")
        return override
    config = ClassificationConfig.from_yaml(mode=cast(Literal["train", "test"], mode), config_path=config_path)
    return config.training.k


def load_labels() -> LabelMapping:
    return LabelMapping.from_split_config()


def load_repository_inspection(*, repository_id: int | None, repository_name: str | None) -> RepositoryInspection:
    with SessionLocal() as session:
        query = (
            session.query(
                RepositoryModel.repository_id,
                RepositoryModel.name,
                RepositoryModel.label,
                RepositoryModel.split,
                RepositoryModel.processed,
                FileModel.file_id,
                FileModel.filepath,
                FileModel.filename,
                FileModel.language,
                SnippetModel.snippet_id,
                SnippetModel.start_line,
                SnippetModel.end_line,
                FunctionalityModel.functionality_id,
                FunctionalityModel.description,
                FunctionalityModel.tag,
                FunctionalityModel.cluster_id,
            )
            .select_from(RepositoryModel)
            .join(FileModel, FileModel.repository_id == RepositoryModel.repository_id, isouter=True)
            .join(SnippetModel, SnippetModel.file_id == FileModel.file_id, isouter=True)
            .join(FunctionalityModel, FunctionalityModel.snippet_id == SnippetModel.snippet_id, isouter=True)
            .order_by(FileModel.filepath, SnippetModel.start_line, FunctionalityModel.functionality_id)
        )
        if repository_id is not None:
            query = query.filter(RepositoryModel.repository_id == repository_id)
        elif repository_name is not None:
            query = query.filter(RepositoryModel.name == repository_name)

        rows = list(query)

    if not rows:
        selector = f"id={repository_id}" if repository_id is not None else f"name={repository_name!r}"
        raise ValueError(f"Repository not found for {selector}.")

    first = rows[0]
    inspection = RepositoryInspection(
        repository_id=int(first.repository_id),
        name=str(first.name),
        label=str(first.label) if first.label is not None else None,
        split=str(first.split) if first.split is not None else None,
        processed=bool(first.processed),
    )
    seen_functionality_ids: set[int] = set()

    for row in rows:
        if row.file_id is None:
            continue

        file_id = int(row.file_id)
        file_inspection = inspection.files.get(file_id)
        if file_inspection is None:
            file_inspection = FileInspection(
                file_id=file_id,
                filepath=str(row.filepath),
                filename=str(row.filename),
                language=str(row.language),
            )
            inspection.files[file_id] = file_inspection

        if row.snippet_id is None:
            continue

        snippet_id = int(row.snippet_id)
        snippet = file_inspection.snippets.get(snippet_id)
        if snippet is None:
            snippet = SnippetInspection(
                snippet_id=snippet_id,
                start_line=int(row.start_line),
                end_line=int(row.end_line),
            )
            file_inspection.snippets[snippet_id] = snippet

        if row.functionality_id is None:
            continue

        functionality_id = int(row.functionality_id)
        if functionality_id in seen_functionality_ids:
            continue
        functionality = FunctionalityInspection(
            functionality_id=functionality_id,
            snippet_id=snippet_id,
            start_line=int(row.start_line),
            end_line=int(row.end_line),
            description=str(row.description),
            tag=str(row.tag),
            cluster_id=int(row.cluster_id) if row.cluster_id is not None else None,
        )
        snippet.functionalities.append(functionality)
        inspection.functionalities.append(functionality)
        seen_functionality_ids.add(functionality_id)

    inspection.functionalities.sort(key=lambda functionality: functionality.functionality_id)
    return inspection


def print_repository_tree(inspection: RepositoryInspection, *, max_description_chars: int) -> None:
    clustered = sum(1 for functionality in inspection.functionalities if functionality.cluster_id is not None)
    print(f"Repository {inspection.repository_id}: {inspection.name}")
    print(f"label={inspection.label} split={inspection.split} processed={inspection.processed}")
    print(f"files={len(inspection.files)} functionalities={len(inspection.functionalities)} clustered={clustered}")
    print()
    print("Files and functionalities")
    if not inspection.files:
        print("  (no files)")
        return
    for file in sorted(inspection.files.values(), key=lambda item: item.filepath):
        print(f"- file_id={file.file_id} {file.filepath} language={file.language}")
        if not file.snippets:
            print("  (no snippets)")
            continue
        for snippet in sorted(file.snippets.values(), key=lambda item: (item.start_line, item.snippet_id)):
            print(f"  snippet_id={snippet.snippet_id} lines={snippet.start_line}-{snippet.end_line}")
            if not snippet.functionalities:
                print("    (no functionalities)")
                continue
            for functionality in sorted(snippet.functionalities, key=lambda item: item.functionality_id):
                description = _truncate(functionality.description, max_description_chars)
                print(f"    functionality_id={functionality.functionality_id} cluster_id={functionality.cluster_id} tag={functionality.tag}")
                print(f"      {description}")


def print_nonzero_vector(vector: np.ndarray, *, prefix: str = "  ") -> None:
    indexes = np.nonzero(vector)[0]
    if len(indexes) == 0:
        print(f"{prefix}(all zeros)")
        return
    for index in indexes:
        print(f"{prefix}cluster_id={int(index)} value={float(vector[index]):.6f}")


def format_vector(vector: Sequence[float | int]) -> str:
    return "[" + ", ".join(_format_number(value) for value in vector) + "]"


def _format_number(value: float) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.6f}".rstrip("0").rstrip(".")


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."
