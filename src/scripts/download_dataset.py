import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from datasets import Dataset, load_dataset
from tqdm import tqdm

from argus.configs import get_logger
from scripts.utils import load_yaml

logger = get_logger(__name__)

CODESEARCHNET_LANGUAGES = ("go", "java", "javascript", "php", "python", "ruby")
CODESEARCHNET_DATASET_ID = "code-search-net/code_search_net"
DEFAULT_REPOSITORIES_PER_LANGUAGE = 100
DEFAULT_SELECTION_SEED = 42
SEPARATOR = "\n\n"


def _normalize_code(source: str) -> str:
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _required_string(entry: Mapping[str, object], field: str, *, language: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CodeSearchNet {language!r} entry has an invalid or missing {field!r} field.")
    return value


def _repository_directory_name(repository_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", repository_name.replace("/", "__")).strip("._")
    if not normalized:
        raise ValueError(f"CodeSearchNet repository name cannot be mapped to a directory: {repository_name!r}.")
    return normalized


def _relative_source_path(filepath: str) -> Path:
    source_path = PurePosixPath(filepath)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError(f"CodeSearchNet source path must be relative and cannot traverse directories: {filepath!r}.")
    return Path(*source_path.parts)


def _selection_key(language: str, repository_name: str, seed: int) -> bytes:
    value = f"{seed}:{language}:{repository_name}".encode()
    return hashlib.sha256(value).digest()


def _load_language_datasets(split: Literal["train", "test"]) -> dict[str, Dataset]:
    datasets_by_language: dict[str, Dataset] = {}
    for language in CODESEARCHNET_LANGUAGES:
        logger.info("Loading CodeSearchNet language split", language=language, split=split)
        try:
            dataset = load_dataset(CODESEARCHNET_DATASET_ID, language, split=split)
        except Exception as error:
            logger.exception("Failed to load CodeSearchNet language split", language=language, split=split, error=str(error))
            raise RuntimeError(f"Failed to load CodeSearchNet {language!r} {split!r} split: {error}") from error
        datasets_by_language[language] = cast(Dataset, dataset)
    return datasets_by_language


def _repository_names_by_language(datasets_by_language: Mapping[str, Dataset]) -> dict[str, set[str]]:
    repository_names: dict[str, set[str]] = {}
    for language, dataset in datasets_by_language.items():
        names = {_required_string(entry, "repository_name", language=language) for entry in tqdm(dataset, desc=f"Indexing {language} repositories")}
        if not names:
            raise RuntimeError(f"CodeSearchNet {language!r} split contains no repositories.")
        repository_names[language] = names
        logger.info("Indexed CodeSearchNet repositories", language=language, repositories=len(names))
    return repository_names


def _exclude_cross_language_repositories(repository_names: Mapping[str, set[str]]) -> tuple[dict[str, set[str]], int]:
    languages_by_repository: defaultdict[str, set[str]] = defaultdict(set)
    for language, names in repository_names.items():
        for repository_name in names:
            languages_by_repository[repository_name].add(language)

    overlapping_names = {name for name, languages in languages_by_repository.items() if len(languages) > 1}
    eligible_names = {language: names - overlapping_names for language, names in repository_names.items()}
    return eligible_names, len(overlapping_names)


def _select_balanced_repositories(
    repository_names: Mapping[str, set[str]],
    *,
    repositories_per_language: int,
    seed: int,
) -> dict[str, set[str]]:
    if repositories_per_language <= 0:
        raise ValueError("repositories_per_language must be greater than zero.")

    smallest_language_count = min(len(names) for names in repository_names.values())
    if smallest_language_count == 0:
        unavailable_languages = sorted(language for language, names in repository_names.items() if not names)
        raise RuntimeError(f"No eligible repositories remain for languages: {', '.join(unavailable_languages)}.")

    balanced_count = min(repositories_per_language, smallest_language_count)
    if balanced_count < repositories_per_language:
        logger.warning(
            "Reducing repository quota to preserve language balance",
            requested_per_language=repositories_per_language,
            selected_per_language=balanced_count,
        )

    return {language: set(sorted(names, key=lambda name: _selection_key(language, name, seed))[:balanced_count]) for language, names in repository_names.items()}


def _validate_output_directories(split_dir: Path, selected_names: Mapping[str, set[str]]) -> dict[str, Path]:
    output_directories: dict[str, Path] = {}
    directory_owners: dict[str, str] = {}
    existing_directories: list[Path] = []

    for names in selected_names.values():
        for repository_name in names:
            directory_name = _repository_directory_name(repository_name)
            previous_owner = directory_owners.get(directory_name)
            if previous_owner is not None and previous_owner != repository_name:
                raise ValueError(f"CodeSearchNet repository directory collision between {previous_owner!r} and {repository_name!r}: {directory_name!r}.")
            directory_owners[directory_name] = repository_name
            output_directory = split_dir / directory_name
            output_directories[repository_name] = output_directory
            if output_directory.exists():
                existing_directories.append(output_directory)

    if existing_directories:
        examples = ", ".join(str(path) for path in sorted(existing_directories)[:5])
        raise FileExistsError(f"Refusing to append to existing selected repository directories. Move or remove the previous export first. Examples: {examples}")
    return output_directories


def _write_selected_repositories(
    datasets_by_language: Mapping[str, Dataset],
    selected_names: Mapping[str, set[str]],
    output_directories: Mapping[str, Path],
) -> dict[str, int]:
    written_repositories: dict[str, set[str]] = {language: set() for language in datasets_by_language}

    for language, dataset in datasets_by_language.items():
        language_selection = selected_names[language]
        for entry in tqdm(dataset, desc=f"Writing {language} repositories"):
            repository_name = _required_string(entry, "repository_name", language=language)
            if repository_name not in language_selection:
                continue

            filepath = _required_string(entry, "func_path_in_repository", language=language)
            raw_code = _required_string(entry, "whole_func_string", language=language)
            output_path = output_directories[repository_name] / _relative_source_path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
                    output_file.write(_normalize_code(raw_code))
                    output_file.write(SEPARATOR)
            except OSError as error:
                raise RuntimeError(f"Failed to write CodeSearchNet source file {output_path}.") from error
            written_repositories[language].add(repository_name)

    counts = {language: len(names) for language, names in written_repositories.items()}
    incomplete_languages = [language for language, names in selected_names.items() if counts[language] != len(names)]
    if incomplete_languages:
        raise RuntimeError(f"Selected repositories were not fully written for languages: {', '.join(sorted(incomplete_languages))}.")
    return counts


def download_benign_dataset(
    split: Literal["train", "test"] = "train",
    repositories_per_language: int = DEFAULT_REPOSITORIES_PER_LANGUAGE,
    seed: int = DEFAULT_SELECTION_SEED,
) -> None:
    """Download a repository-balanced, multilingual CodeSearchNet subset."""
    config = load_yaml()
    configured_dataset_path = config.get("paths", {}).get("dataset")
    if not isinstance(configured_dataset_path, str) or not configured_dataset_path.strip():
        raise ValueError("`paths.dataset` is not defined in the YAML config.")
    if repositories_per_language <= 0:
        raise ValueError("repositories_per_language must be greater than zero.")

    datasets_by_language = _load_language_datasets(split)
    repository_names = _repository_names_by_language(datasets_by_language)
    eligible_names, overlapping_count = _exclude_cross_language_repositories(repository_names)
    selected_names = _select_balanced_repositories(
        eligible_names,
        repositories_per_language=repositories_per_language,
        seed=seed,
    )
    logger.info(
        "Selected balanced CodeSearchNet repositories",
        available_repositories={language: len(names) for language, names in repository_names.items()},
        eligible_repositories={language: len(names) for language, names in eligible_names.items()},
        selected_repositories={language: len(names) for language, names in selected_names.items()},
        excluded_cross_language_repositories=overlapping_count,
        seed=seed,
    )

    split_dir = Path(configured_dataset_path).expanduser() / "benign"
    output_directories = _validate_output_directories(split_dir, selected_names)
    split_dir.mkdir(parents=True, exist_ok=True)
    written_counts = _write_selected_repositories(datasets_by_language, selected_names, output_directories)

    logger.info(
        "Completed balanced CodeSearchNet download",
        split=split,
        seed=seed,
        excluded_cross_language_repositories=overlapping_count,
        repositories_per_language=min(written_counts.values()),
        written_repositories=written_counts,
        output_path=str(split_dir),
    )
