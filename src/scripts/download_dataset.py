import os
import re
from typing import Literal, Optional

from datasets import load_dataset
from tqdm import tqdm

from sastllm.configs import get_logger
from scripts.utils import load_yaml

logger = get_logger(__name__)


# Seperator
SEP = "\n\n"


def _normalize_code(src: str) -> str:
    s = src.replace("\r\n", "\n").replace("\r", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s


def _line_count(s: str) -> int:
    # s must be normalized and end with '\n'
    return s.count("\n")


def normalize_string(s: str) -> str:
    """
    Normalize a string by removing newlines and tabs,
    and collapsing multiple spaces into a single space.
    """
    # Replace newlines and tabs with spaces
    s = s.replace("\n", " ").replace("\t", " ")
    # Collapse multiple spaces into one
    s = re.sub(r"\s+", " ", s)
    # Trim leading/trailing spaces
    return s.strip()


def download_benign_dataset(name: Literal["train", "test"] = "train") -> None:
    config = load_yaml()
    database_path: Optional[str] = config.get("paths", {}).get("dataset")

    if not database_path:
        msg = "`paths.dataset` is not defined in the YAML config."
        logger.error(msg)
        raise ValueError(msg)

    dataset = load_dataset("code_search_net", "all", split=name, trust_remote_code=True)

    split_dir = os.path.join(database_path, "benign")
    os.makedirs(split_dir, exist_ok=True)

    # Normalize separator and precompute line count
    sep = SEP
    if not sep.endswith("\n"):
        sep += "\n"

    for entry in tqdm(dataset, desc=f"Downloading {name} split"):
        repo_full = entry["repository_name"]  # e.g. "owner/repo"
        _owner, repo = repo_full.split("/", 1)  # owner, repo
        filepath = entry["func_path_in_repository"]  # e.g. "repo/file"
        raw_code = entry["whole_func_string"]
        functionality = entry["func_documentation_string"]

        _normalized_functionality = normalize_string(functionality)

        normalized_code = _normalize_code(raw_code)
        code_lines = _line_count(normalized_code)

        full_path = os.path.join(split_dir, repo, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            # Count existing lines already written to this file
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    existing_lines = _line_count(f.read().replace("\r\n", "\n").replace("\r", "\n"))
            else:
                existing_lines = 0

            _start_line = existing_lines + 1
            _end_line = existing_lines + code_lines  # inclusive

            # Append code + separator
            with open(full_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(normalized_code)
                f.write(sep)

        except Exception as e:
            raise RuntimeError(f"Failed to write {full_path}: {e}") from e
